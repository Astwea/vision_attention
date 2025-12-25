"""Probabilistic attention routing models for resource allocation experiments.

This module implements:
1. ProbabilisticAttentionModel: Proposed method with probabilistic routing
2. BaselineLargeMLP: All patches use Large MLP
3. BaselineSmallMLP: All patches use Small MLP
4. DeterministicAttentionModel: Standard Transformer-style attention (continuous weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .probabilistic_router import ProbabilisticRouter


# Backbone encoder
class LightweightEncoder(nn.Module):
    """Lightweight convolutional encoder for feature extraction."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# Feature aggregator
class FeatureAggregator(nn.Module):
    """Aggregate patch features into global representation."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_patches: int,
        mode: str = "concat",
    ):
        super().__init__()
        self.mode = mode
        self.num_patches = num_patches
        
        if mode == "concat":
            self.proj = nn.Linear(input_dim * num_patches, output_dim)
        elif mode == "mean":
            self.proj = nn.Linear(input_dim, output_dim)
        elif mode == "weighted_mean":
            self.proj = nn.Linear(input_dim, output_dim)
            self.weights = nn.Parameter(torch.ones(num_patches) / num_patches)
        else:
            raise ValueError(f"Unknown aggregator mode: {mode}")
    
    def forward(
        self,
        patch_features: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        routing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            patch_features: (B, N_patches, D)
            attention_scores: Optional (B, N_patches)
            routing_mask: Optional (B, N_patches)
        """
        B, N, D = patch_features.shape
        
        if self.mode == "concat":
            features = patch_features.view(B, -1)
        elif self.mode == "mean":
            features = patch_features.mean(dim=1)
        elif self.mode == "weighted_mean":
            if attention_scores is not None:
                weights = F.softmax(attention_scores, dim=1)  # (B, N)
                weights = weights.unsqueeze(-1)  # (B, N, 1)
                features = (patch_features * weights).sum(dim=1)
            else:
                features = patch_features.mean(dim=1)
        else:
            features = patch_features.mean(dim=1)
        
        return self.proj(features)
    
    def forward_split(
        self,
        high_features: torch.Tensor,
        low_features: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        routing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate split features (high + low attention)."""
        # Combine features
        patch_features = high_features + low_features
        return self.forward(patch_features, attention_scores, routing_mask)


class SimpleMLP(nn.Module):
    """Simple MLP for patch processing."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.mlp(x)
    
    def forward_with_mask(
        self,
        patches: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with routing mask.
        
        Args:
            patches: (B, N_patches, D)
            mask: (B, N_patches) - 1 for this MLP, 0 for other MLP
            
        Returns:
            embeddings: (B, N_patches, output_dim)
        """
        B, N, D = patches.shape
        output_dim = self.mlp[-1].out_features
        
        # Process all patches
        patches_flat = patches.view(-1, D)  # (B*N, D)
        embeddings_flat = self.mlp(patches_flat)  # (B*N, output_dim)
        embeddings = embeddings_flat.view(B, N, output_dim)  # (B, N, output_dim)
        
        # Apply mask: only keep embeddings where mask=1
        mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
        embeddings = embeddings * mask_expanded
        
        return embeddings


class ProbabilisticAttentionHead(nn.Module):
    """
    Attention head that outputs probability values p ∈ (0,1) for each patch.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_patches: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
    ):
        """
        Args:
            feature_dim: Dimension of backbone features
            num_patches: Number of patches
            hidden_dim: Hidden dimension for attention network
            num_layers: Number of layers in attention network
        """
        super().__init__()
        self.num_patches = num_patches
        
        # Build attention network
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(feature_dim, num_patches))
        else:
            layers.append(nn.Linear(feature_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, num_patches))
        
        self.attention_net = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Backbone features (B, C, H', W')
            
        Returns:
            attention_scores: Attention scores (B, N_patches)
                These will be converted to probabilities via sigmoid in router
        """
        B, C, H, W = features.shape
        
        # Global average pooling
        pooled = F.adaptive_avg_pool2d(features, (1, 1))  # (B, C, 1, 1)
        pooled = pooled.view(B, C)  # (B, C)
        
        # Compute attention scores (raw logits, will be sigmoided in router)
        attention_scores = self.attention_net(pooled)  # (B, N_patches)
        
        return attention_scores


class ProbabilisticAttentionModel(nn.Module):
    """
    Proposed probabilistic attention routing model.
    
    Attention outputs probability values, which are used to randomly sample
    routing decisions (Large MLP vs Small MLP) during training.
    """
    
    def __init__(
        self,
        # Backbone
        in_channels: int = 3,
        backbone_channels: int = 64,
        backbone_layers: int = 2,
        # Attention
        num_patches: int = 64,
        attention_hidden_dim: int = 32,
        attention_layers: int = 1,
        # Router
        router_mode: str = "gumbel_softmax",
        temperature: float = 1.0,
        threshold: float = 0.5,
        # MLPs
        patch_input_dim: int = 192,  # 3*8*8 for 8x8 patches
        mlp_big_hidden_dim: int = 512,
        mlp_big_output_dim: int = 128,
        mlp_big_layers: int = 3,
        mlp_small_hidden_dim: int = 64,
        mlp_small_output_dim: int = 128,
        mlp_small_layers: int = 1,
        # Aggregator
        aggregator_mode: str = "concat",
        aggregator_output_dim: int = 256,
        # Task head
        num_classes: int = 10,
        task_head_hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.num_patches = num_patches
        
        # Backbone encoder
        self.backbone = LightweightEncoder(
            in_channels=in_channels,
            out_channels=backbone_channels,
            num_layers=backbone_layers,
        )
        
        # Probabilistic attention head
        self.attention = ProbabilisticAttentionHead(
            feature_dim=backbone_channels,
            num_patches=num_patches,
            hidden_dim=attention_hidden_dim,
            num_layers=attention_layers,
        )
        
        # Probabilistic router
        self.router = ProbabilisticRouter(
            mode=router_mode,
            temperature=temperature,
            threshold=threshold,
        )
        
        # MLPs
        self.mlp_big = SimpleMLP(
            input_dim=patch_input_dim,
            hidden_dim=mlp_big_hidden_dim,
            output_dim=mlp_big_output_dim,
            num_layers=mlp_big_layers,
        )
        
        self.mlp_small = SimpleMLP(
            input_dim=patch_input_dim,
            hidden_dim=mlp_small_hidden_dim,
            output_dim=mlp_small_output_dim,
            num_layers=mlp_small_layers,
        )
        
        # Aggregator
        self.aggregator = FeatureAggregator(
            input_dim=mlp_big_output_dim,  # Should be same for both MLPs
            output_dim=aggregator_output_dim,
            num_patches=num_patches,
            mode=aggregator_mode,
        )
        
        # Task head (classification)
        self.task_head = nn.Sequential(
            nn.Linear(aggregator_output_dim, task_head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(task_head_hidden_dim, num_classes),
        )
    
    def forward(
        self,
        images: torch.Tensor,
        patches: torch.Tensor,
        return_attention: bool = False,
        return_stats: bool = False,
        inference_mode: str = "sampling",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with probabilistic routing.
        
        Args:
            images: Original images (B, C, H, W)
            patches: Extracted patches (B, N_patches, C, patch_H, patch_W)
            return_attention: If True, return attention scores
            return_stats: If True, return routing statistics
            inference_mode: Inference mode ("sampling", "expectation", "top_p", "hard_threshold")
            
        Returns:
            Dictionary with logits and optional attention/routing info
        """
        B = images.size(0)
        
        # 1. Encode with backbone
        features = self.backbone(images)  # (B, F, H', W')
        
        # 2. Compute attention scores (raw logits)
        attention_scores = self.attention(features)  # (B, N_patches)
        
        # 3. Probabilistic routing
        routing_mask, router_stats = self.router(
            attention_scores,
            return_stats=True,
            inference_mode=inference_mode,
        )  # (B, N_patches)
        
        # 4. Flatten patches
        patches_flat = patches.view(B, self.num_patches, -1)  # (B, N_patches, D)
        
        # 5. Conditional computation
        # Patches with routing_mask=1 → Large MLP
        # Patches with routing_mask=0 → Small MLP
        large_embeddings = self.mlp_big.forward_with_mask(patches_flat, routing_mask)
        small_embeddings = self.mlp_small.forward_with_mask(patches_flat, 1 - routing_mask)
        
        # Combine embeddings
        patch_embeddings = large_embeddings + small_embeddings  # (B, N_patches, output_dim)
        
        # 6. Aggregate features
        global_features = self.aggregator(patch_embeddings)  # (B, aggregator_output_dim)
        
        # 7. Classification
        logits = self.task_head(global_features)  # (B, num_classes)
        
        result = {'logits': logits}
        
        if return_attention or return_stats:
            result['attention_scores'] = attention_scores
            result['routing_mask'] = routing_mask
            result['router_stats'] = router_stats
        
        return result


class BaselineLargeMLP(nn.Module):
    """Baseline: All patches processed by Large MLP."""
    
    def __init__(
        self,
        in_channels: int = 3,
        backbone_channels: int = 64,
        backbone_layers: int = 2,
        num_patches: int = 64,
        patch_input_dim: int = 192,
        mlp_hidden_dim: int = 512,
        mlp_output_dim: int = 128,
        mlp_layers: int = 3,
        aggregator_mode: str = "concat",
        aggregator_output_dim: int = 256,
        num_classes: int = 10,
        task_head_hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.num_patches = num_patches
        
        self.backbone = LightweightEncoder(
            in_channels=in_channels,
            out_channels=backbone_channels,
            num_layers=backbone_layers,
        )
        
        self.mlp = SimpleMLP(
            input_dim=patch_input_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=mlp_output_dim,
            num_layers=mlp_layers,
        )
        
        self.aggregator = FeatureAggregator(
            input_dim=mlp_output_dim,
            output_dim=aggregator_output_dim,
            num_patches=num_patches,
            mode=aggregator_mode,
        )
        
        self.task_head = nn.Sequential(
            nn.Linear(aggregator_output_dim, task_head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(task_head_hidden_dim, num_classes),
        )
    
    def forward(
        self,
        images: torch.Tensor,
        patches: torch.Tensor,
        return_attention: bool = False,
        return_stats: bool = False,
        inference_mode: str = "sampling",
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        B = images.size(0)
        _ = self.backbone(images)
        
        # Process all patches with Large MLP
        patches_flat = patches.view(B, self.num_patches, -1)
        embeddings = self.mlp(patches_flat.view(-1, patches_flat.size(-1)))
        embeddings = embeddings.view(B, self.num_patches, -1)
        
        global_features = self.aggregator(embeddings)
        logits = self.task_head(global_features)
        
        return {'logits': logits}


class BaselineSmallMLP(nn.Module):
    """Baseline: All patches processed by Small MLP."""
    
    def __init__(
        self,
        in_channels: int = 3,
        backbone_channels: int = 64,
        backbone_layers: int = 2,
        num_patches: int = 64,
        patch_input_dim: int = 192,
        mlp_hidden_dim: int = 64,
        mlp_output_dim: int = 128,
        mlp_layers: int = 1,
        aggregator_mode: str = "concat",
        aggregator_output_dim: int = 256,
        num_classes: int = 10,
        task_head_hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.num_patches = num_patches
        
        self.backbone = LightweightEncoder(
            in_channels=in_channels,
            out_channels=backbone_channels,
            num_layers=backbone_layers,
        )
        
        self.mlp = SimpleMLP(
            input_dim=patch_input_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=mlp_output_dim,
            num_layers=mlp_layers,
        )
        
        self.aggregator = FeatureAggregator(
            input_dim=mlp_output_dim,
            output_dim=aggregator_output_dim,
            num_patches=num_patches,
            mode=aggregator_mode,
        )
        
        self.task_head = nn.Sequential(
            nn.Linear(aggregator_output_dim, task_head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(task_head_hidden_dim, num_classes),
        )
    
    def forward(
        self,
        images: torch.Tensor,
        patches: torch.Tensor,
        return_attention: bool = False,
        return_stats: bool = False,
        inference_mode: str = "sampling",
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        B = images.size(0)
        _ = self.backbone(images)
        
        # Process all patches with Small MLP
        patches_flat = patches.view(B, self.num_patches, -1)
        embeddings = self.mlp(patches_flat)
        
        global_features = self.aggregator(embeddings)
        logits = self.task_head(global_features)
        
        return {'logits': logits}


class DeterministicAttentionModel(nn.Module):
    """
    Baseline: Standard Transformer-style attention (continuous weights).
    
    Attention is used as continuous weights to combine Large and Small MLP outputs,
    rather than as probabilities for routing decisions.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        backbone_channels: int = 64,
        backbone_layers: int = 2,
        num_patches: int = 64,
        attention_hidden_dim: int = 32,
        attention_layers: int = 1,
        patch_input_dim: int = 192,
        mlp_big_hidden_dim: int = 512,
        mlp_big_output_dim: int = 128,
        mlp_big_layers: int = 3,
        mlp_small_hidden_dim: int = 64,
        mlp_small_output_dim: int = 128,
        mlp_small_layers: int = 1,
        aggregator_mode: str = "concat",
        aggregator_output_dim: int = 256,
        num_classes: int = 10,
        task_head_hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.num_patches = num_patches
        
        # Backbone
        self.backbone = LightweightEncoder(
            in_channels=in_channels,
            out_channels=backbone_channels,
            num_layers=backbone_layers,
        )
        
        # Attention head (outputs continuous weights)
        self.attention = ProbabilisticAttentionHead(
            feature_dim=backbone_channels,
            num_patches=num_patches,
            hidden_dim=attention_hidden_dim,
            num_layers=attention_layers,
        )
        
        # MLPs
        self.mlp_big = SimpleMLP(
            input_dim=patch_input_dim,
            hidden_dim=mlp_big_hidden_dim,
            output_dim=mlp_big_output_dim,
            num_layers=mlp_big_layers,
        )
        
        self.mlp_small = SimpleMLP(
            input_dim=patch_input_dim,
            hidden_dim=mlp_small_hidden_dim,
            output_dim=mlp_small_output_dim,
            num_layers=mlp_small_layers,
        )
        
        # Aggregator
        self.aggregator = FeatureAggregator(
            input_dim=mlp_big_output_dim,
            output_dim=aggregator_output_dim,
            num_patches=num_patches,
            mode=aggregator_mode,
        )
        
        # Task head
        self.task_head = nn.Sequential(
            nn.Linear(aggregator_output_dim, task_head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(task_head_hidden_dim, num_classes),
        )
    
    def forward(
        self,
        images: torch.Tensor,
        patches: torch.Tensor,
        return_attention: bool = False,
        return_stats: bool = False,
        inference_mode: str = "sampling",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with deterministic attention weighting.
        
        Attention scores are used as continuous weights to combine
        Large and Small MLP outputs.
        """
        B = images.size(0)
        
        # 1. Encode with backbone
        features = self.backbone(images)
        
        # 2. Compute attention scores
        attention_scores = self.attention(features)  # (B, N_patches)
        
        # 3. Convert to attention weights (softmax for normalization)
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, N_patches)
        
        # 4. Process all patches with both MLPs
        patches_flat = patches.view(B, self.num_patches, -1)
        large_embeddings = self.mlp_big(patches_flat.view(-1, patches_flat.size(-1)))
        large_embeddings = large_embeddings.view(B, self.num_patches, -1)
        
        small_embeddings = self.mlp_small(patches_flat)
        
        # 5. Weighted combination using attention
        # Higher attention → more weight on Large MLP output
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (B, N_patches, 1)
        patch_embeddings = (
            attention_weights_expanded * large_embeddings +
            (1 - attention_weights_expanded) * small_embeddings
        )  # (B, N_patches, output_dim)
        
        # 6. Aggregate features
        global_features = self.aggregator(patch_embeddings)
        
        # 7. Classification
        logits = self.task_head(global_features)
        
        result = {'logits': logits}
        
        if return_attention or return_stats:
            result['attention_scores'] = attention_scores
            result['attention_weights'] = attention_weights
            # Compute effective routing ratio (weighted average)
            effective_ratio = attention_weights.mean().item()
            result['router_stats'] = {
                'routing_ratio': effective_ratio,
                'probs_mean': attention_weights.mean().item(),
                'probs_std': attention_weights.std().item(),
            }
        
        return result


def create_probabilistic_model(
    model_name: str,
    num_patches: int = 64,
    patch_input_dim: int = 192,
    num_classes: int = 10,
    router_mode: str = "gumbel_softmax",
    **kwargs
) -> nn.Module:
    """
    Factory function to create probabilistic models.
    
    Args:
        model_name: "probabilistic", "baseline_large", "baseline_small", "deterministic"
        num_patches: Number of patches
        patch_input_dim: Input dimension per patch
        num_classes: Number of classes
        router_mode: Router mode for probabilistic model
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    common_args = {
        'num_patches': num_patches,
        'patch_input_dim': patch_input_dim,
        'num_classes': num_classes,
        **kwargs
    }
    
    if model_name == "probabilistic":
        return ProbabilisticAttentionModel(
            router_mode=router_mode,
            **common_args
        )
    elif model_name == "baseline_large":
        return BaselineLargeMLP(**common_args)
    elif model_name == "baseline_small":
        return BaselineSmallMLP(**common_args)
    elif model_name == "deterministic":
        return DeterministicAttentionModel(**common_args)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

