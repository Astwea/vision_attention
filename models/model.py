"""Complete model assembly with attention routing and baselines."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .backbone import LightweightEncoder
from .attention import AttentionHead
from .router import Router
from .mlp_big import LargeMLP
from .mlp_small import SmallMLP
from .aggregator import FeatureAggregator
from .task_heads.classification_head import ClassificationHead
from .task_heads.detection_head import DetectionHead
from .task_heads.segmentation_head import SegmentationHead, InstanceSegmentationHead


class TaskHead(nn.Module):
    """Task head for classification."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AttentionRoutingModel(nn.Module):
    """
    Main model with attention-guided hard routing.
    
    Pipeline:
    1. Lightweight encoder extracts features
    2. Attention head predicts attention scores
    3. Router makes hard routing decisions
    4. High-attention patches → Large MLP (expensive)
    5. Low-attention patches → Small MLP (cheap)
    6. Aggregate and classify
    """
    
    def __init__(
        self,
        # Backbone
        in_channels: int = 3,
        backbone_channels: int = 64,
        backbone_layers: int = 2,
        # Attention
        num_patches: int = 16,
        attention_hidden_dim: int = 32,
        attention_layers: int = 1,
        # Router
        router_mode: str = "learnable_threshold",
        threshold_init: float = 0.5,
        temperature: float = 1.0,
        # MLPs
        patch_input_dim: int = 3072,  # 3*32*32 for CIFAR
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
        
        # Attention head
        self.attention = AttentionHead(
            feature_dim=backbone_channels,
            num_patches=num_patches,
            hidden_dim=attention_hidden_dim,
            num_layers=attention_layers,
        )
        
        # Router
        self.router = Router(
            mode=router_mode,
            threshold_init=threshold_init,
            temperature=temperature,
        )
        
        # MLPs
        self.mlp_big = LargeMLP(
            input_dim=patch_input_dim,
            hidden_dim=mlp_big_hidden_dim,
            output_dim=mlp_big_output_dim,
            num_layers=mlp_big_layers,
        )
        
        self.mlp_small = SmallMLP(
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
        self.task_head = TaskHead(
            input_dim=aggregator_output_dim,
            num_classes=num_classes,
            hidden_dim=task_head_hidden_dim,
        )
    
    def forward(
        self,
        images: torch.Tensor,
        patches: torch.Tensor,
        return_attention: bool = False,
        return_stats: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with conditional computation.
        
        Args:
            images: Original images (B, C, H, W)
            patches: Extracted patches (B, N_patches, C, patch_H, patch_W)
            return_attention: If True, return attention scores and routing stats
            return_stats: If True, return routing statistics
            
        Returns:
            Dictionary containing:
                - logits: Classification logits (B, num_classes)
                - attention_scores: (Optional) attention scores (B, N_patches)
                - routing_mask: (Optional) routing mask (B, N_patches)
                - stats: (Optional) routing statistics
        """
        B = images.size(0)
        
        # 1. Encode with lightweight backbone
        features = self.backbone(images)  # (B, F, H', W')
        
        # 2. Compute attention scores
        attention_scores = self.attention(features)  # (B, N_patches)
        
        # 3. Make routing decisions
        routing_mask, router_stats = self.router(attention_scores, return_stats=True)
        
        # 4. Flatten patches for MLP processing
        # patches: (B, N_patches, C, patch_H, patch_W)
        patch_shape = patches.shape
        patches_flat = patches.view(B, self.num_patches, -1)  # (B, N_patches, D)
        
        # 5. Conditional computation: Route patches
        # High-attention patches → Large MLP
        high_attention_embeddings = self.mlp_big.forward_with_mask(
            patches_flat, routing_mask
        )  # (B, N_patches, output_dim)
        
        # Low-attention patches → Small MLP
        low_attention_embeddings = self.mlp_small.forward_with_mask(
            patches_flat, routing_mask
        )  # (B, N_patches, output_dim)
        
        # 6. Aggregate features
        global_features = self.aggregator.forward_split(
            high_attention_embeddings,
            low_attention_embeddings,
            attention_scores=attention_scores,
            routing_mask=routing_mask,
        )  # (B, aggregator_output_dim)
        
        # 7. Classification
        logits = self.task_head(global_features)  # (B, num_classes)
        
        # Prepare return dict
        result = {'logits': logits}
        
        if return_attention or return_stats:
            result['attention_scores'] = attention_scores
            result['routing_mask'] = routing_mask
            result['router_stats'] = router_stats
        
        return result


class FullComputeBaseline(nn.Module):
    """Baseline: All patches processed by Large MLP."""
    
    def __init__(
        self,
        in_channels: int = 3,
        backbone_channels: int = 64,
        backbone_layers: int = 2,
        num_patches: int = 16,
        patch_input_dim: int = 3072,
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
        self.mlp = LargeMLP(
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
        self.task_head = TaskHead(
            input_dim=aggregator_output_dim,
            num_classes=num_classes,
            hidden_dim=task_head_hidden_dim,
        )
    
    def forward(self, images: torch.Tensor, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = images.size(0)
        _ = self.backbone(images)  # Use backbone for consistency, but don't need output
        
        # Process all patches with Large MLP
        patches_flat = patches.view(B, self.num_patches, -1)
        embeddings = self.mlp(patches_flat.view(-1, patches_flat.size(-1)))
        embeddings = embeddings.view(B, self.num_patches, -1)
        
        global_features = self.aggregator(embeddings)
        logits = self.task_head(global_features)
        
        return {'logits': logits}


class CheapComputeBaseline(nn.Module):
    """Baseline: All patches processed by Small MLP."""
    
    def __init__(
        self,
        in_channels: int = 3,
        backbone_channels: int = 64,
        backbone_layers: int = 2,
        num_patches: int = 16,
        patch_input_dim: int = 3072,
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
        self.mlp = SmallMLP(
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
        self.task_head = TaskHead(
            input_dim=aggregator_output_dim,
            num_classes=num_classes,
            hidden_dim=task_head_hidden_dim,
        )
    
    def forward(self, images: torch.Tensor, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = images.size(0)
        _ = self.backbone(images)
        
        # Process all patches with Small MLP
        patches_flat = patches.view(B, self.num_patches, -1)
        embeddings = self.mlp(patches_flat)
        
        global_features = self.aggregator(embeddings)
        logits = self.task_head(global_features)
        
        return {'logits': logits}


class NoRoutingBaseline(nn.Module):
    """Baseline: Same backbone without attention gating, use Small MLP."""
    
    def __init__(
        self,
        in_channels: int = 3,
        backbone_channels: int = 64,
        backbone_layers: int = 2,
        num_patches: int = 16,
        patch_input_dim: int = 3072,
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
        self.mlp = SmallMLP(
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
        self.task_head = TaskHead(
            input_dim=aggregator_output_dim,
            num_classes=num_classes,
            hidden_dim=task_head_hidden_dim,
        )
    
    def forward(self, images: torch.Tensor, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = images.size(0)
        _ = self.backbone(images)
        
        # Process all patches with Small MLP (no routing)
        patches_flat = patches.view(B, self.num_patches, -1)
        embeddings = self.mlp(patches_flat)
        
        global_features = self.aggregator(embeddings)
        logits = self.task_head(global_features)
        
        return {'logits': logits}


def create_model(model_name: str, config: Dict, task_type: Optional[str] = None) -> nn.Module:
    """Factory function to create models based on config and task type."""
    
    model_config = config.get('model', {})
    dataset_config = config.get('dataset', {})
    
    # Get task type from config if not provided
    if task_type is None:
        task_type = config.get('task_type', 'classification')
    task_type = task_type.lower()
    
    # Common arguments
    patch_grid_size = dataset_config.get('patch_grid_size', 4)
    input_size = dataset_config.get('input_size', 32)
    num_patches = patch_grid_size ** 2
    patch_input_dim = (input_size // patch_grid_size) ** 2 * 3
    
    # Get num_classes from dataset info or config
    num_classes = dataset_config.get('num_classes')
    if num_classes is None:
        # Try to infer from dataset name
        dataset_name = dataset_config.get('name', 'cifar10')
        if 'cifar10' in dataset_name.lower():
            num_classes = 10
        elif 'cifar100' in dataset_name.lower():
            num_classes = 100
        elif 'imagenet' in dataset_name.lower():
            num_classes = 1000
        elif 'coco' in dataset_name.lower():
            num_classes = 80
        elif 'voc' in dataset_name.lower():
            num_classes = 20 if task_type == 'detection' else 21
        else:
            num_classes = 10  # default
    
    common_args = {
        'in_channels': 3,
        'backbone_channels': model_config.get('backbone', {}).get('out_channels', 64),
        'backbone_layers': model_config.get('backbone', {}).get('num_layers', 2),
        'num_patches': num_patches,
        'patch_input_dim': patch_input_dim,
        'aggregator_mode': model_config.get('aggregator', {}).get('mode', 'concat'),
        'aggregator_output_dim': model_config.get('aggregator', {}).get('output_dim', 256),
        'num_classes': num_classes,
    }
    
    # Task-specific model creation
    if task_type == "classification":
        common_args['task_head_hidden_dim'] = model_config.get('task_head', {}).get('hidden_dim', 128)
        
        if model_name == "attention_routing":
            return AttentionRoutingModel(
                **common_args,
                attention_hidden_dim=model_config.get('attention', {}).get('hidden_dim', 32),
                attention_layers=model_config.get('attention', {}).get('num_layers', 1),
                router_mode=model_config.get('router', {}).get('mode', 'learnable_threshold'),
                threshold_init=model_config.get('router', {}).get('threshold_init', 0.5),
                temperature=model_config.get('router', {}).get('temperature', 1.0),
                mlp_big_hidden_dim=model_config.get('mlp_big', {}).get('hidden_dim', 512),
                mlp_big_output_dim=model_config.get('mlp_big', {}).get('output_dim', 128),
                mlp_big_layers=model_config.get('mlp_big', {}).get('num_layers', 3),
                mlp_small_hidden_dim=model_config.get('mlp_small', {}).get('hidden_dim', 64),
                mlp_small_output_dim=model_config.get('mlp_small', {}).get('output_dim', 128),
                mlp_small_layers=model_config.get('mlp_small', {}).get('num_layers', 1),
            )
        elif model_name == "full_compute":
            return FullComputeBaseline(
                **common_args,
                mlp_hidden_dim=model_config.get('mlp_big', {}).get('hidden_dim', 512),
                mlp_output_dim=model_config.get('mlp_big', {}).get('output_dim', 128),
                mlp_layers=model_config.get('mlp_big', {}).get('num_layers', 3),
            )
        elif model_name == "cheap_compute":
            return CheapComputeBaseline(
                **common_args,
                mlp_hidden_dim=model_config.get('mlp_small', {}).get('hidden_dim', 64),
                mlp_output_dim=model_config.get('mlp_small', {}).get('output_dim', 128),
                mlp_layers=model_config.get('mlp_small', {}).get('num_layers', 1),
            )
        elif model_name == "no_routing":
            return NoRoutingBaseline(
                **common_args,
                mlp_hidden_dim=model_config.get('mlp_small', {}).get('hidden_dim', 64),
                mlp_output_dim=model_config.get('mlp_small', {}).get('output_dim', 128),
                mlp_layers=model_config.get('mlp_small', {}).get('num_layers', 1),
            )
    
    elif task_type == "detection":
        detection_head_config = model_config.get('detection_head', {})
        common_args['detection_hidden_dim'] = detection_head_config.get('hidden_dim', 256)
        common_args['num_anchors'] = detection_head_config.get('num_anchors', 1)
        
        return AttentionRoutingDetector(
            **common_args,
            attention_hidden_dim=model_config.get('attention', {}).get('hidden_dim', 32),
            attention_layers=model_config.get('attention', {}).get('num_layers', 1),
            router_mode=model_config.get('router', {}).get('mode', 'learnable_threshold'),
            threshold_init=model_config.get('router', {}).get('threshold_init', 0.5),
            temperature=model_config.get('router', {}).get('temperature', 1.0),
            mlp_big_hidden_dim=model_config.get('mlp_big', {}).get('hidden_dim', 512),
            mlp_big_output_dim=model_config.get('mlp_big', {}).get('output_dim', 128),
            mlp_big_layers=model_config.get('mlp_big', {}).get('num_layers', 3),
            mlp_small_hidden_dim=model_config.get('mlp_small', {}).get('hidden_dim', 64),
            mlp_small_output_dim=model_config.get('mlp_small', {}).get('output_dim', 128),
            mlp_small_layers=model_config.get('mlp_small', {}).get('num_layers', 1),
        )
    
    elif task_type in ["segmentation", "instance_segmentation"]:
        segmentation_head_config = model_config.get('segmentation_head', {})
        common_args['patch_grid_size'] = patch_grid_size
        common_args['segmentation_hidden_dim'] = segmentation_head_config.get('hidden_dim', 256)
        common_args['instance_seg'] = (task_type == "instance_segmentation")
        
        # For segmentation, num_classes includes background
        if 'voc' in dataset_config.get('name', '').lower() and task_type == "segmentation":
            common_args['num_classes'] = 21  # VOC semantic segmentation has 21 classes
        
        return AttentionRoutingSegmenter(
            **common_args,
            attention_hidden_dim=model_config.get('attention', {}).get('hidden_dim', 32),
            attention_layers=model_config.get('attention', {}).get('num_layers', 1),
            router_mode=model_config.get('router', {}).get('mode', 'learnable_threshold'),
            threshold_init=model_config.get('router', {}).get('threshold_init', 0.5),
            temperature=model_config.get('router', {}).get('temperature', 1.0),
            mlp_big_hidden_dim=model_config.get('mlp_big', {}).get('hidden_dim', 512),
            mlp_big_output_dim=model_config.get('mlp_big', {}).get('output_dim', 128),
            mlp_big_layers=model_config.get('mlp_big', {}).get('num_layers', 3),
            mlp_small_hidden_dim=model_config.get('mlp_small', {}).get('hidden_dim', 64),
            mlp_small_output_dim=model_config.get('mlp_small', {}).get('output_dim', 128),
            mlp_small_layers=model_config.get('mlp_small', {}).get('num_layers', 1),
        )
    
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Supported: classification, detection, segmentation, instance_segmentation")


# Multi-task models
class AttentionRoutingDetector(nn.Module):
    """
    Attention routing model for object detection.
    
    Similar to AttentionRoutingModel but uses DetectionHead instead of classification head.
    """
    
    def __init__(
        self,
        # Backbone
        in_channels: int = 3,
        backbone_channels: int = 64,
        backbone_layers: int = 2,
        # Attention
        num_patches: int = 256,
        attention_hidden_dim: int = 32,
        attention_layers: int = 1,
        # Router
        router_mode: str = "learnable_threshold",
        threshold_init: float = 0.5,
        temperature: float = 1.0,
        # MLPs
        patch_input_dim: int = 3072,
        mlp_big_hidden_dim: int = 512,
        mlp_big_output_dim: int = 128,
        mlp_big_layers: int = 3,
        mlp_small_hidden_dim: int = 64,
        mlp_small_output_dim: int = 128,
        mlp_small_layers: int = 1,
        # Aggregator
        aggregator_mode: str = "concat",
        aggregator_output_dim: int = 256,
        # Detection head
        num_classes: int = 80,
        detection_hidden_dim: int = 256,
        num_anchors: int = 1,
    ):
        super().__init__()
        
        self.num_patches = num_patches
        self.patch_input_dim = patch_input_dim
        
        # Backbone encoder
        self.backbone = LightweightEncoder(
            in_channels=in_channels,
            out_channels=backbone_channels,
            num_layers=backbone_layers,
        )
        
        # Attention head
        self.attention = AttentionHead(
            feature_dim=backbone_channels,
            num_patches=num_patches,
            hidden_dim=attention_hidden_dim,
            num_layers=attention_layers,
        )
        
        # Router
        self.router = Router(
            mode=router_mode,
            threshold_init=threshold_init,
            temperature=temperature,
        )
        
        # MLPs
        self.mlp_big = LargeMLP(
            input_dim=patch_input_dim,
            hidden_dim=mlp_big_hidden_dim,
            output_dim=mlp_big_output_dim,
            num_layers=mlp_big_layers,
        )
        
        self.mlp_small = SmallMLP(
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
        
        # Detection head
        self.detection_head = DetectionHead(
            input_dim=aggregator_output_dim,
            patch_features_dim=mlp_big_output_dim,  # Use same as output_dim
            num_classes=num_classes,
            num_patches=num_patches,
            num_anchors=num_anchors,
            hidden_dim=detection_hidden_dim,
        )
    
    def forward(
        self,
        images: torch.Tensor,
        patches: torch.Tensor,
        patch_coords: Optional[list] = None,
        return_attention: bool = False,
        return_stats: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for detection.
        
        Args:
            images: Original images (B, C, H, W)
            patches: Extracted patches (B, N_patches, C, patch_H, patch_W)
            patch_coords: Optional patch coordinates
            return_attention: If True, return attention scores
            return_stats: If True, return routing statistics
            
        Returns:
            Dictionary with detection outputs
        """
        B = images.size(0)
        H, W = images.shape[2], images.shape[3]
        
        # 1. Encode with lightweight backbone
        features = self.backbone(images)  # (B, F, H', W')
        
        # 2. Compute attention scores
        attention_scores = self.attention(features)  # (B, N_patches)
        
        # 3. Make routing decisions
        routing_mask, router_stats = self.router(attention_scores, return_stats=True)
        
        # 4. Flatten patches for MLP processing
        patches_flat = patches.view(B, self.num_patches, -1)  # (B, N_patches, D)
        
        # 5. Conditional computation: Route patches
        high_attention_embeddings = self.mlp_big.forward_with_mask(
            patches_flat, routing_mask
        )  # (B, N_patches, output_dim)
        
        low_attention_embeddings = self.mlp_small.forward_with_mask(
            patches_flat, routing_mask
        )  # (B, N_patches, output_dim)
        
        # 6. Aggregate features
        global_features = self.aggregator.forward_split(
            high_attention_embeddings,
            low_attention_embeddings,
            attention_scores=attention_scores,
            routing_mask=routing_mask,
        )  # (B, aggregator_output_dim)
        
        # 7. Detection predictions
        det_outputs = self.detection_head(
            global_features,
            high_attention_embeddings + low_attention_embeddings,  # Combined patch features
            patch_coords=patch_coords,
        )
        
        result = {
            'cls_logits': det_outputs['cls_logits'],
            'reg_preds': det_outputs['reg_preds'],
        }
        
        if return_attention or return_stats:
            result['attention_scores'] = attention_scores
            result['routing_mask'] = routing_mask
            result['router_stats'] = router_stats
        
        return result


class AttentionRoutingSegmenter(nn.Module):
    """
    Attention routing model for semantic segmentation.
    """
    
    def __init__(
        self,
        # Backbone
        in_channels: int = 3,
        backbone_channels: int = 64,
        backbone_layers: int = 2,
        # Attention
        num_patches: int = 256,
        attention_hidden_dim: int = 32,
        attention_layers: int = 1,
        # Router
        router_mode: str = "learnable_threshold",
        threshold_init: float = 0.5,
        temperature: float = 1.0,
        # MLPs
        patch_input_dim: int = 3072,
        mlp_big_hidden_dim: int = 512,
        mlp_big_output_dim: int = 128,
        mlp_big_layers: int = 3,
        mlp_small_hidden_dim: int = 64,
        mlp_small_output_dim: int = 128,
        mlp_small_layers: int = 1,
        # Aggregator
        aggregator_mode: str = "concat",
        aggregator_output_dim: int = 256,
        # Segmentation head
        num_classes: int = 21,
        patch_grid_size: int = 16,
        segmentation_hidden_dim: int = 256,
        instance_seg: bool = False,
    ):
        super().__init__()
        
        self.num_patches = num_patches
        self.patch_input_dim = patch_input_dim
        self.patch_grid_size = patch_grid_size
        self.instance_seg = instance_seg
        
        # Backbone encoder
        self.backbone = LightweightEncoder(
            in_channels=in_channels,
            out_channels=backbone_channels,
            num_layers=backbone_layers,
        )
        
        # Attention head
        self.attention = AttentionHead(
            feature_dim=backbone_channels,
            num_patches=num_patches,
            hidden_dim=attention_hidden_dim,
            num_layers=attention_layers,
        )
        
        # Router
        self.router = Router(
            mode=router_mode,
            threshold_init=threshold_init,
            temperature=temperature,
        )
        
        # MLPs
        self.mlp_big = LargeMLP(
            input_dim=patch_input_dim,
            hidden_dim=mlp_big_hidden_dim,
            output_dim=mlp_big_output_dim,
            num_layers=mlp_big_layers,
        )
        
        self.mlp_small = SmallMLP(
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
        
        # Segmentation head
        if instance_seg:
            self.segmentation_head = InstanceSegmentationHead(
                input_dim=aggregator_output_dim,
                patch_features_dim=mlp_big_output_dim,
                num_classes=num_classes,
                num_patches=num_patches,
                patch_grid_size=patch_grid_size,
                hidden_dim=segmentation_hidden_dim,
            )
        else:
            self.segmentation_head = SegmentationHead(
                input_dim=aggregator_output_dim,
                patch_features_dim=mlp_big_output_dim,
                num_classes=num_classes,
                num_patches=num_patches,
                patch_grid_size=patch_grid_size,
                hidden_dim=segmentation_hidden_dim,
            )
    
    def forward(
        self,
        images: torch.Tensor,
        patches: torch.Tensor,
        return_attention: bool = False,
        return_stats: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for segmentation.
        
        Args:
            images: Original images (B, C, H, W)
            patches: Extracted patches (B, N_patches, C, patch_H, patch_W)
            return_attention: If True, return attention scores
            return_stats: If True, return routing statistics
            
        Returns:
            Dictionary with segmentation outputs
        """
        B = images.size(0)
        H, W = images.shape[2], images.shape[3]
        image_size = (H, W)
        
        # 1. Encode with lightweight backbone
        features = self.backbone(images)
        
        # 2. Compute attention scores
        attention_scores = self.attention(features)
        
        # 3. Make routing decisions
        routing_mask, router_stats = self.router(attention_scores, return_stats=True)
        
        # 4. Flatten patches for MLP processing
        patches_flat = patches.view(B, self.num_patches, -1)
        
        # 5. Conditional computation: Route patches
        high_attention_embeddings = self.mlp_big.forward_with_mask(
            patches_flat, routing_mask
        )
        
        low_attention_embeddings = self.mlp_small.forward_with_mask(
            patches_flat, routing_mask
        )
        
        # 6. Aggregate features
        global_features = self.aggregator.forward_split(
            high_attention_embeddings,
            low_attention_embeddings,
            attention_scores=attention_scores,
            routing_mask=routing_mask,
        )
        
        # 7. Segmentation predictions
        seg_outputs = self.segmentation_head(
            global_features,
            high_attention_embeddings + low_attention_embeddings,
            image_size=image_size,
        )
        
        result = seg_outputs.copy()
        
        if return_attention or return_stats:
            result['attention_scores'] = attention_scores
            result['routing_mask'] = routing_mask
            result['router_stats'] = router_stats
        
        return result

