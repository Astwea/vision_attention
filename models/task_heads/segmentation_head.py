"""Segmentation heads for semantic and instance segmentation tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SegmentationHead(nn.Module):
    """Segmentation head for semantic segmentation.
    
    Maps patch-level features back to pixel-level predictions.
    """
    
    def __init__(
        self,
        input_dim: int,  # From aggregator (global features)
        patch_features_dim: int,  # From patch embeddings
        num_classes: int,
        num_patches: int,
        patch_grid_size: int,
        hidden_dim: int = 256,
    ):
        """
        Args:
            input_dim: Global feature dimension from aggregator
            patch_features_dim: Feature dimension per patch
            num_classes: Number of segmentation classes (including background)
            num_patches: Number of patches (should equal patch_grid_size^2)
            patch_grid_size: Grid size (e.g., 16 for 16x16 patches)
            hidden_dim: Hidden dimension for segmentation head
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.patch_grid_size = patch_grid_size
        self.patch_features_dim = patch_features_dim
        self.patch_size = None  # Will be set based on input size
        
        # Fuse global and patch features
        self.fusion = nn.Sequential(
            nn.Linear(patch_features_dim + input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Pixel-level prediction head
        # First, predict per-patch segmentation
        self.seg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        global_features: torch.Tensor,
        patch_features: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            global_features: Global features from aggregator (B, input_dim)
            patch_features: Patch-level features (B, N_patches, patch_features_dim)
            image_size: Target image size (H, W)
            
        Returns:
            Dictionary containing:
                - logits: Segmentation logits (B, num_classes, H, W)
        """
        B, N_patches, F = patch_features.shape
        H, W = image_size
        self.patch_size = H // self.patch_grid_size
        
        # Expand global features to match patch features
        global_expanded = global_features.unsqueeze(1).expand(-1, N_patches, -1)  # (B, N_patches, input_dim)
        
        # Fuse global and patch features
        fused = torch.cat([patch_features, global_expanded], dim=-1)  # (B, N_patches, ...)
        fused = fused.view(-1, fused.size(-1))  # (B * N_patches, ...)
        features = self.fusion(fused)  # (B * N_patches, hidden_dim)
        
        # Per-patch segmentation predictions
        patch_logits = self.seg_head(features)  # (B * N_patches, num_classes)
        patch_logits = patch_logits.view(B, N_patches, self.num_classes)  # (B, N_patches, num_classes)
        
        # Reshape to spatial grid: (B, patch_grid_size, patch_grid_size, num_classes)
        patch_logits = patch_logits.view(B, self.patch_grid_size, self.patch_grid_size, self.num_classes)
        
        # Upsample to full resolution using interpolation
        # Permute to (B, num_classes, patch_grid_size, patch_grid_size)
        patch_logits = patch_logits.permute(0, 3, 1, 2)  # (B, num_classes, patch_grid_size, patch_grid_size)
        
        # Upsample to target image size
        logits = F.interpolate(
            patch_logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # (B, num_classes, H, W)
        
        return {
            'logits': logits,
        }


class InstanceSegmentationHead(nn.Module):
    """Instance segmentation head (combines detection + mask prediction)."""
    
    def __init__(
        self,
        input_dim: int,
        patch_features_dim: int,
        num_classes: int,
        num_patches: int,
        patch_grid_size: int,
        hidden_dim: int = 256,
    ):
        """
        Args:
            input_dim: Global feature dimension from aggregator
            patch_features_dim: Feature dimension per patch
            num_classes: Number of object classes
            num_patches: Number of patches
            patch_grid_size: Grid size for patches
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.patch_grid_size = patch_grid_size
        self.patch_features_dim = patch_features_dim
        
        # Detection components (similar to DetectionHead)
        self.fusion = nn.Sequential(
            nn.Linear(patch_features_dim + input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        # Regression head (bbox)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4),  # [x1, y1, x2, y2]
        )
        
        # Mask head: predict instance masks
        # Use a small FCN-like structure
        mask_feat_dim = hidden_dim
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, mask_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mask_feat_dim, mask_feat_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # Mask prediction (will be upsampled)
        self.mask_predictor = nn.Linear(mask_feat_dim // 2, patch_grid_size * patch_grid_size)
    
    def forward(
        self,
        global_features: torch.Tensor,
        patch_features: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            global_features: Global features from aggregator (B, input_dim)
            patch_features: Patch-level features (B, N_patches, patch_features_dim)
            image_size: Target image size (H, W)
            
        Returns:
            Dictionary containing:
                - cls_logits: Classification logits (B, N_patches, num_classes)
                - reg_preds: Bounding box predictions (B, N_patches, 4)
                - mask_logits: Mask logits (B, N_patches, H, W)
        """
        B, N_patches, F = patch_features.shape
        H, W = image_size
        
        # Expand global features
        global_expanded = global_features.unsqueeze(1).expand(-1, N_patches, -1)
        
        # Fuse features
        fused = torch.cat([patch_features, global_expanded], dim=-1)
        fused = fused.view(-1, fused.size(-1))
        features = self.fusion(fused)  # (B * N_patches, hidden_dim)
        
        # Classification predictions
        cls_logits = self.cls_head(features)  # (B * N_patches, num_classes)
        cls_logits = cls_logits.view(B, N_patches, self.num_classes)
        
        # Regression predictions
        reg_preds = self.reg_head(features)  # (B * N_patches, 4)
        reg_preds = reg_preds.view(B, N_patches, 4)
        
        # Mask predictions
        mask_features = self.mask_head(features)  # (B * N_patches, mask_feat_dim // 2)
        mask_logits = self.mask_predictor(mask_features)  # (B * N_patches, patch_grid_size^2)
        mask_logits = mask_logits.view(B, N_patches, self.patch_grid_size, self.patch_grid_size)
        
        # Upsample masks to image resolution
        mask_logits = F.interpolate(
            mask_logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # (B, N_patches, H, W)
        
        return {
            'cls_logits': cls_logits,
            'reg_preds': reg_preds,
            'mask_logits': mask_logits,
        }

