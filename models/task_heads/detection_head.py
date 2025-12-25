"""Detection head for object detection tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class DetectionHead(nn.Module):
    """Detection head for object detection.
    
    Based on single-stage detector architecture (similar to RetinaNet/FCOS).
    Takes patch-level features and produces detection predictions.
    """
    
    def __init__(
        self,
        input_dim: int,  # From aggregator (global features)
        patch_features_dim: int,  # From patch embeddings
        num_classes: int,
        num_patches: int,
        num_anchors: int = 1,  # Number of anchors per patch
        hidden_dim: int = 256,
    ):
        """
        Args:
            input_dim: Global feature dimension from aggregator
            patch_features_dim: Feature dimension per patch
            num_classes: Number of object classes
            num_patches: Number of patches (N_patches)
            num_anchors: Number of anchor boxes per patch
            hidden_dim: Hidden dimension for detection head
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.num_anchors = num_anchors
        self.patch_features_dim = patch_features_dim
        
        # Fuse global and patch features
        # We'll use patch features directly for spatial predictions
        # Global features can be used as additional context
        self.fusion = nn.Sequential(
            nn.Linear(patch_features_dim + input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Classification head: predict class probabilities
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes * num_anchors),
        )
        
        # Regression head: predict bounding box offsets
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4 * num_anchors),  # [x1, y1, x2, y2]
        )
        
        # Objectness head: predict object/background (optional, can use cls_head)
        # For simplicity, we'll use cls_head with background class
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        global_features: torch.Tensor,
        patch_features: torch.Tensor,
        patch_coords: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            global_features: Global features from aggregator (B, input_dim)
            patch_features: Patch-level features (B, N_patches, patch_features_dim)
            patch_coords: Optional patch coordinates for spatial information
            
        Returns:
            Dictionary containing:
                - cls_logits: Classification logits (B, N_patches, num_anchors, num_classes)
                - reg_preds: Bounding box predictions (B, N_patches, num_anchors, 4)
        """
        B, N_patches, F = patch_features.shape
        
        # Expand global features to match patch features
        global_expanded = global_features.unsqueeze(1).expand(-1, N_patches, -1)  # (B, N_patches, input_dim)
        
        # Fuse global and patch features
        fused = torch.cat([patch_features, global_expanded], dim=-1)  # (B, N_patches, patch_features_dim + input_dim)
        fused = fused.view(-1, fused.size(-1))  # (B * N_patches, ...)
        features = self.fusion(fused)  # (B * N_patches, hidden_dim)
        
        # Classification predictions
        cls_logits = self.cls_head(features)  # (B * N_patches, num_classes * num_anchors)
        cls_logits = cls_logits.view(B, N_patches, self.num_anchors, self.num_classes)  # (B, N_patches, num_anchors, num_classes)
        
        # Regression predictions
        reg_preds = self.reg_head(features)  # (B * N_patches, 4 * num_anchors)
        reg_preds = reg_preds.view(B, N_patches, self.num_anchors, 4)  # (B, N_patches, num_anchors, 4)
        
        return {
            'cls_logits': cls_logits,
            'reg_preds': reg_preds,
        }
    
    def decode_boxes(
        self,
        reg_preds: torch.Tensor,
        patch_coords: list,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Decode predicted box offsets to absolute coordinates.
        
        Args:
            reg_preds: Predicted box offsets (B, N_patches, num_anchors, 4)
            patch_coords: List of patch coordinate dictionaries
            image_size: (H, W) of input image
            
        Returns:
            Decoded boxes in [x1, y1, x2, y2] format (B, N_patches, num_anchors, 4)
        """
        B, N_patches, num_anchors, _ = reg_preds.shape
        H, W = image_size
        
        # Get patch centers and sizes
        patch_centers = []
        patch_sizes = []
        for coord in patch_coords:
            center_x = (coord['x_start'] + coord['x_end']) / 2.0
            center_y = (coord['y_start'] + coord['y_end']) / 2.0
            patch_w = coord['x_end'] - coord['x_start']
            patch_h = coord['y_end'] - coord['y_start']
            patch_centers.append([center_x, center_y])
            patch_sizes.append([patch_w, patch_h])
        
        patch_centers = torch.tensor(patch_centers, device=reg_preds.device, dtype=torch.float32)  # (N_patches, 2)
        patch_sizes = torch.tensor(patch_sizes, device=reg_preds.device, dtype=torch.float32)  # (N_patches, 2)
        
        # Expand for batch and anchors
        patch_centers = patch_centers.unsqueeze(0).unsqueeze(2)  # (1, N_patches, 1, 2)
        patch_sizes = patch_sizes.unsqueeze(0).unsqueeze(2)  # (1, N_patches, 1, 2)
        
        # Decode: reg_preds are offsets relative to patch center and size
        # Simple decoding: treat reg_preds as relative offsets
        centers = patch_centers.expand(B, -1, num_anchors, -1)  # (B, N_patches, num_anchors, 2)
        sizes = patch_sizes.expand(B, -1, num_anchors, -1)  # (B, N_patches, num_anchors, 2)
        
        # Apply offsets (reg_preds: [dx, dy, dw, dh])
        dx = reg_preds[:, :, :, 0] * sizes[:, :, :, 0]  # Scale by patch width
        dy = reg_preds[:, :, :, 1] * sizes[:, :, :, 1]  # Scale by patch height
        dw = reg_preds[:, :, :, 2] * sizes[:, :, :, 0]  # Width offset
        dh = reg_preds[:, :, :, 3] * sizes[:, :, :, 1]  # Height offset
        
        # Convert to absolute coordinates
        center_x = centers[:, :, :, 0] + dx
        center_y = centers[:, :, :, 1] + dy
        w = sizes[:, :, :, 0] * torch.exp(dw)  # Exponential for width
        h = sizes[:, :, :, 1] * torch.exp(dh)  # Exponential for height
        
        # Convert center + size to x1, y1, x2, y2
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2
        
        # Clip to image bounds
        x1 = torch.clamp(x1, 0, W)
        y1 = torch.clamp(y1, 0, H)
        x2 = torch.clamp(x2, 0, W)
        y2 = torch.clamp(y2, 0, H)
        
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, N_patches, num_anchors, 4)
        
        return boxes

