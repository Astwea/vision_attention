"""Loss functions for different vision tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class ClassificationLoss(nn.Module):
    """Classification loss (Cross Entropy)."""
    
    def __init__(self, weight: Optional[torch.Tensor] = None):
        """
        Args:
            weight: Optional class weights for imbalanced datasets
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: Classification logits (B, num_classes)
            labels: Ground truth labels (B,)
            
        Returns:
            Dictionary with 'loss' key
        """
        loss = self.criterion(logits, labels)
        return {'loss': loss}


class DetectionLoss(nn.Module):
    """Loss function for object detection.
    
    Combines classification loss and bounding box regression loss.
    """
    
    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reg_weight: float = 1.0,
        cls_weight: float = 1.0,
        use_focal: bool = True,
    ):
        """
        Args:
            num_classes: Number of object classes
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            reg_weight: Weight for regression loss
            cls_weight: Weight for classification loss
            use_focal: Whether to use focal loss for classification
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight
        self.use_focal = use_focal
        
        # Smooth L1 loss for regression
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for classification."""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss
    
    def forward(
        self,
        cls_logits: torch.Tensor,
        reg_preds: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        patch_coords: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            cls_logits: Classification logits (B, N_patches, num_anchors, num_classes)
            reg_preds: Regression predictions (B, N_patches, num_anchors, 4)
            gt_boxes: List of ground truth boxes per image (each is N_boxes, 4)
            gt_labels: List of ground truth labels per image (each is N_boxes,)
            patch_coords: Optional patch coordinates
            
        Returns:
            Dictionary with 'loss', 'cls_loss', 'reg_loss' keys
        """
        B = cls_logits.size(0)
        
        # For simplicity, we'll use a basic assignment strategy
        # In practice, you'd want to use anchor matching or Hungarian matching
        cls_losses = []
        reg_losses = []
        
        for b in range(B):
            gt_boxes_b = gt_boxes[b]  # (N_boxes, 4)
            gt_labels_b = gt_labels[b]  # (N_boxes,)
            
            if len(gt_boxes_b) == 0:
                # No ground truth objects - all patches are background
                # Use background class (assumed to be last class or handled separately)
                cls_logits_b = cls_logits[b].view(-1, self.num_classes)  # (N_patches * num_anchors, num_classes)
                bg_target = torch.zeros(cls_logits_b.size(0), dtype=torch.long, device=cls_logits.device)
                
                if self.use_focal:
                    cls_loss_b = self.focal_loss(cls_logits_b, bg_target).mean()
                else:
                    cls_loss_b = F.cross_entropy(cls_logits_b, bg_target)
                
                cls_losses.append(cls_loss_b)
                continue
            
            # Simple assignment: assign each GT box to nearest patch center
            # This is a simplified version; production code would use proper matching
            reg_preds_b = reg_preds[b].view(-1, 4)  # (N_patches * num_anchors, 4)
            cls_logits_b = cls_logits[b].view(-1, self.num_classes)  # (N_patches * num_anchors, num_classes)
            
            # Compute IoU or distance for assignment (simplified)
            # For now, use random assignment or assign to first patches
            num_pos = min(len(gt_boxes_b), len(reg_preds_b))
            pos_indices = torch.arange(num_pos, device=cls_logits.device)
            
            # Classification loss for positive samples
            if num_pos > 0:
                pos_cls_logits = cls_logits_b[pos_indices]  # (num_pos, num_classes)
                pos_labels = gt_labels_b[:num_pos]
                
                if self.use_focal:
                    cls_loss_pos = self.focal_loss(pos_cls_logits, pos_labels).mean()
                else:
                    cls_loss_pos = F.cross_entropy(pos_cls_logits, pos_labels)
                
                # Background classification for negative samples
                neg_indices = torch.arange(num_pos, len(cls_logits_b), device=cls_logits.device)
                if len(neg_indices) > 0:
                    neg_cls_logits = cls_logits_b[neg_indices]
                    bg_target = torch.zeros(len(neg_indices), dtype=torch.long, device=cls_logits.device)
                    
                    if self.use_focal:
                        cls_loss_neg = self.focal_loss(neg_cls_logits, bg_target).mean()
                    else:
                        cls_loss_neg = F.cross_entropy(neg_cls_logits, bg_target)
                    
                    cls_loss_b = cls_loss_pos + cls_loss_neg
                else:
                    cls_loss_b = cls_loss_pos
                
                # Regression loss for positive samples
                pos_reg_preds = reg_preds_b[pos_indices]  # (num_pos, 4)
                pos_gt_boxes = gt_boxes_b[:num_pos]  # (num_pos, 4)
                reg_loss_b = self.reg_loss(pos_reg_preds, pos_gt_boxes).mean()
                
                cls_losses.append(cls_loss_b)
                reg_losses.append(reg_loss_b)
            else:
                # All background
                bg_target = torch.zeros(cls_logits_b.size(0), dtype=torch.long, device=cls_logits.device)
                if self.use_focal:
                    cls_loss_b = self.focal_loss(cls_logits_b, bg_target).mean()
                else:
                    cls_loss_b = F.cross_entropy(cls_logits_b, bg_target)
                cls_losses.append(cls_loss_b)
        
        # Aggregate losses
        cls_loss = torch.stack(cls_losses).mean()
        reg_loss = torch.stack(reg_losses).mean() if len(reg_losses) > 0 else torch.tensor(0.0, device=cls_logits.device)
        
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
        }


class SegmentationLoss(nn.Module):
    """Loss function for semantic segmentation."""
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        weight: Optional[torch.Tensor] = None,
        use_dice: bool = False,
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in loss computation
            weight: Optional class weights
            use_dice: Whether to use Dice loss in addition to CE loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.use_dice = use_dice
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss."""
        # Convert predictions to probabilities
        pred_probs = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Compute Dice coefficient per class
        smooth = 1.0
        dice_scores = []
        
        for c in range(self.num_classes):
            pred_c = pred_probs[:, c, :, :]
            target_c = target_one_hot[:, c, :, :]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)
        
        dice_loss = 1.0 - torch.stack(dice_scores).mean()
        return dice_loss
    
    def forward(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation loss.
        
        Args:
            logits: Segmentation logits (B, num_classes, H, W)
            masks: Ground truth masks (B, H, W) with class indices
            
        Returns:
            Dictionary with 'loss' and optionally 'dice_loss' keys
        """
        # Cross entropy loss
        ce_loss = self.ce_loss(logits, masks)
        
        if self.use_dice:
            dice_loss = self.dice_loss(logits, masks)
            total_loss = ce_loss + dice_loss
            return {
                'loss': total_loss,
                'ce_loss': ce_loss,
                'dice_loss': dice_loss,
            }
        else:
            return {
                'loss': ce_loss,
            }


class InstanceSegLoss(nn.Module):
    """Loss function for instance segmentation (detection + mask prediction)."""
    
    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reg_weight: float = 1.0,
        cls_weight: float = 1.0,
        mask_weight: float = 1.0,
        use_focal: bool = True,
    ):
        """
        Args:
            num_classes: Number of object classes
            alpha: Focal loss alpha
            gamma: Focal loss gamma
            reg_weight: Weight for bbox regression loss
            cls_weight: Weight for classification loss
            mask_weight: Weight for mask prediction loss
            use_focal: Whether to use focal loss
        """
        super().__init__()
        self.detection_loss = DetectionLoss(
            num_classes=num_classes,
            alpha=alpha,
            gamma=gamma,
            reg_weight=reg_weight,
            cls_weight=cls_weight,
            use_focal=use_focal,
        )
        self.mask_weight = mask_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        cls_logits: torch.Tensor,
        reg_preds: torch.Tensor,
        mask_logits: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
        patch_coords: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute instance segmentation loss.
        
        Args:
            cls_logits: Classification logits (B, N_patches, num_classes)
            reg_preds: Regression predictions (B, N_patches, 4)
            mask_logits: Mask logits (B, N_patches, H, W)
            gt_boxes: List of GT boxes per image
            gt_labels: List of GT labels per image
            gt_masks: List of GT masks per image (each is N_instances, H, W)
            patch_coords: Optional patch coordinates
            
        Returns:
            Dictionary with loss components
        """
        # Detection loss
        det_loss_dict = self.detection_loss(
            cls_logits.unsqueeze(2),  # Add anchor dimension
            reg_preds.unsqueeze(2),
            gt_boxes,
            gt_labels,
            patch_coords,
        )
        
        # Mask loss (simplified - would need proper assignment)
        # For now, use a simple approach
        B = mask_logits.size(0)
        mask_losses = []
        
        for b in range(B):
            mask_logits_b = mask_logits[b]  # (N_patches, H, W)
            gt_masks_b = gt_masks[b]  # (N_instances, H, W)
            
            if len(gt_masks_b) == 0:
                continue
            
            # Simple assignment (in practice, use proper matching)
            num_matched = min(len(gt_masks_b), mask_logits_b.size(0))
            matched_masks = gt_masks_b[:num_matched]  # (num_matched, H, W)
            matched_logits = mask_logits_b[:num_matched]  # (num_matched, H, W)
            
            # BCE loss for masks
            mask_loss_b = self.bce_loss(matched_logits, matched_masks)
            mask_losses.append(mask_loss_b)
        
        mask_loss = torch.stack(mask_losses).mean() if len(mask_losses) > 0 else torch.tensor(0.0, device=mask_logits.device)
        
        total_loss = det_loss_dict['loss'] + self.mask_weight * mask_loss
        
        return {
            'loss': total_loss,
            'cls_loss': det_loss_dict['cls_loss'],
            'reg_loss': det_loss_dict['reg_loss'],
            'mask_loss': mask_loss,
        }

