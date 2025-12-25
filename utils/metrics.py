"""Evaluation metrics for different vision tasks."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCO_AVAILABLE = True
except ImportError:
    PYCOCO_AVAILABLE = False


def compute_classification_accuracy(logits: torch.Tensor, labels: torch.Tensor, top_k: int = 1) -> float:
    """
    Compute classification accuracy (top-k).
    
    Args:
        logits: Classification logits (B, num_classes)
        labels: Ground truth labels (B,)
        top_k: Top-k accuracy (1 or 5)
        
    Returns:
        Accuracy as float (0-1)
    """
    _, pred = logits.topk(top_k, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    correct_k = correct[:top_k].reshape(-1).float().sum(0)
    accuracy = correct_k / labels.size(0)
    return accuracy.item()


def compute_miou(pred_masks: torch.Tensor, gt_masks: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    """
    Compute mean Intersection over Union (mIoU) for semantic segmentation.
    
    Args:
        pred_masks: Predicted masks (B, H, W) with class indices
        gt_masks: Ground truth masks (B, H, W) with class indices
        num_classes: Number of classes
        ignore_index: Index to ignore in computation
        
    Returns:
        mIoU as float
    """
    pred_masks = pred_masks.cpu().numpy()
    gt_masks = gt_masks.cpu().numpy()
    
    # Flatten masks
    pred_flat = pred_masks.flatten()
    gt_flat = gt_masks.flatten()
    
    # Remove ignored pixels
    valid = gt_flat != ignore_index
    pred_flat = pred_flat[valid]
    gt_flat = gt_flat[valid]
    
    # Compute confusion matrix
    confusion_matrix = np.bincount(
        num_classes * gt_flat.astype(int) + pred_flat.astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    
    # Compute IoU per class
    intersection = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - intersection
    iou = intersection / (union + 1e-8)
    
    # Compute mIoU (average over valid classes)
    valid_classes = union > 0
    miou = iou[valid_classes].mean()
    
    return float(miou)


def compute_detection_map_coco(
    predictions: List[Dict],
    gt_annotations: str,
    image_ids: List[int],
    score_threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Compute mAP for object detection using COCO evaluation API.
    
    Args:
        predictions: List of prediction dictionaries, each containing:
            - 'image_id': int
            - 'boxes': (N, 4) tensor in [x1, y1, x2, y2] format
            - 'scores': (N,) tensor
            - 'labels': (N,) tensor (0-indexed class labels)
        gt_annotations: Path to COCO format ground truth annotation file
        image_ids: List of image IDs
        score_threshold: Score threshold for filtering predictions
        
    Returns:
        Dictionary with mAP metrics
    """
    if not PYCOCO_AVAILABLE:
        raise ImportError("pycocotools is required for COCO evaluation")
    
    # Load ground truth
    coco_gt = COCO(gt_annotations)
    
    # Convert predictions to COCO format
    coco_results = []
    for pred in predictions:
        image_id = pred['image_id']
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # Filter by score threshold
        valid = scores >= score_threshold
        boxes = boxes[valid]
        scores = scores[valid]
        labels = labels[valid]
        
        # Convert boxes to COCO format [x, y, width, height]
        boxes_coco = boxes.copy()
        boxes_coco[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        boxes_coco[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
        
        # Create COCO format results
        for box, score, label in zip(boxes_coco, scores, labels):
            coco_results.append({
                'image_id': image_id,
                'category_id': int(label + 1),  # COCO uses 1-indexed labels
                'bbox': box.tolist(),
                'score': float(score),
            })
    
    # Load results into COCO API
    coco_dt = coco_gt.loadRes(coco_results)
    
    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        'mAP': coco_eval.stats[0],  # mAP @ IoU=0.50:0.95
        'mAP_50': coco_eval.stats[1],  # mAP @ IoU=0.50
        'mAP_75': coco_eval.stats[2],  # mAP @ IoU=0.75
        'mAP_small': coco_eval.stats[3],
        'mAP_medium': coco_eval.stats[4],
        'mAP_large': coco_eval.stats[5],
    }
    
    return metrics


def compute_detection_map_voc(
    predictions: List[Dict],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute mAP for Pascal VOC style evaluation.
    
    Args:
        predictions: List of prediction dictionaries
        gt_boxes: List of ground truth boxes per image
        gt_labels: List of ground truth labels per image
        num_classes: Number of classes
        iou_threshold: IoU threshold for positive detection
        
    Returns:
        Dictionary with mAP metric
    """
    # Compute AP for each class
    aps = []
    
    for cls_idx in range(num_classes):
        # Collect predictions and ground truths for this class
        pred_scores = []
        pred_boxes = []
        gt_boxes_cls = []
        gt_flags = []  # Whether each GT has been matched
        
        for i, pred in enumerate(predictions):
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores_batch = pred['scores'].cpu().numpy()
            pred_boxes_batch = pred['boxes'].cpu().numpy()
            
            # Filter predictions for this class
            cls_mask = pred_labels == cls_idx
            if cls_mask.sum() > 0:
                pred_scores.extend(pred_scores_batch[cls_mask].tolist())
                pred_boxes.extend(pred_boxes_batch[cls_mask].tolist())
            
            # Ground truth for this class
            gt_labels_batch = gt_labels[i].cpu().numpy()
            gt_boxes_batch = gt_boxes[i].cpu().numpy()
            gt_mask = gt_labels_batch == cls_idx
            if gt_mask.sum() > 0:
                gt_boxes_cls.extend(gt_boxes_batch[gt_mask].tolist())
                gt_flags.extend([False] * gt_mask.sum())
        
        if len(gt_boxes_cls) == 0:
            continue  # No ground truth for this class
        
        # Sort predictions by score
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_scores = [pred_scores[i] for i in sorted_indices]
        pred_boxes = [pred_boxes[i] for i in sorted_indices]
        
        # Compute precision and recall
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        
        for i, pred_box in enumerate(pred_boxes):
            # Find best matching GT
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes_cls):
                if gt_flags[j]:
                    continue
                
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                gt_flags[best_gt_idx] = True
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(gt_boxes_cls)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Compute AP (area under precision-recall curve)
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
    
    # Compute mAP
    map_value = np.mean(aps) if len(aps) > 0 else 0.0
    
    return {
        'mAP': float(map_value),
        'AP_per_class': aps,
    }


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-8)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute average precision from precision-recall curve."""
    # Append sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap

