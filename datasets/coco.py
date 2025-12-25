"""COCO dataset loader with patch splitting support for detection and instance segmentation."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, List
import os
import json
from PIL import Image
import cv2

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as coco_mask
    PYCOCO_AVAILABLE = True
except ImportError:
    PYCOCO_AVAILABLE = False
    print("Warning: pycocotools not available. COCO dataset functionality will be limited.")


class COCODetectionDataset(Dataset):
    """COCO dataset for object detection with patch extraction support."""
    
    def __init__(
        self,
        root: str,
        ann_file: str,
        input_size: int = 512,
        patch_grid_size: int = 16,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
    ):
        """
        Args:
            root: Root directory of dataset (should contain images/)
            ann_file: Path to annotation file (JSON format)
            input_size: Target input size
            patch_grid_size: Grid size for patch splitting
            transform: Optional transforms to apply
            is_train: Whether this is training set
        """
        if not PYCOCO_AVAILABLE:
            raise ImportError("pycocotools is required for COCO dataset. Install with: pip install pycocotools")
        
        self.root = root
        self.ann_file = ann_file
        self.input_size = input_size
        self.patch_grid_size = patch_grid_size
        self.patch_size = input_size // patch_grid_size
        self.num_patches = patch_grid_size * patch_grid_size
        self.is_train = is_train
        
        # Load COCO API
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.cat_ids = self.coco.getCatIds()
        self.num_classes = len(self.cat_ids)  # 80 for COCO
        
        # Build transforms
        if transform is None:
            if is_train:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        
        self.transform = transform
        
        # Pre-compute patch coordinates
        self._compute_patch_coordinates()
    
    def _compute_patch_coordinates(self):
        """Pre-compute patch coordinates for efficient extraction."""
        self.patch_coords = []
        for i in range(self.patch_grid_size):
            for j in range(self.patch_grid_size):
                y_start = i * self.patch_size
                y_end = (i + 1) * self.patch_size
                x_start = j * self.patch_size
                x_end = (j + 1) * self.patch_size
                self.patch_coords.append({
                    'y_start': y_start,
                    'y_end': y_end,
                    'x_start': x_start,
                    'x_end': x_end,
                    'patch_idx': i * self.patch_grid_size + j
                })
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dictionary containing:
                - image: Full image tensor (C, H, W)
                - patches: Extracted patches (N_patches, C, patch_H, patch_W)
                - boxes: Bounding boxes (N_boxes, 4) in format [x1, y1, x2, y2]
                - labels: Class labels (N_boxes,)
                - image_id: Original image ID
                - original_size: Original image size (H, W)
        """
        img_id = self.img_ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        # Resize image
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        image_array = np.array(image)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        
        for ann in anns:
            if ann.get('iscrowd', 0) == 0:  # Skip crowd annotations for detection
                bbox = ann['bbox']  # [x, y, width, height]
                # Convert to [x1, y1, x2, y2] and scale to input_size
                x1 = bbox[0] * self.input_size / original_size[0]
                y1 = bbox[1] * self.input_size / original_size[1]
                x2 = (bbox[0] + bbox[2]) * self.input_size / original_size[0]
                y2 = (bbox[1] + bbox[3]) * self.input_size / original_size[1]
                
                # Clip to image bounds
                x1 = max(0, min(x1, self.input_size))
                y1 = max(0, min(y1, self.input_size))
                x2 = max(0, min(x2, self.input_size))
                y2 = max(0, min(y2, self.input_size))
                
                # Only keep valid boxes
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    # Map category ID to index (0-79)
                    cat_id = ann['category_id']
                    cat_idx = self.cat_ids.index(cat_id)
                    labels.append(cat_idx)
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        # Convert image to tensor and apply transforms
        image_tensor = transforms.functional.to_tensor(image_array)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        else:
            image_tensor = transforms.functional.normalize(
                image_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        
        # Extract patches
        patches = self._extract_patches(image_tensor)
        
        return {
            'image': image_tensor,
            'patches': patches,
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
            'original_size': torch.tensor(original_size, dtype=torch.int32),
            'patch_coords': self.patch_coords,
            'patch_grid_size': self.patch_grid_size,
            'num_patches': self.num_patches
        }
    
    def _extract_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping patches from image."""
        C, H, W = image.shape
        patches = []
        
        for coord in self.patch_coords:
            patch = image[
                :,
                coord['y_start']:coord['y_end'],
                coord['x_start']:coord['x_end']
            ]
            patches.append(patch)
        
        return torch.stack(patches, dim=0)


class COCOInstanceSegDataset(COCODetectionDataset):
    """COCO dataset for instance segmentation with patch extraction support."""
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dictionary containing:
                - image: Full image tensor (C, H, W)
                - patches: Extracted patches (N_patches, C, patch_H, patch_W)
                - boxes: Bounding boxes (N_boxes, 4)
                - labels: Class labels (N_boxes,)
                - masks: Instance masks (N_boxes, H, W) - binary masks
                - image_id: Original image ID
                - original_size: Original image size (H, W)
        """
        img_id = self.img_ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        # Resize image
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        image_array = np.array(image)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract bounding boxes, labels, and masks
        boxes = []
        labels = []
        masks = []
        
        for ann in anns:
            if ann.get('iscrowd', 0) == 0:
                bbox = ann['bbox']
                # Convert bbox and scale
                x1 = bbox[0] * self.input_size / original_size[0]
                y1 = bbox[1] * self.input_size / original_size[1]
                x2 = (bbox[0] + bbox[2]) * self.input_size / original_size[0]
                y2 = (bbox[1] + bbox[3]) * self.input_size / original_size[1]
                
                # Clip to image bounds
                x1 = max(0, min(x1, self.input_size))
                y1 = max(0, min(y1, self.input_size))
                x2 = max(0, min(x2, self.input_size))
                y2 = max(0, min(y2, self.input_size))
                
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    
                    # Map category ID to index
                    cat_id = ann['category_id']
                    cat_idx = self.cat_ids.index(cat_id)
                    labels.append(cat_idx)
                    
                    # Decode and resize mask
                    rle = ann['segmentation']
                    if isinstance(rle, list):
                        # Polygon format - convert to RLE
                        rle = coco_mask.frPyObjects(rle, original_size[1], original_size[0])
                        rle = coco_mask.merge(rle)
                    
                    mask = coco_mask.decode(rle)
                    # Resize mask to input_size
                    mask = cv2.resize(mask.astype(np.uint8), (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
                    masks.append(mask.astype(np.float32))
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            masks = torch.tensor(np.stack(masks), dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            masks = torch.zeros((0, self.input_size, self.input_size), dtype=torch.float32)
        
        # Convert image to tensor and apply transforms
        image_tensor = transforms.functional.to_tensor(image_array)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        else:
            image_tensor = transforms.functional.normalize(
                image_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        
        # Extract patches
        patches = self._extract_patches(image_tensor)
        
        return {
            'image': image_tensor,
            'patches': patches,
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': img_id,
            'original_size': torch.tensor(original_size, dtype=torch.int32),
            'patch_coords': self.patch_coords,
            'patch_grid_size': self.patch_grid_size,
            'num_patches': self.num_patches
        }


def get_coco_dataloaders(
    root: str = "./data/coco",
    ann_file_train: str = "annotations/instances_train2017.json",
    ann_file_val: str = "annotations/instances_val2017.json",
    task_type: str = "detection",  # "detection" or "instance_segmentation"
    input_size: int = 512,
    patch_grid_size: int = 16,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get COCO train and validation dataloaders.
    
    Args:
        root: Root directory for dataset
        ann_file_train: Path to training annotation file
        ann_file_val: Path to validation annotation file
        task_type: "detection" or "instance_segmentation"
        input_size: Input image size
        patch_grid_size: Patch grid size
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    ann_file_train_path = os.path.join(root, ann_file_train)
    ann_file_val_path = os.path.join(root, ann_file_val)
    images_dir = os.path.join(root, "train2017") if "train" in ann_file_train else os.path.join(root, "images")
    
    if task_type == "detection":
        dataset_class = COCODetectionDataset
    elif task_type == "instance_segmentation":
        dataset_class = COCOInstanceSegDataset
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    train_dataset = dataset_class(
        root=images_dir,
        ann_file=ann_file_train_path,
        input_size=input_size,
        patch_grid_size=patch_grid_size,
        is_train=True,
    )
    
    val_images_dir = os.path.join(root, "val2017") if "val" in ann_file_val else images_dir
    val_dataset = dataset_class(
        root=val_images_dir,
        ann_file=ann_file_val_path,
        input_size=input_size,
        patch_grid_size=patch_grid_size,
        is_train=False,
    )
    
    def collate_fn(batch):
        """Custom collate function for variable number of objects."""
        images = torch.stack([item['image'] for item in batch])
        patches = torch.stack([item['patches'] for item in batch])
        boxes = [item['boxes'] for item in batch]
        labels = [item['labels'] for item in batch]
        image_ids = [item['image_id'] for item in batch]
        original_sizes = torch.stack([item['original_size'] for item in batch])
        
        result = {
            'image': images,
            'patches': patches,
            'boxes': boxes,
            'labels': labels,
            'image_id': image_ids,
            'original_size': original_sizes,
        }
        
        # Add masks if present
        if 'masks' in batch[0]:
            result['masks'] = [item['masks'] for item in batch]
        
        return result
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader

