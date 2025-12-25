"""Pascal VOC dataset loader with patch splitting support for detection and semantic segmentation."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, List
import os
import xml.etree.ElementTree as ET
from PIL import Image
import cv2


VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
VOC_NUM_CLASSES = len(VOC_CLASSES)  # 21 (including background)


class VOCDetectionDataset(Dataset):
    """Pascal VOC dataset for object detection with patch extraction support."""
    
    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        input_size: int = 512,
        patch_grid_size: int = 16,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ):
        """
        Args:
            root: Root directory of dataset
            year: Dataset year ("2007", "2012", or "0712" for combined)
            image_set: "train", "val", "trainval", or "test"
            input_size: Target input size
            patch_grid_size: Grid size for patch splitting
            transform: Optional transforms to apply
            download: Whether to download dataset
        """
        self.root = root
        self.year = year
        self.image_set = image_set
        self.input_size = input_size
        self.patch_grid_size = patch_grid_size
        self.patch_size = input_size // patch_grid_size
        self.num_patches = patch_grid_size * patch_grid_size
        self.num_classes = VOC_NUM_CLASSES - 1  # 20 classes (excluding background)
        
        # Build transforms
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.transform = transform
        
        # Get image and annotation paths
        self._get_image_list()
        
        # Pre-compute patch coordinates
        self._compute_patch_coordinates()
    
    def _get_image_list(self):
        """Get list of image files and corresponding annotation files."""
        if self.year == "0712":
            # Combined 2007 + 2012
            self.image_list = []
            for year in ["2007", "2012"]:
                splits_dir = os.path.join(self.root, f"VOC{year}")
                split_file = os.path.join(splits_dir, "ImageSets", "Main", f"{self.image_set}.txt")
                if os.path.exists(split_file):
                    with open(split_file, 'r') as f:
                        for line in f:
                            image_id = line.strip()
                            self.image_list.append((year, image_id))
        else:
            splits_dir = os.path.join(self.root, f"VOC{self.year}")
            split_file = os.path.join(splits_dir, "ImageSets", "Main", f"{self.image_set}.txt")
            
            if not os.path.exists(split_file):
                if os.path.exists(os.path.join(splits_dir, "ImageSets", "Main", f"{self.image_set}.txt")):
                    pass
                else:
                    raise RuntimeError(f"Split file not found: {split_file}")
            
            self.image_list = []
            with open(split_file, 'r') as f:
                for line in f:
                    image_id = line.strip()
                    self.image_list.append((self.year, image_id))
    
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
        return len(self.image_list)
    
    def _parse_voc_xml(self, xml_path: str) -> Tuple[List, List]:
        """Parse Pascal VOC XML annotation file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # Skip difficult objects
            if obj.find('difficult') is not None and int(obj.find('difficult').text) == 1:
                continue
            
            # Get class name and convert to index
            class_name = obj.find('name').text
            if class_name in VOC_CLASSES:
                class_idx = VOC_CLASSES.index(class_name) - 1  # -1 to exclude background
                
                # Get bounding box
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)
                
                boxes.append([x1, y1, x2, y2])
                labels.append(class_idx)
        
        return boxes, labels
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dictionary containing:
                - image: Full image tensor (C, H, W)
                - patches: Extracted patches (N_patches, C, patch_H, patch_W)
                - boxes: Bounding boxes (N_boxes, 4) in format [x1, y1, x2, y2]
                - labels: Class labels (N_boxes,)
                - image_id: Image ID
                - original_size: Original image size (H, W)
        """
        year, image_id = self.image_list[idx]
        
        # Load image
        img_path = os.path.join(self.root, f"VOC{year}", "JPEGImages", f"{image_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        # Resize image
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        image_array = np.array(image)
        
        # Load annotations
        xml_path = os.path.join(self.root, f"VOC{year}", "Annotations", f"{image_id}.xml")
        boxes, labels = self._parse_voc_xml(xml_path)
        
        # Scale bounding boxes to input_size
        scale_x = self.input_size / original_size[0]
        scale_y = self.input_size / original_size[1]
        
        scaled_boxes = []
        valid_labels = []
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            
            # Clip to image bounds
            x1 = max(0, min(x1, self.input_size))
            y1 = max(0, min(y1, self.input_size))
            x2 = max(0, min(x2, self.input_size))
            y2 = max(0, min(y2, self.input_size))
            
            # Only keep valid boxes
            if x2 > x1 and y2 > y1:
                scaled_boxes.append([x1, y1, x2, y2])
                valid_labels.append(label)
        
        # Convert to tensors
        if len(scaled_boxes) > 0:
            boxes_tensor = torch.tensor(scaled_boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(valid_labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        
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
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': image_id,
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


class VOCSegmentationDataset(Dataset):
    """Pascal VOC dataset for semantic segmentation with patch extraction support."""
    
    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        input_size: int = 512,
        patch_grid_size: int = 16,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ):
        """
        Args:
            root: Root directory of dataset
            year: Dataset year ("2012" for segmentation)
            image_set: "train", "val", or "trainval"
            input_size: Target input size
            patch_grid_size: Grid size for patch splitting
            transform: Optional transforms to apply
            download: Whether to download dataset
        """
        self.root = root
        self.year = year
        self.image_set = image_set
        self.input_size = input_size
        self.patch_grid_size = patch_grid_size
        self.patch_size = input_size // patch_grid_size
        self.num_patches = patch_grid_size * patch_grid_size
        self.num_classes = VOC_NUM_CLASSES  # 21 classes (including background)
        
        # Build transforms
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.transform = transform
        
        # Get image and mask paths
        self._get_image_list()
        
        # Pre-compute patch coordinates
        self._compute_patch_coordinates()
    
    def _get_image_list(self):
        """Get list of image files and corresponding mask files."""
        splits_dir = os.path.join(self.root, f"VOC{self.year}")
        split_file = os.path.join(splits_dir, "ImageSets", "Segmentation", f"{self.image_set}.txt")
        
        if not os.path.exists(split_file):
            raise RuntimeError(f"Split file not found: {split_file}")
        
        self.image_list = []
        with open(split_file, 'r') as f:
            for line in f:
                image_id = line.strip()
                self.image_list.append(image_id)
    
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
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dictionary containing:
                - image: Full image tensor (C, H, W)
                - patches: Extracted patches (N_patches, C, patch_H, patch_W)
                - mask: Segmentation mask (H, W) with class indices (0-20)
                - image_id: Image ID
                - original_size: Original image size (H, W)
        """
        image_id = self.image_list[idx]
        
        # Load image
        img_path = os.path.join(self.root, f"VOC{self.year}", "JPEGImages", f"{image_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        # Resize image
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        image_array = np.array(image)
        
        # Load segmentation mask
        mask_path = os.path.join(self.root, f"VOC{self.year}", "SegmentationClass", f"{image_id}.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
            # Resize mask to input_size (use nearest neighbor to preserve class indices)
            mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)
            mask_array = np.array(mask)
            
            # Convert RGB mask to class indices (VOC masks use color coding)
            # Map RGB colors to class indices
            mask_indices = np.zeros((self.input_size, self.input_size), dtype=np.int64)
            
            # Simple approach: use the first channel as class index (if mask is already in index format)
            # Otherwise, we need to map RGB colors to indices
            # For now, assume mask is already in index format (0-255, where 255 is background/ignore)
            if len(mask_array.shape) == 3:
                # RGB mask - convert to index
                # This is a simplified version; full implementation would map RGB colors
                mask_indices = mask_array[:, :, 0]
            else:
                mask_indices = mask_array
            
            # Convert 255 (background/ignore) to 0 (background class)
            mask_indices = np.where(mask_indices == 255, 0, mask_indices)
            # Clip to valid range [0, 20]
            mask_indices = np.clip(mask_indices, 0, 20)
        else:
            # If mask doesn't exist, create empty mask (all background)
            mask_indices = np.zeros((self.input_size, self.input_size), dtype=np.int64)
        
        # Convert to tensors
        mask_tensor = torch.tensor(mask_indices, dtype=torch.long)
        
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
            'mask': mask_tensor,
            'image_id': image_id,
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


def get_voc_dataloaders(
    root: str = "./data/voc",
    year: str = "2012",
    task_type: str = "detection",  # "detection" or "segmentation"
    input_size: int = 512,
    patch_grid_size: int = 16,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get Pascal VOC train and validation dataloaders.
    
    Args:
        root: Root directory for dataset
        year: Dataset year
        task_type: "detection" or "segmentation"
        input_size: Input image size
        patch_grid_size: Patch grid size
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if task_type == "detection":
        dataset_class = VOCDetectionDataset
        image_set_train = "trainval" if year == "2007" else "train"
        image_set_val = "test" if year == "2007" else "val"
    elif task_type == "segmentation":
        dataset_class = VOCSegmentationDataset
        image_set_train = "train"
        image_set_val = "val"
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    train_dataset = dataset_class(
        root=root,
        year=year,
        image_set=image_set_train,
        input_size=input_size,
        patch_grid_size=patch_grid_size,
    )
    
    val_dataset = dataset_class(
        root=root,
        year=year,
        image_set=image_set_val,
        input_size=input_size,
        patch_grid_size=patch_grid_size,
    )
    
    def collate_fn(batch):
        """Custom collate function for variable number of objects."""
        images = torch.stack([item['image'] for item in batch])
        patches = torch.stack([item['patches'] for item in batch])
        image_ids = [item['image_id'] for item in batch]
        original_sizes = torch.stack([item['original_size'] for item in batch])
        
        result = {
            'image': images,
            'patches': patches,
            'image_id': image_ids,
            'original_size': original_sizes,
        }
        
        # Add task-specific items
        if task_type == "detection":
            result['boxes'] = [item['boxes'] for item in batch]
            result['labels'] = [item['labels'] for item in batch]
        elif task_type == "segmentation":
            result['mask'] = torch.stack([item['mask'] for item in batch])
        
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

