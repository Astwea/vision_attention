"""ImageNet dataset loader with patch splitting support."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, List
import os
from PIL import Image


class ImageNetPatchDataset(Dataset):
    """ImageNet dataset with patch extraction support."""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = False,
        input_size: int = 224,
        patch_grid_size: int = 7,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            root: Root directory of dataset
            split: 'train' or 'val'
            download: If True, download the dataset (only valid for torchvision ImageNet)
            input_size: Target input size (224, 256, 384, etc.)
            patch_grid_size: Grid size for patch splitting (e.g., 7 means 7x7=49 patches)
            transform: Optional transforms to apply
        """
        self.root = root
        self.split = split
        self.input_size = input_size
        self.patch_grid_size = patch_grid_size
        self.patch_size = input_size // patch_grid_size
        self.num_patches = patch_grid_size * patch_grid_size
        
        # Load ImageNet dataset using torchvision
        # Note: ImageNet requires manual download, so download parameter is ignored
        self.dataset = torchvision.datasets.ImageNet(
            root=root,
            split=split,
            transform=None,  # We'll apply transforms manually
        )
        
        # Build transforms
        if transform is None:
            if split == "train":
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        
        self.transform = transform
        self.num_classes = 1000
        
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
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - image: Full image tensor (C, H, W)
                - label: Class label
                - patches: Extracted patches (N_patches, C, patch_H, patch_W)
                - patch_coords: Patch coordinates information
        """
        image, label = self.dataset[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Ensure image is correct size
        if image.shape[1] != self.input_size or image.shape[2] != self.input_size:
            image = transforms.functional.resize(image, (self.input_size, self.input_size))
        
        # Extract patches
        patches = self._extract_patches(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'patches': patches,
            'patch_coords': self.patch_coords,
            'patch_grid_size': self.patch_grid_size,
            'num_patches': self.num_patches
        }
    
    def _extract_patches(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract non-overlapping patches from image.
        
        Args:
            image: Tensor of shape (C, H, W)
            
        Returns:
            Tensor of shape (N_patches, C, patch_H, patch_W)
        """
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
    
    def get_patch_from_original(
        self,
        original_image: torch.Tensor,
        patch_idx: int
    ) -> torch.Tensor:
        """
        Extract a specific patch from original (un-normalized) image.
        
        Args:
            original_image: Original image tensor (C, H, W) before normalization
            patch_idx: Index of patch to extract
            
        Returns:
            Patch tensor (C, patch_H, patch_W)
        """
        if patch_idx >= self.num_patches:
            raise ValueError(f"Patch index {patch_idx} >= num_patches {self.num_patches}")
        
        coord = self.patch_coords[patch_idx]
        patch = original_image[
            :,
            coord['y_start']:coord['y_end'],
            coord['x_start']:coord['x_end']
        ]
        return patch


def get_imagenet_dataloaders(
    root: str = "./data/imagenet",
    input_size: int = 224,
    patch_grid_size: int = 7,
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ImageNet train and validation dataloaders.
    
    Args:
        root: Root directory for dataset
        input_size: Input image size
        patch_grid_size: Patch grid size
        batch_size: Batch size
        num_workers: Number of data loading workers
        download: Whether to download dataset (ignored for ImageNet)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = ImageNetPatchDataset(
        root=root,
        split="train",
        download=download,
        input_size=input_size,
        patch_grid_size=patch_grid_size,
    )
    
    val_dataset = ImageNetPatchDataset(
        root=root,
        split="val",
        download=download,
        input_size=input_size,
        patch_grid_size=patch_grid_size,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader

