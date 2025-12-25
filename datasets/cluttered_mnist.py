"""Cluttered MNIST dataset for probabilistic attention routing experiments.

This dataset creates 64x64 images with small MNIST digits (14x14) placed randomly,
surrounded by strong background clutter (noise, digit fragments, textures).
Only the central digit determines the label, making it necessary to use
fine-grained processing (Large MLP) for key regions.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, Tuple, Optional
import random


class ClutteredMNIST(Dataset):
    """
    Cluttered MNIST dataset for attention routing experiments.
    
    Creates 64x64 images with:
    - Small MNIST digit (14x14) randomly placed
    - Strong background clutter (noise, fragments, textures)
    - Label determined only by the central digit
    """
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        image_size: int = 64,
        digit_size: int = 14,
        num_clutter_digits: int = 4,
        noise_intensity: float = 0.3,
        download: bool = True,
        patch_grid_size: int = 8,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            root: Root directory for MNIST data
            train: Whether to use training set
            image_size: Size of output images (64x64)
            digit_size: Size of MNIST digits (14x14)
            num_clutter_digits: Number of clutter digit fragments
            noise_intensity: Intensity of background noise (0-1)
            download: Whether to download MNIST if not present
            patch_grid_size: Grid size for patch extraction (8x8 = 64 patches)
            transform: Optional image transforms
        """
        self.image_size = image_size
        self.digit_size = digit_size
        self.num_clutter_digits = num_clutter_digits
        self.noise_intensity = noise_intensity
        self.patch_grid_size = patch_grid_size
        self.transform = transform
        
        # Load MNIST dataset
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=None  # We'll handle transforms ourselves
        )
        
        # Compute patch coordinates
        self._compute_patch_coordinates()
    
    def _compute_patch_coordinates(self):
        """Compute patch coordinates for patch extraction."""
        patch_size = self.image_size // self.patch_grid_size
        self.patch_coords = []
        
        for i in range(self.patch_grid_size):
            for j in range(self.patch_grid_size):
                y_start = i * patch_size
                y_end = (i + 1) * patch_size
                x_start = j * patch_size
                x_end = (j + 1) * patch_size
                
                self.patch_coords.append({
                    'y_start': y_start,
                    'y_end': y_end,
                    'x_start': x_start,
                    'x_end': x_end,
                    'patch_idx': i * self.patch_grid_size + j
                })
        
        self.num_patches = len(self.patch_coords)
    
    def __len__(self) -> int:
        return len(self.mnist)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a cluttered MNIST image.
        
        Returns:
            Dictionary containing:
                - image: Full image tensor (C, H, W)
                - label: Class label (0-9)
                - patches: Extracted patches (N_patches, C, patch_H, patch_W)
                - patch_coords: Patch coordinates information
        """
        # Get main digit and label
        main_digit, label = self.mnist[idx]
        main_digit = transforms.functional.to_tensor(main_digit)  # (1, 28, 28)
        
        # Resize main digit to target size
        main_digit = F.interpolate(
            main_digit.unsqueeze(0),
            size=(self.digit_size, self.digit_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (1, digit_size, digit_size)
        
        # Create blank 64x64 image
        image = torch.zeros(1, self.image_size, self.image_size)
        
        # Add background noise
        noise = torch.randn(1, self.image_size, self.image_size) * self.noise_intensity
        image = image + noise
        
        # Randomly place main digit (but keep it visible)
        # Place it in a region that ensures it's mostly visible
        margin = self.digit_size // 2
        max_y = self.image_size - self.digit_size - margin
        max_x = self.image_size - self.digit_size - margin
        min_y = margin
        min_x = margin
        
        # Prefer center region for main digit (but with some randomness)
        center_y = self.image_size // 2
        center_x = self.image_size // 2
        
        # Random offset from center
        offset_y = random.randint(-self.digit_size, self.digit_size)
        offset_x = random.randint(-self.digit_size, self.digit_size)
        
        main_y = max(min_y, min(max_y, center_y + offset_y - self.digit_size // 2))
        main_x = max(min_x, min(max_x, center_x + offset_x - self.digit_size // 2))
        
        # Place main digit
        image[
            :,
            main_y:main_y + self.digit_size,
            main_x:main_x + self.digit_size
        ] = torch.maximum(
            image[
                :,
                main_y:main_y + self.digit_size,
                main_x:main_x + self.digit_size
            ],
            main_digit
        )
        
        # Add clutter digits (fragments of other digits)
        for _ in range(self.num_clutter_digits):
            clutter_idx = random.randint(0, len(self.mnist) - 1)
            clutter_digit, _ = self.mnist[clutter_idx]
            clutter_digit = transforms.functional.to_tensor(clutter_digit)
            
            # Resize to random smaller size (fragments)
            fragment_size = random.randint(8, 12)
            clutter_digit = F.interpolate(
                clutter_digit.unsqueeze(0),
                size=(fragment_size, fragment_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Randomly place fragment
            frag_y = random.randint(0, self.image_size - fragment_size)
            frag_x = random.randint(0, self.image_size - fragment_size)
            
            # Add with some transparency to make it less prominent
            alpha = random.uniform(0.3, 0.6)
            image[
                :,
                frag_y:frag_y + fragment_size,
                frag_x:frag_x + fragment_size
            ] = torch.maximum(
                image[
                    :,
                    frag_y:frag_y + fragment_size,
                    frag_x:frag_x + fragment_size
                ],
                clutter_digit * alpha
            )
        
        # Add texture patterns (random lines/patterns)
        if random.random() < 0.5:
            # Add some random lines
            for _ in range(random.randint(2, 5)):
                y1 = random.randint(0, self.image_size - 1)
                x1 = random.randint(0, self.image_size - 1)
                y2 = random.randint(0, self.image_size - 1)
                x2 = random.randint(0, self.image_size - 1)
                
                # Draw line (simple approximation)
                steps = max(abs(y2 - y1), abs(x2 - x1))
                if steps > 0:
                    for s in range(steps):
                        y = int(y1 + (y2 - y1) * s / steps)
                        x = int(x1 + (x2 - x1) * s / steps)
                        if 0 <= y < self.image_size and 0 <= x < self.image_size:
                            image[:, y, x] = min(1.0, image[:, y, x] + 0.2)
        
        # Clamp values to [0, 1]
        image = torch.clamp(image, 0.0, 1.0)
        
        # Convert to RGB (repeat channel)
        image = image.repeat(3, 1, 1)  # (3, H, W)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default normalization
            image = transforms.functional.normalize(
                image,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        
        # Extract patches
        patches = self._extract_patches(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'patches': patches,
            'patch_coords': self.patch_coords,
            'patch_grid_size': self.patch_grid_size,
            'num_patches': self.num_patches,
            'main_digit_pos': (main_y, main_x),  # For visualization/debugging
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


def get_cluttered_mnist_dataloaders(
    root: str = "./data",
    image_size: int = 64,
    digit_size: int = 14,
    num_clutter_digits: int = 4,
    noise_intensity: float = 0.3,
    patch_grid_size: int = 8,
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ClutteredMNIST dataloaders.
    
    Args:
        root: Root directory for data
        image_size: Size of images (64x64)
        digit_size: Size of main digit (14x14)
        num_clutter_digits: Number of clutter fragments
        noise_intensity: Background noise intensity
        patch_grid_size: Patch grid size (8x8 = 64 patches)
        batch_size: Batch size
        num_workers: Number of data loading workers
        download: Whether to download MNIST
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = ClutteredMNIST(
        root=root,
        train=True,
        image_size=image_size,
        digit_size=digit_size,
        num_clutter_digits=num_clutter_digits,
        noise_intensity=noise_intensity,
        patch_grid_size=patch_grid_size,
        download=download,
    )
    
    val_dataset = ClutteredMNIST(
        root=root,
        train=False,
        image_size=image_size,
        digit_size=digit_size,
        num_clutter_digits=num_clutter_digits,
        noise_intensity=noise_intensity,
        patch_grid_size=patch_grid_size,
        download=download,
    )
    
    # Disable pin_memory if num_workers is 0 to avoid "Too many open files" error
    pin_memory = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader

