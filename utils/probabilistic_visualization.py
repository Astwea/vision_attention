"""Visualization tools for probabilistic attention routing experiments."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple
import os


def visualize_attention_heatmap(
    image: torch.Tensor,
    attention_scores: torch.Tensor,
    patch_grid_size: int = 8,
    save_path: Optional[str] = None,
    title: str = "Attention Heatmap",
):
    """
    Visualize attention scores as heatmap overlaid on image.
    
    Args:
        image: Image tensor (C, H, W) or (H, W, C) numpy array
        attention_scores: Attention scores (N_patches,)
        patch_grid_size: Grid size (8x8 = 64 patches)
        save_path: Path to save figure
        title: Figure title
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            # (C, H, W) -> (H, W, C)
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image.cpu().numpy()
    else:
        image_np = image
    
    # Denormalize if needed (assuming ImageNet normalization)
    if image_np.min() < 0:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
    
    # Convert attention scores to numpy
    if isinstance(attention_scores, torch.Tensor):
        attn_np = attention_scores.cpu().numpy()
    else:
        attn_np = attention_scores
    
    # Reshape attention to grid
    attn_grid = attn_np.reshape(patch_grid_size, patch_grid_size)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(image_np)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Attention heatmap
    im = ax2.imshow(attn_grid, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_title(title)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_routing_decisions(
    image: torch.Tensor,
    attention_scores: torch.Tensor,
    routing_mask: torch.Tensor,
    patch_grid_size: int = 8,
    save_path: Optional[str] = None,
    title: str = "Routing Decisions",
):
    """
    Visualize routing decisions (Large MLP vs Small MLP) overlaid on image.
    
    Args:
        image: Image tensor (C, H, W)
        attention_scores: Attention scores (N_patches,)
        routing_mask: Routing mask (N_patches,) - 1 = Large MLP, 0 = Small MLP
        patch_grid_size: Grid size
        save_path: Path to save figure
        title: Figure title
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image.cpu().numpy()
    else:
        image_np = image
    
    # Denormalize if needed
    if image_np.min() < 0:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
    
    if isinstance(attention_scores, torch.Tensor):
        attn_np = attention_scores.cpu().numpy()
    else:
        attn_np = attention_scores
    
    if isinstance(routing_mask, torch.Tensor):
        routing_np = routing_mask.cpu().numpy()
    else:
        routing_np = routing_mask
    
    # Reshape to grid
    attn_grid = attn_np.reshape(patch_grid_size, patch_grid_size)
    routing_grid = routing_np.reshape(patch_grid_size, patch_grid_size)
    
    H, W = image_np.shape[:2]
    patch_h = H // patch_grid_size
    patch_w = W // patch_grid_size
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image_np)
    ax.set_title(title)
    ax.axis('off')
    
    # Draw patches with colors
    for i in range(patch_grid_size):
        for j in range(patch_grid_size):
            patch_idx = i * patch_grid_size + j
            y_start = i * patch_h
            x_start = j * patch_w
            
            # Color: Green = Large MLP, Red = Small MLP
            # Opacity based on attention score
            if routing_grid[i, j] > 0.5:
                color = 'green'
                alpha = 0.3 * attn_grid[i, j]
            else:
                color = 'red'
                alpha = 0.3 * (1 - attn_grid[i, j])
            
            rect = patches.Rectangle(
                (x_start, y_start),
                patch_w,
                patch_h,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_probability_distribution(
    probs: torch.Tensor,
    routing_mask: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    title: str = "Probability Distribution",
):
    """
    Visualize probability distribution and routing decisions.
    
    Args:
        probs: Probabilities (N_patches,) or (B, N_patches)
        routing_mask: Optional routing mask (N_patches,) or (B, N_patches)
        save_path: Path to save figure
        title: Figure title
    """
    # Convert to numpy
    if isinstance(probs, torch.Tensor):
        probs_np = probs.cpu().numpy()
    else:
        probs_np = probs
    
    if routing_mask is not None:
        if isinstance(routing_mask, torch.Tensor):
            routing_np = routing_mask.cpu().numpy()
        else:
            routing_np = routing_mask
    
    # Handle batch dimension
    if probs_np.ndim == 2:
        probs_np = probs_np[0]  # Take first sample
        if routing_mask is not None:
            routing_np = routing_np[0]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Probability histogram
    ax1.hist(probs_np, bins=20, range=(0, 1), alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Probability Distribution')
    ax1.axvline(probs_np.mean(), color='red', linestyle='--', label=f'Mean: {probs_np.mean():.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Routing decisions
    if routing_mask is not None:
        large_mlp_count = (routing_np > 0.5).sum()
        small_mlp_count = (routing_np <= 0.5).sum()
        
        ax2.bar(['Large MLP', 'Small MLP'], [large_mlp_count, small_mlp_count],
                color=['green', 'red'], alpha=0.7)
        ax2.set_ylabel('Number of Patches')
        ax2.set_title('Routing Decisions')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No routing mask provided', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_tradeoff_curve(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Performance vs Computation Trade-off",
):
    """
    Plot performance vs computation trade-off curve.
    
    Args:
        results: Dictionary mapping model names to metrics
            Each value should have 'accuracy' and 'routing_ratio' or 'computation'
        save_path: Path to save figure
        title: Figure title
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    model_names = []
    accuracies = []
    computations = []
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, (model_name, metrics) in enumerate(results.items()):
        model_names.append(model_name)
        accuracies.append(metrics.get('accuracy', 0.0))
        
        # Use routing_ratio as proxy for computation
        # Higher routing_ratio = more computation (more Large MLP usage)
        if 'routing_ratio' in metrics:
            computations.append(metrics['routing_ratio'])
        elif 'computation' in metrics:
            computations.append(metrics['computation'])
        else:
            # Default: assume full computation for baselines
            if 'large' in model_name.lower():
                computations.append(1.0)
            else:
                computations.append(0.0)
    
    # Plot points
    for i, (name, acc, comp) in enumerate(zip(model_names, accuracies, computations)):
        color = colors[i % len(colors)]
        ax.scatter(comp, acc, s=200, alpha=0.7, color=color, label=name)
        ax.annotate(name, (comp, acc), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Computation (Large MLP Usage Ratio)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_batch(
    images: torch.Tensor,
    patches: torch.Tensor,
    attention_scores: torch.Tensor,
    routing_masks: Optional[torch.Tensor] = None,
    patch_grid_size: int = 8,
    num_samples: int = 4,
    save_dir: Optional[str] = None,
    prefix: str = "batch",
):
    """
    Visualize a batch of samples with attention and routing.
    
    Args:
        images: Images (B, C, H, W)
        patches: Patches (B, N_patches, C, patch_H, patch_W)
        attention_scores: Attention scores (B, N_patches)
        routing_masks: Optional routing masks (B, N_patches)
        patch_grid_size: Grid size
        num_samples: Number of samples to visualize
        save_dir: Directory to save figures
        prefix: Prefix for saved files
    """
    B = images.size(0)
    num_samples = min(num_samples, B)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_samples):
        image = images[i]
        attn = attention_scores[i]
        routing = routing_masks[i] if routing_masks is not None else None
        
        # Attention heatmap
        if save_dir:
            heatmap_path = os.path.join(save_dir, f'{prefix}_heatmap_{i}.png')
        else:
            heatmap_path = None
        visualize_attention_heatmap(
            image, attn, patch_grid_size, heatmap_path,
            title=f'Sample {i} - Attention Heatmap'
        )
        
        # Routing decisions
        if routing is not None:
            if save_dir:
                routing_path = os.path.join(save_dir, f'{prefix}_routing_{i}.png')
            else:
                routing_path = None
            visualize_routing_decisions(
                image, attn, routing, patch_grid_size, routing_path,
                title=f'Sample {i} - Routing Decisions'
            )


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
):
    """
    Plot training curves (loss and accuracy).
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Training accuracies per epoch
        val_accs: Validation accuracies per epoch
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

