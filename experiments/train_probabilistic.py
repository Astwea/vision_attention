"""Training script for probabilistic attention routing experiments."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import json
import argparse
import yaml

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.cluttered_mnist import get_cluttered_mnist_dataloaders
from models.probabilistic_model import create_probabilistic_model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    routing_ratios = []
    large_mlp_ratios = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        patches = batch['patches'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(
            images,
            patches,
            return_attention=True,
            return_stats=True,
            inference_mode="sampling",  # Use sampling during training
        )
        
        logits = outputs['logits']
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Routing statistics
        if 'router_stats' in outputs:
            stats = outputs['router_stats']
            routing_ratios.append(stats.get('routing_ratio', 0.0))
            large_mlp_ratios.append(stats.get('routing_ratio', 0.0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
        
        # Log to tensorboard
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            writer.add_scalar('Train/BatchAcc', 100.0 * correct / total, global_step)
            if routing_ratios:
                writer.add_scalar('Train/RoutingRatio', np.mean(routing_ratios), global_step)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    metrics = {
        'loss': epoch_loss,
        'acc': epoch_acc,
    }
    
    if routing_ratios:
        metrics['routing_ratio'] = np.mean(routing_ratios)
        metrics['large_mlp_ratio'] = np.mean(large_mlp_ratios)
    
    return metrics


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    inference_mode: str = "expectation",
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    routing_ratios = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            patches = batch['patches'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(
                images,
                patches,
                return_attention=True,
                return_stats=True,
                inference_mode=inference_mode,
            )
            
            logits = outputs['logits']
            loss = criterion(logits, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Routing statistics
            if 'router_stats' in outputs:
                stats = outputs['router_stats']
                routing_ratios.append(stats.get('routing_ratio', 0.0))
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    metrics = {
        'loss': epoch_loss,
        'acc': epoch_acc,
    }
    
    if routing_ratios:
        metrics['routing_ratio'] = np.mean(routing_ratios)
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Val/Loss', epoch_loss, epoch)
        writer.add_scalar('Val/Acc', epoch_acc, epoch)
        if routing_ratios:
            writer.add_scalar('Val/RoutingRatio', np.mean(routing_ratios), epoch)
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict,
    filepath: str,
    config: Dict,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
    }
    torch.save(checkpoint, filepath)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train probabilistic attention routing model')
    parser.add_argument('--config', type=str, default='configs/probabilistic_experiment.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='probabilistic',
                        choices=['probabilistic', 'baseline_large', 'baseline_small', 'deterministic'],
                        help='Model name')
    parser.add_argument('--router_mode', type=str, default='gumbel_softmax',
                        choices=['gumbel_softmax', 'straight_through'],
                        help='Router mode')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override with command line arguments
    if args.model:
        config['model_name'] = args.model
    if args.router_mode:
        config['router_mode'] = args.router_mode
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.epochs:
        config['num_epochs'] = args.epochs
    
    # Set defaults
    model_name = config.get('model_name', 'probabilistic')
    router_mode = config.get('router_mode', 'gumbel_softmax')
    batch_size = config.get('batch_size', 64)
    learning_rate = config.get('learning_rate', 0.001)
    num_epochs = config.get('num_epochs', 100)
    seed = config.get('seed', 42)
    device_name = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup
    set_seed(seed)
    device = torch.device(device_name)
    print(f'Using device: {device}')
    print(f'Model: {model_name}')
    print(f'Router mode: {router_mode}')
    
    # Create directories
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints/probabilistic')
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(checkpoint_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Data loaders
    # Reduce num_workers to avoid "Too many open files" error
    # Force num_workers=0 if not explicitly set in config to avoid multiprocessing issues
    num_workers = config.get('num_workers', 0)
    if num_workers > 0:
        print(f'Warning: Using num_workers={num_workers}. If you encounter "Too many open files" error, set num_workers=0 in config.')
    train_loader, val_loader = get_cluttered_mnist_dataloaders(
        root=config.get('data_root', './data'),
        image_size=64,
        digit_size=14,
        num_clutter_digits=4,
        noise_intensity=0.3,
        patch_grid_size=8,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Model
    num_patches = 8 * 8  # 64 patches
    patch_input_dim = 3 * 8 * 8  # 192 for 8x8 patches
    num_classes = 10
    
    model = create_probabilistic_model(
        model_name=model_name,
        num_patches=num_patches,
        patch_input_dim=patch_input_dim,
        num_classes=num_classes,
        router_mode=router_mode,
        temperature=config.get('temperature', 1.0),
        threshold=config.get('threshold', 0.5),
    )
    model = model.to(device)
    
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        scheduler.step()
        
        train_losses.append(train_metrics['loss'])
        train_accs.append(train_metrics['acc'])
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer,
            inference_mode="expectation",  # Use expectation for validation
        )
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['acc'])
        
        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path, config)
        
        # Save best model
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, best_path, config)
        
        # Print metrics
        print(f'\nEpoch {epoch}/{num_epochs}:')
        print(f'  Train Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["acc"]:.2f}%')
        print(f'  Val Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["acc"]:.2f}%')
        if 'routing_ratio' in train_metrics:
            print(f'  Train Routing Ratio: {train_metrics["routing_ratio"]:.3f}')
        if 'routing_ratio' in val_metrics:
            print(f'  Val Routing Ratio: {val_metrics["routing_ratio"]:.3f}')
        print()
    
    writer.close()
    
    # Save final results
    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_acc,
        'model_name': model_name,
        'router_mode': router_mode,
    }
    results_path = os.path.join(checkpoint_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Training completed. Best validation accuracy: {best_acc:.2f}%')
    print(f'Results saved to: {results_path}')


if __name__ == '__main__':
    main()

