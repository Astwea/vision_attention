"""Multi-task training script supporting classification, detection, and segmentation."""

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

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.factory import get_dataloaders, get_dataset_info
from models import create_model
from models.losses import ClassificationLoss, DetectionLoss, SegmentationLoss, InstanceSegLoss
from utils.config import get_config, setup_directories
from utils.flops import calculate_conditional_flops, format_flops, estimate_model_flops
from utils.visualization import save_attention_batch
import os


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
    config: Dict,
    task_type: str,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    # Task-specific metrics
    if task_type == "classification":
        correct = 0
        total = 0
    elif task_type == "detection":
        cls_losses = []
        reg_losses = []
    elif task_type in ["segmentation", "instance_segmentation"]:
        pass  # Can add segmentation-specific metrics
    
    routing_ratios = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        patches = batch['patches'].to(device)
        
        # Get task-specific inputs
        if task_type == "classification":
            labels = batch['label'].to(device)
        elif task_type == "detection":
            gt_boxes = [b.to(device) for b in batch['boxes']]
            gt_labels = [l.to(device) for l in batch['labels']]
        elif task_type == "segmentation":
            gt_masks = batch['mask'].to(device)
        elif task_type == "instance_segmentation":
            gt_boxes = [b.to(device) for b in batch['boxes']]
            gt_labels = [l.to(device) for l in batch['labels']]
            gt_masks = [m.to(device) for m in batch['masks']]
        
        # Forward pass
        if hasattr(model, 'router'):
            outputs = model(
                images,
                patches,
                patch_coords=batch.get('patch_coords'),
                return_attention=True,
                return_stats=True,
            )
            routing_stats = outputs.get('router_stats', {})
            routing_ratios.append(routing_stats.get('routing_ratio', 0.0))
        else:
            outputs = model(images, patches)
        
        # Compute loss
        if task_type == "classification":
            logits = outputs['logits']
            loss_dict = criterion(logits, labels)
            loss = loss_dict['loss']
            
            # Accuracy
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        elif task_type == "detection":
            cls_logits = outputs['cls_logits']
            reg_preds = outputs['reg_preds']
            loss_dict = criterion(
                cls_logits,
                reg_preds,
                gt_boxes,
                gt_labels,
                patch_coords=batch.get('patch_coords'),
            )
            loss = loss_dict['loss']
            cls_losses.append(loss_dict['cls_loss'].item())
            reg_losses.append(loss_dict['reg_loss'].item())
            
        elif task_type == "segmentation":
            logits = outputs['logits']
            loss_dict = criterion(logits, gt_masks)
            loss = loss_dict['loss']
            
        elif task_type == "instance_segmentation":
            cls_logits = outputs['cls_logits']
            reg_preds = outputs['reg_preds']
            mask_logits = outputs['mask_logits']
            loss_dict = criterion(
                cls_logits,
                reg_preds,
                mask_logits,
                gt_boxes,
                gt_labels,
                gt_masks,
                patch_coords=batch.get('patch_coords'),
            )
            loss = loss_dict['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Log to tensorboard
        if writer and batch_idx % config.get('training', {}).get('log_interval', 100) == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            if routing_ratios:
                writer.add_scalar('Train/RoutingRatio', np.mean(routing_ratios), global_step)
    
    epoch_loss = running_loss / len(dataloader)
    metrics = {'loss': epoch_loss}
    
    if task_type == "classification":
        epoch_acc = 100.0 * correct / total
        metrics['acc'] = epoch_acc
    elif task_type == "detection" and cls_losses:
        metrics['cls_loss'] = np.mean(cls_losses)
        metrics['reg_loss'] = np.mean(reg_losses)
    
    if routing_ratios:
        metrics['routing_ratio'] = np.mean(routing_ratios)
    
    return metrics


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: Dict,
    task_type: str,
    writer: Optional[SummaryWriter] = None,
    save_attention: bool = False,
    vis_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    running_loss = 0.0
    
    # Task-specific metrics
    if task_type == "classification":
        correct = 0
        total = 0
    
    routing_ratios = []
    
    # For attention visualization
    all_attention_scores = []
    all_routing_masks = []
    sample_images = []
    sample_patches = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            patches = batch['patches'].to(device)
            
            # Get task-specific inputs
            if task_type == "classification":
                labels = batch['label'].to(device)
            elif task_type == "detection":
                gt_boxes = [b.to(device) for b in batch['boxes']]
                gt_labels = [l.to(device) for l in batch['labels']]
            elif task_type == "segmentation":
                gt_masks = batch['mask'].to(device)
            elif task_type == "instance_segmentation":
                gt_boxes = [b.to(device) for b in batch['boxes']]
                gt_labels = [l.to(device) for l in batch['labels']]
                gt_masks = [m.to(device) for m in batch['masks']]
            
            # Forward pass
            if hasattr(model, 'router'):
                outputs = model(
                    images,
                    patches,
                    patch_coords=batch.get('patch_coords'),
                    return_attention=True,
                    return_stats=True,
                )
                routing_stats = outputs.get('router_stats', {})
                routing_ratios.append(routing_stats.get('routing_ratio', 0.0))
                
                # Save attention for visualization (first few batches)
                if save_attention and batch_idx < 2:
                    all_attention_scores.append(outputs['attention_scores'].cpu())
                    all_routing_masks.append(outputs['routing_mask'].cpu())
                    sample_images.append(images.cpu())
                    sample_patches.append(patches.cpu())
            else:
                outputs = model(images, patches)
            
            # Compute loss (similar to training)
            if task_type == "classification":
                logits = outputs['logits']
                loss_dict = criterion(logits, labels)
                loss = loss_dict['loss']
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            elif task_type == "detection":
                cls_logits = outputs['cls_logits']
                reg_preds = outputs['reg_preds']
                loss_dict = criterion(
                    cls_logits,
                    reg_preds,
                    gt_boxes,
                    gt_labels,
                    patch_coords=batch.get('patch_coords'),
                )
                loss = loss_dict['loss']
            elif task_type == "segmentation":
                logits = outputs['logits']
                loss_dict = criterion(logits, gt_masks)
                loss = loss_dict['loss']
            elif task_type == "instance_segmentation":
                cls_logits = outputs['cls_logits']
                reg_preds = outputs['reg_preds']
                mask_logits = outputs['mask_logits']
                loss_dict = criterion(
                    cls_logits,
                    reg_preds,
                    mask_logits,
                    gt_boxes,
                    gt_labels,
                    gt_masks,
                    patch_coords=batch.get('patch_coords'),
                )
                loss = loss_dict['loss']
            
            # Statistics
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    metrics = {'loss': epoch_loss}
    
    if task_type == "classification":
        epoch_acc = 100.0 * correct / total
        metrics['acc'] = epoch_acc
    
    if routing_ratios:
        metrics['routing_ratio'] = np.mean(routing_ratios)
    
    # Save attention visualizations
    if save_attention and all_attention_scores and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        patch_grid_size = config.get('dataset', {}).get('patch_grid_size', 4)
        images_batch = torch.cat(sample_images, dim=0)
        patches_batch = torch.cat(sample_patches, dim=0)
        attention_batch = torch.cat(all_attention_scores, dim=0)
        routing_batch = torch.cat(all_routing_masks, dim=0) if all_routing_masks else None
        
        save_attention_batch(
            images_batch,
            patches_batch,
            attention_batch,
            routing_batch,
            patch_grid_size,
            vis_dir,
            prefix=f'epoch_{epoch}',
            num_samples=4,
        )
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Val/Loss', epoch_loss, epoch)
        if 'acc' in metrics:
            writer.add_scalar('Val/Acc', metrics['acc'], epoch)
        if 'routing_ratio' in metrics:
            writer.add_scalar('Val/RoutingRatio', metrics['routing_ratio'], epoch)
    
    return metrics


def create_loss_function(task_type: str, config: Dict, num_classes: int, device: torch.device):
    """Create appropriate loss function for task type."""
    task_type = task_type.lower()
    loss_config = config.get('training', {}).get('loss', {})
    
    if task_type == "classification":
        return ClassificationLoss()
    
    elif task_type == "detection":
        return DetectionLoss(
            num_classes=num_classes,
            alpha=loss_config.get('alpha', 0.25),
            gamma=loss_config.get('gamma', 2.0),
            reg_weight=loss_config.get('reg_weight', 1.0),
            cls_weight=loss_config.get('cls_weight', 1.0),
            use_focal=loss_config.get('use_focal', True),
        ).to(device)
    
    elif task_type == "segmentation":
        return SegmentationLoss(
            num_classes=num_classes,
            ignore_index=loss_config.get('ignore_index', 255),
            use_dice=loss_config.get('use_dice', False),
        )
    
    elif task_type == "instance_segmentation":
        return InstanceSegLoss(
            num_classes=num_classes,
            alpha=loss_config.get('alpha', 0.25),
            gamma=loss_config.get('gamma', 2.0),
            reg_weight=loss_config.get('reg_weight', 1.0),
            cls_weight=loss_config.get('cls_weight', 1.0),
            mask_weight=loss_config.get('mask_weight', 1.0),
            use_focal=loss_config.get('use_focal', True),
        ).to(device)
    
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def create_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Create optimizer based on config."""
    training_config = config.get('training', {})
    optimizer_name = training_config.get('optimizer', 'adam').lower()
    lr = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0001)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = training_config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict, num_epochs: int):
    """Create learning rate scheduler."""
    training_config = config.get('training', {})
    scheduler_name = training_config.get('scheduler', 'cosine').lower()
    
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == 'step':
        step_size = training_config.get('step_size', 30)
        gamma = training_config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None
    
    return scheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict,
    filepath: str,
    config: Dict,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': config,
    }
    torch.save(checkpoint, filepath)


def main():
    """Main training function."""
    from utils.config import parse_args
    
    args = parse_args()
    config = get_config(args.config, args)
    
    # Setup
    set_seed(config.get('seed', 42))
    setup_directories(config)
    
    # Get task type
    task_type = config.get('task_type', 'classification')
    
    # Device
    device_name = config.get('device', 'cuda')
    if device_name == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA requested but not available. Using CPU instead.')
        device_name = 'cpu'
    device = torch.device(device_name)
    print(f'Using device: {device}')
    print(f'Task type: {task_type}')
    
    # Get dataset info
    dataset_config = config.get('dataset', {})
    dataset_name = dataset_config.get('name')
    dataset_info = get_dataset_info(dataset_name, task_type)
    num_classes = dataset_info['num_classes']
    
    # Data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        task_type=task_type,
        root=dataset_config.get('root', './data'),
        input_size=dataset_config.get('input_size', 224),
        patch_grid_size=dataset_config.get('patch_grid_size', 7),
        batch_size=config.get('training', {}).get('batch_size', 64),
        num_workers=dataset_config.get('num_workers', 4),
        **dataset_config
    )
    
    # Model
    model_name = config.get('model', {}).get('name', 'attention_routing')
    model = create_model(model_name, config, task_type=task_type)
    model = model.to(device)
    
    print(f'Model: {model_name}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Loss function
    criterion = create_loss_function(task_type, config, num_classes, device)
    
    # Optimizer
    optimizer = create_optimizer(model, config)
    
    num_epochs = config.get('training', {}).get('num_epochs', 100)
    scheduler = create_scheduler(optimizer, config, num_epochs)
    
    # Tensorboard writer
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', './checkpoints')
    log_dir = os.path.join(checkpoint_dir, 'logs')
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_metric = 0.0  # Will use accuracy for classification, mAP for detection
    train_losses = []
    val_losses = []
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, task_type, writer
        )
        
        if scheduler:
            scheduler.step()
        
        train_losses.append(train_metrics['loss'])
        
        # Validate
        save_attention = (epoch % config.get('training', {}).get('visualize_interval', 50) == 0)
        vis_dir = config.get('training', {}).get('vis_dir', './visualizations') if save_attention else None
        
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, config, task_type, writer,
            save_attention=save_attention,
            vis_dir=vis_dir,
        )
        val_losses.append(val_metrics['loss'])
        
        # Save checkpoint
        if epoch % config.get('training', {}).get('save_interval', 10) == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'], val_metrics,
                checkpoint_path, config
            )
        
        # Save best model
        metric_key = 'acc' if task_type == 'classification' else 'loss'
        current_metric = val_metrics.get(metric_key, 0.0)
        if (metric_key == 'acc' and current_metric > best_metric) or \
           (metric_key == 'loss' and current_metric < best_metric):
            best_metric = current_metric
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'], val_metrics,
                best_path, config
            )
        
        # Print metrics
        print(f'\nEpoch {epoch}/{num_epochs}:')
        print(f'  Train Loss: {train_metrics["loss"]:.4f}')
        print(f'  Val Loss: {val_metrics["loss"]:.4f}')
        if 'acc' in train_metrics:
            print(f'  Train Acc: {train_metrics["acc"]:.2f}%')
        if 'acc' in val_metrics:
            print(f'  Val Acc: {val_metrics["acc"]:.2f}%')
        if 'routing_ratio' in train_metrics:
            print(f'  Routing Ratio: {train_metrics["routing_ratio"]:.3f}')
        print()
    
    writer.close()
    
    # Save final results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_metric': best_metric,
    }
    if task_type == 'classification':
        # Add accuracy metrics if available
        train_accs = []
        val_accs = []
        best_val_acc = 0.0
        for metrics_dict in train_metrics_list if 'train_metrics_list' in locals() else []:
            if 'acc' in metrics_dict:
                train_accs.append(metrics_dict['acc'])
        for metrics_dict in val_metrics_list if 'val_metrics_list' in locals() else []:
            if 'acc' in metrics_dict:
                val_accs.append(metrics_dict['acc'])
                if metrics_dict['acc'] > best_val_acc:
                    best_val_acc = metrics_dict['acc']
        if train_accs:
            results['train_accs'] = train_accs
        if val_accs:
            results['val_accs'] = val_accs
            results['best_val_acc'] = best_val_acc
    results_path = os.path.join(checkpoint_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Training completed. Best metric: {best_metric:.4f}')
    print(f'Results saved to: {results_path}')


if __name__ == '__main__':
    main()

