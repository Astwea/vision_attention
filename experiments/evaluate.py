"""Evaluation script for vision attention routing models."""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.cifar import get_cifar_dataloaders
from datasets.factory import get_dataloaders, get_dataset_info
from models import create_model
from utils.config import get_config, setup_directories
from utils.flops import calculate_conditional_flops, format_flops, estimate_model_flops
from utils.visualization import save_attention_batch, visualize_attention_heatmap
from utils.metrics import (
    compute_classification_accuracy,
    compute_miou,
    compute_detection_map_coco,
    compute_detection_map_voc,
)


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict,
    save_attention: bool = True,
    vis_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    correct = 0
    total = 0
    all_losses = []
    
    routing_ratios = []
    flops_list = []
    
    all_attention_scores = []
    all_routing_masks = []
    sample_images = []
    sample_patches = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            patches = batch['patches'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            if hasattr(model, 'router'):
                outputs = model(
                    images,
                    patches,
                    return_attention=True,
                    return_stats=True,
                )
                logits = outputs['logits']
                
                # Collect routing stats
                routing_stats = outputs.get('router_stats', {})
                routing_ratios.append(routing_stats.get('routing_ratio', 0.0))
                
                # Estimate FLOPs
                routing_mask = outputs.get('routing_mask')
                if routing_mask is not None:
                    flops_dict = estimate_model_flops(
                        model,
                        images.shape,
                        patches,
                        routing_mask,
                    )
                    flops_list.append(flops_dict['total'] / images.size(0))  # Per image
                
                # Save attention for visualization (first few batches)
                if save_attention and batch_idx < 4:
                    all_attention_scores.append(outputs['attention_scores'].cpu())
                    all_routing_masks.append(outputs['routing_mask'].cpu())
                    sample_images.append(images.cpu())
                    sample_patches.append(patches.cpu())
            else:
                outputs = model(images, patches)
                logits = outputs['logits']
            
            # Loss
            loss = criterion(logits, labels)
            all_losses.append(loss.item())
            
            # Accuracy
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            acc = 100.0 * correct / total
            pbar.set_postfix({'acc': f'{acc:.2f}%'})
    
    # Calculate metrics
    accuracy = 100.0 * correct / total
    avg_loss = np.mean(all_losses)
    avg_routing_ratio = np.mean(routing_ratios) if routing_ratios else 0.0
    avg_flops = np.mean(flops_list) if flops_list else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'routing_ratio': avg_routing_ratio,
        'flops_per_image': avg_flops,
    }
    
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
            prefix='eval',
            num_samples=min(8, images_batch.size(0)),
        )
    
    return metrics


def compare_baselines(
    config: Dict,
    device: torch.device,
    checkpoint_dir: str = './checkpoints',
) -> Dict[str, Dict]:
    """
    Compare all baselines and main model.
    
    Returns:
        Dictionary with results for each model
    """
    # Data loader (use test set)
    dataset_config = config.get('dataset', {})
    _, _, test_loader = get_cifar_dataloaders(
        root=dataset_config.get('root', './data'),
        dataset_name=dataset_config.get('name', 'cifar10'),
        input_size=dataset_config.get('input_size', 32),
        patch_grid_size=dataset_config.get('patch_grid_size', 4),
        batch_size=config.get('evaluation', {}).get('batch_size', 128),
        num_workers=dataset_config.get('num_workers', 4),
        download=False,
    )
    
    # Models to evaluate
    model_names = ['attention_routing', 'full_compute', 'cheap_compute', 'no_routing']
    results = {}
    
    for model_name in model_names:
        print(f'\nEvaluating {model_name}...')
        
        # Create model
        model_config = config.copy()
        model_config['model']['name'] = model_name
        model = create_model(model_name, model_config)
        model = model.to(device)
        
        # Try to load checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded checkpoint from {checkpoint_path}')
        else:
            print(f'Warning: No checkpoint found for {model_name}, using untrained model')
        
        # Evaluate
        vis_dir = os.path.join(checkpoint_dir, 'visualizations', model_name) if config.get('evaluation', {}).get('visualize', True) else None
        
        metrics = evaluate_model(
            model,
            test_loader,
            device,
            model_config,
            save_attention=(model_name == 'attention_routing'),
            vis_dir=vis_dir,
        )
        
        results[model_name] = metrics
        
        print(f'  Accuracy: {metrics["accuracy"]:.2f}%')
        if metrics.get('routing_ratio', 0) > 0:
            print(f'  Routing Ratio: {metrics["routing_ratio"]:.3f}')
        if metrics.get('flops_per_image', 0) > 0:
            print(f'  FLOPs per image: {format_flops(metrics["flops_per_image"])}')
    
    return results


def main():
    """Main evaluation function."""
    from utils.config import parse_args
    
    args = parse_args()
    config = get_config(args.config, args)
    
    # Determine device: use config if specified, but fallback to CPU if CUDA unavailable
    device_name = config.get('device', 'cuda')
    if device_name == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA requested but not available. Using CPU instead.')
        device_name = 'cpu'
    device = torch.device(device_name)
    print(f'Using device: {device}')
    
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', './checkpoints')
    model_name = config.get('model', {}).get('name', 'attention_routing')
    
    # Get task type
    task_type = config.get('task_type', 'classification')
    
    # Load model
    model = create_model(model_name, config, task_type=task_type)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from {checkpoint_path}')
        print(f'Checkpoint epoch: {checkpoint.get("epoch", "unknown")}')
        print(f'Checkpoint accuracy: {checkpoint.get("acc", "unknown"):.2f}%')
    else:
        print(f'Warning: No checkpoint found at {checkpoint_path}')
    
    # Data loader
    dataset_config = config.get('dataset', {})
    dataset_name = dataset_config.get('name', 'cifar10')
    
    # Use factory for unified data loading
    _, _, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        task_type=task_type,
        root=dataset_config.get('root', './data'),
        input_size=dataset_config.get('input_size', 32),
        patch_grid_size=dataset_config.get('patch_grid_size', 4),
        batch_size=config.get('evaluation', {}).get('batch_size', 128),
        num_workers=dataset_config.get('num_workers', 4),
        **dataset_config
    )
    
    # Evaluate based on task type
    vis_dir = os.path.join(checkpoint_dir, 'visualizations', 'evaluation') if config.get('evaluation', {}).get('visualize', True) else None
    
    if task_type == "classification":
        metrics = evaluate_classification(
            model,
            test_loader,
            device,
            config,
            save_attention=config.get('evaluation', {}).get('save_attention_maps', True),
            vis_dir=vis_dir,
        )
        
        # Print results
        print('\n' + '='*50)
        print('Evaluation Results (Classification)')
        print('='*50)
        print(f'Accuracy: {metrics["accuracy"]:.2f}%')
        if 'top5_accuracy' in metrics:
            print(f'Top-5 Accuracy: {metrics["top5_accuracy"]:.2f}%')
        print(f'Loss: {metrics["loss"]:.4f}')
        if metrics.get('routing_ratio', 0) > 0:
            print(f'Average Routing Ratio: {metrics["routing_ratio"]:.3f}')
        if metrics.get('flops_per_image', 0) > 0:
            print(f'Average FLOPs per image: {format_flops(metrics["flops_per_image"])}')
        print('='*50)
    
    elif task_type == "detection":
        metrics = evaluate_detection(
            model,
            test_loader,
            device,
            config,
            dataset_name=dataset_name,
            save_attention=config.get('evaluation', {}).get('save_attention_maps', True),
            vis_dir=vis_dir,
        )
        
        # Print results
        print('\n' + '='*50)
        print('Evaluation Results (Detection)')
        print('='*50)
        print(f'mAP: {metrics.get("mAP", 0):.4f}')
        if 'mAP_50' in metrics:
            print(f'mAP@0.5: {metrics["mAP_50"]:.4f}')
        if 'mAP_75' in metrics:
            print(f'mAP@0.75: {metrics["mAP_75"]:.4f}')
        if metrics.get('routing_ratio', 0) > 0:
            print(f'Average Routing Ratio: {metrics["routing_ratio"]:.3f}')
        print('='*50)
    
    elif task_type in ["segmentation", "instance_segmentation"]:
        metrics = evaluate_segmentation(
            model,
            test_loader,
            device,
            config,
            task_type=task_type,
            save_attention=config.get('evaluation', {}).get('save_attention_maps', True),
            vis_dir=vis_dir,
        )
        
        # Print results
        print('\n' + '='*50)
        print(f'Evaluation Results ({task_type.replace("_", " ").title()})')
        print('='*50)
        if 'miou' in metrics:
            print(f'mIoU: {metrics["miou"]:.4f}')
        if 'mAP' in metrics:
            print(f'mAP: {metrics["mAP"]:.4f}')
        if metrics.get('routing_ratio', 0) > 0:
            print(f'Average Routing Ratio: {metrics["routing_ratio"]:.3f}')
        print('='*50)
    
    # Save results
    results_path = os.path.join(checkpoint_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nResults saved to: {results_path}')
    
    # Compare baselines if requested
    if args.mode == 'eval_all':
        print('\n' + '='*50)
        print('Comparing all baselines...')
        print('='*50)
        all_results = compare_baselines(config, device, checkpoint_dir)
        
        # Save comparison
        comparison_path = os.path.join(checkpoint_dir, 'baseline_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print comparison table
        print('\nBaseline Comparison:')
        print('-' * 70)
        print(f'{"Model":<20} {"Accuracy":<12} {"Routing Ratio":<15} {"FLOPs/img":<15}')
        print('-' * 70)
        for name, metrics in all_results.items():
            acc = metrics.get('accuracy', 0)
            routing = metrics.get('routing_ratio', 0)
            flops = format_flops(metrics.get('flops_per_image', 0)) if metrics.get('flops_per_image', 0) > 0 else 'N/A'
            print(f'{name:<20} {acc:<12.2f} {routing:<15.3f} {flops:<15}')
        print('-' * 70)


def evaluate_classification(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict,
    save_attention: bool = True,
    vis_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate classification model."""
    model.eval()
    all_logits = []
    all_labels = []
    all_losses = []
    routing_ratios = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            patches = batch['patches'].to(device)
            labels = batch['label'].to(device)
            
            if hasattr(model, 'router'):
                outputs = model(images, patches, return_attention=True, return_stats=True)
                logits = outputs['logits']
                routing_ratios.append(outputs.get('router_stats', {}).get('routing_ratio', 0.0))
            else:
                outputs = model(images, patches)
                logits = outputs['logits']
            
            loss = criterion(logits, labels)
            all_losses.append(loss.item())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    # Compute metrics
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    accuracy = compute_classification_accuracy(all_logits, all_labels, top_k=1) * 100
    top5_accuracy = compute_classification_accuracy(all_logits, all_labels, top_k=5) * 100
    avg_loss = np.mean(all_losses)
    avg_routing_ratio = np.mean(routing_ratios) if routing_ratios else 0.0
    
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'loss': avg_loss,
        'routing_ratio': avg_routing_ratio,
    }


def evaluate_detection(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict,
    dataset_name: str,
    save_attention: bool = True,
    vis_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate detection model."""
    model.eval()
    all_predictions = []
    routing_ratios = []
    
    score_threshold = config.get('evaluation', {}).get('score_threshold', 0.05)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            patches = batch['patches'].to(device)
            image_ids = batch['image_id']
            
            if hasattr(model, 'router'):
                outputs = model(
                    images,
                    patches,
                    patch_coords=batch.get('patch_coords'),
                    return_stats=True,
                )
                routing_ratios.append(outputs.get('router_stats', {}).get('routing_ratio', 0.0))
            else:
                outputs = model(images, patches, patch_coords=batch.get('patch_coords'))
            
            cls_logits = outputs['cls_logits']  # (B, N_patches, num_anchors, num_classes)
            reg_preds = outputs['reg_preds']  # (B, N_patches, num_anchors, 4)
            
            # Convert to detection format (simplified - would need proper NMS)
            B = cls_logits.size(0)
            for b in range(B):
                # Get predictions for this image
                cls_logits_b = cls_logits[b].view(-1, cls_logits.size(-1))  # (N_patches * num_anchors, num_classes)
                reg_preds_b = reg_preds[b].view(-1, 4)  # (N_patches * num_anchors, 4)
                
                # Get scores and labels
                scores, labels = torch.softmax(cls_logits_b, dim=1).max(dim=1)
                
                # Filter by score threshold
                valid = scores >= score_threshold
                if valid.sum() > 0:
                    all_predictions.append({
                        'image_id': image_ids[b],
                        'boxes': reg_preds_b[valid].cpu(),
                        'scores': scores[valid].cpu(),
                        'labels': labels[valid].cpu(),
                    })
    
    # Compute mAP (simplified - full implementation would use proper COCO/VOC evaluation)
    # For now, return placeholder
    avg_routing_ratio = np.mean(routing_ratios) if routing_ratios else 0.0
    
    # TODO: Implement full mAP computation using utils.metrics
    return {
        'mAP': 0.0,  # Placeholder
        'routing_ratio': avg_routing_ratio,
    }


def evaluate_segmentation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict,
    task_type: str,
    save_attention: bool = True,
    vis_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate segmentation model."""
    model.eval()
    all_pred_masks = []
    all_gt_masks = []
    routing_ratios = []
    
    dataset_config = config.get('dataset', {})
    num_classes = dataset_config.get('num_classes', 21)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            patches = batch['patches'].to(device)
            gt_masks = batch['mask']
            
            if hasattr(model, 'router'):
                outputs = model(images, patches, return_stats=True)
                routing_ratios.append(outputs.get('router_stats', {}).get('routing_ratio', 0.0))
            else:
                outputs = model(images, patches)
            
            logits = outputs['logits']  # (B, num_classes, H, W)
            pred_masks = logits.argmax(dim=1).cpu()  # (B, H, W)
            
            all_pred_masks.append(pred_masks)
            all_gt_masks.append(gt_masks)
    
    # Compute mIoU
    all_pred_masks = torch.cat(all_pred_masks, dim=0)
    all_gt_masks = torch.cat(all_gt_masks, dim=0)
    
    miou = compute_miou(all_pred_masks, all_gt_masks, num_classes, ignore_index=255)
    avg_routing_ratio = np.mean(routing_ratios) if routing_ratios else 0.0
    
    return {
        'miou': miou,
        'routing_ratio': avg_routing_ratio,
    }


if __name__ == '__main__':
    main()

