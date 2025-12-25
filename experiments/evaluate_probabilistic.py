"""Evaluation script for probabilistic attention routing experiments.

Supports evaluation with different inference modes:
- expectation: Use probability-weighted combination
- top_p: Select top-p% patches for Large MLP
- hard_threshold: Hard threshold at 0.5
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
import json
import argparse
import yaml
from typing import Dict, List, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.cluttered_mnist import get_cluttered_mnist_dataloaders
from models.probabilistic_model import create_probabilistic_model
from utils.probabilistic_visualization import (
    visualize_batch,
    plot_tradeoff_curve,
    visualize_attention_heatmap,
    visualize_routing_decisions,
)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    inference_mode: str = "expectation",
    num_samples: int = 10,
    save_vis: bool = False,
    vis_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate model with specified inference mode.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device
        inference_mode: Inference mode ("expectation", "top_p", "hard_threshold")
        num_samples: Number of samples to visualize
        save_vis: Whether to save visualizations
        vis_dir: Directory to save visualizations
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    correct = 0
    total = 0
    routing_ratios = []
    all_attention_scores = []
    all_routing_masks = []
    sample_images = []
    sample_patches = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Evaluating ({inference_mode})')
        
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
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Routing statistics
            if 'router_stats' in outputs:
                stats = outputs['router_stats']
                routing_ratios.append(stats.get('routing_ratio', 0.0))
            
            # Save samples for visualization
            if batch_idx < 2 and save_vis:
                all_attention_scores.append(outputs['attention_scores'].cpu())
                if 'routing_mask' in outputs:
                    all_routing_masks.append(outputs['routing_mask'].cpu())
                sample_images.append(images.cpu())
                sample_patches.append(patches.cpu())
            
            # Update progress
            pbar.set_postfix({
                'acc': f'{100.0 * correct / total:.2f}%'
            })
    
    accuracy = 100.0 * correct / total
    avg_routing_ratio = np.mean(routing_ratios) if routing_ratios else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'routing_ratio': avg_routing_ratio,
        'inference_mode': inference_mode,
    }
    
    # Save visualizations
    if save_vis and all_attention_scores and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        
        images_batch = torch.cat(sample_images, dim=0)
        patches_batch = torch.cat(sample_patches, dim=0)
        attention_batch = torch.cat(all_attention_scores, dim=0)
        routing_batch = torch.cat(all_routing_masks, dim=0) if all_routing_masks else None
        
        visualize_batch(
            images_batch,
            patches_batch,
            attention_batch,
            routing_batch,
            patch_grid_size=8,
            num_samples=min(num_samples, images_batch.size(0)),
            save_dir=vis_dir,
            prefix=f'eval_{inference_mode}',
        )
    
    return metrics


def evaluate_all_models(
    model_configs: List[Dict],
    dataloader: DataLoader,
    device: torch.device,
    inference_modes: List[str] = ["expectation", "top_p", "hard_threshold"],
    save_vis: bool = True,
    vis_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Evaluate all models with all inference modes.
    
    Args:
        model_configs: List of model configurations
        dataloader: Data loader
        device: Device
        inference_modes: List of inference modes to evaluate
        save_vis: Whether to save visualizations
        vis_dir: Directory to save visualizations
        
    Returns:
        Dictionary mapping (model_name, inference_mode) to metrics
    """
    all_results = {}
    
    for model_config in model_configs:
        model_name = model_config['name']
        checkpoint_path = model_config.get('checkpoint')
        
        print(f'\n{"="*60}')
        print(f'Evaluating model: {model_name}')
        print(f'{"="*60}')
        
        # Create model
        model = create_probabilistic_model(
            model_name=model_name,
            num_patches=64,
            patch_input_dim=192,
            num_classes=10,
            router_mode=model_config.get('router_mode', 'gumbel_softmax'),
            temperature=model_config.get('temperature', 1.0),
            threshold=model_config.get('threshold', 0.5),
        )
        model = model.to(device)
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded checkpoint from: {checkpoint_path}')
        
        # Evaluate with each inference mode
        for inference_mode in inference_modes:
            print(f'\nInference mode: {inference_mode}')
            
            vis_dir_mode = None
            if save_vis and vis_dir:
                vis_dir_mode = os.path.join(vis_dir, f'{model_name}_{inference_mode}')
            
            metrics = evaluate_model(
                model,
                dataloader,
                device,
                inference_mode=inference_mode,
                save_vis=save_vis,
                vis_dir=vis_dir_mode,
            )
            
            key = f'{model_name}_{inference_mode}'
            all_results[key] = metrics
            
            print(f'  Accuracy: {metrics["accuracy"]:.2f}%')
            print(f'  Routing Ratio: {metrics["routing_ratio"]:.3f}')
    
    return all_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate probabilistic attention routing models')
    parser.add_argument('--config', type=str, default='configs/probabilistic_experiment.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='probabilistic',
                        choices=['probabilistic', 'baseline_large', 'baseline_small', 'deterministic'],
                        help='Model name')
    parser.add_argument('--inference_mode', type=str, default='expectation',
                        choices=['expectation', 'top_p', 'hard_threshold', 'all'],
                        help='Inference mode')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--save_vis', action='store_true', help='Save visualizations')
    parser.add_argument('--vis_dir', type=str, default=None, help='Visualization directory')
    parser.add_argument('--eval_all', action='store_true', help='Evaluate all models')
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Set defaults
    device_name = args.device or config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    print(f'Using device: {device}')
    
    # Data loader
    val_loader, _ = get_cluttered_mnist_dataloaders(
        root=config.get('data_root', './data'),
        image_size=64,
        digit_size=14,
        num_clutter_digits=4,
        noise_intensity=0.3,
        patch_grid_size=8,
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    # Inference modes
    if args.inference_mode == 'all':
        inference_modes = ['expectation', 'top_p', 'hard_threshold']
    else:
        inference_modes = [args.inference_mode]
    
    # Visualization directory
    vis_dir = args.vis_dir or config.get('vis_dir', './visualizations/probabilistic/eval')
    
    if args.eval_all:
        # Evaluate all models
        model_configs = [
            {'name': 'baseline_large', 'checkpoint': None},
            {'name': 'baseline_small', 'checkpoint': None},
            {'name': 'deterministic', 'checkpoint': None},
            {'name': 'probabilistic', 'checkpoint': args.checkpoint, 'router_mode': 'gumbel_softmax'},
        ]
        
        all_results = evaluate_all_models(
            model_configs,
            val_loader,
            device,
            inference_modes=inference_modes,
            save_vis=args.save_vis,
            vis_dir=vis_dir,
        )
        
        # Print summary
        print('\n' + '='*60)
        print('Evaluation Summary')
        print('='*60)
        for key, metrics in all_results.items():
            print(f'{key}:')
            print(f'  Accuracy: {metrics["accuracy"]:.2f}%')
            print(f'  Routing Ratio: {metrics["routing_ratio"]:.3f}')
        
        # Plot trade-off curve
        if args.save_vis:
            tradeoff_results = {}
            for key, metrics in all_results.items():
                model_name = key.split('_')[0]  # Extract model name
                if model_name not in tradeoff_results:
                    tradeoff_results[model_name] = {
                        'accuracy': metrics['accuracy'],
                        'routing_ratio': metrics['routing_ratio'],
                    }
            
            plot_tradeoff_curve(
                tradeoff_results,
                save_path=os.path.join(vis_dir, 'tradeoff_curve.png'),
            )
        
        # Save results
        results_path = os.path.join(vis_dir, 'evaluation_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nResults saved to: {results_path}')
    
    else:
        # Evaluate single model
        model = create_probabilistic_model(
            model_name=args.model,
            num_patches=64,
            patch_input_dim=192,
            num_classes=10,
            router_mode=config.get('router_mode', 'gumbel_softmax'),
            temperature=config.get('temperature', 1.0),
            threshold=config.get('threshold', 0.5),
        )
        model = model.to(device)
        
        # Load checkpoint
        if args.checkpoint and os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded checkpoint from: {args.checkpoint}')
        
        # Evaluate
        for inference_mode in inference_modes:
            print(f'\nEvaluating with inference mode: {inference_mode}')
            metrics = evaluate_model(
                model,
                val_loader,
                device,
                inference_mode=inference_mode,
                save_vis=args.save_vis,
                vis_dir=vis_dir,
            )
            
            print(f'Accuracy: {metrics["accuracy"]:.2f}%')
            print(f'Routing Ratio: {metrics["routing_ratio"]:.3f}')


if __name__ == '__main__':
    main()

