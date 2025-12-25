"""Generate visualization plots for training results across different tasks."""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import argparse


def load_training_results(results_dir: str) -> Dict[str, Dict]:
    """
    Load training results from all JSON files in results directory.
    
    Returns:
        Dictionary mapping task_name -> results_dict
    """
    results = {}
    
    # Find all training_results.json files
    pattern = os.path.join(results_dir, "**/training_results.json")
    result_files = glob.glob(pattern, recursive=True)
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract task name from path
            # e.g., checkpoints/cifar10/training_results.json -> cifar10
            path_parts = result_file.split(os.sep)
            task_name = path_parts[-2] if len(path_parts) > 1 else "unknown"
            
            results[task_name] = data
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}")
    
    return results


def plot_training_curves(results: Dict[str, Dict], output_dir: str):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Across Tasks', fontsize=16)
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for task_name, data in results.items():
        if 'train_losses' in data:
            epochs = range(1, len(data['train_losses']) + 1)
            ax1.plot(epochs, data['train_losses'], label=f'{task_name} (train)', linestyle='-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for task_name, data in results.items():
        if 'val_losses' in data:
            epochs = range(1, len(data['val_losses']) + 1)
            ax2.plot(epochs, data['val_losses'], label=f'{task_name} (val)', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy (if available)
    ax3 = axes[1, 0]
    for task_name, data in results.items():
        if 'train_accs' in data:
            epochs = range(1, len(data['train_accs']) + 1)
            ax3.plot(epochs, data['train_accs'], label=f'{task_name} (train)', linestyle='-')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Training Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy (if available)
    ax4 = axes[1, 1]
    for task_name, data in results.items():
        if 'val_accs' in data:
            epochs = range(1, len(data['val_accs']) + 1)
            ax4.plot(epochs, data['val_accs'], label=f'{task_name} (val)', linestyle='--')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Validation Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to: {output_path}")
    plt.close()


def plot_metrics_comparison(results: Dict[str, Dict], output_dir: str):
    """Plot comparison of best metrics across tasks."""
    tasks = []
    metrics = {
        'best_val_acc': [],
        'best_val_loss': [],
    }
    
    for task_name, data in results.items():
        tasks.append(task_name)
        metrics['best_val_acc'].append(data.get('best_val_acc', 0))
        metrics['best_val_loss'].append(data.get('best_val_loss', float('inf')))
    
    if not tasks:
        print("No results found for metrics comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Best Metrics Comparison Across Tasks', fontsize=16)
    
    # Plot 1: Best Validation Accuracy
    ax1 = axes[0]
    valid_accs = [(t, m) for t, m in zip(tasks, metrics['best_val_acc']) if m > 0]
    if valid_accs:
        tasks_acc, accs = zip(*valid_accs)
        bars = ax1.bar(tasks_acc, accs, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Task')
        ax1.set_ylabel('Best Validation Accuracy (%)')
        ax1.set_title('Best Validation Accuracy')
        ax1.set_ylim(0, max(accs) * 1.1 if accs else 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%',
                    ha='center', va='bottom')
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Best Validation Loss
    ax2 = axes[1]
    valid_losses = [(t, m) for t, m in zip(tasks, metrics['best_val_loss']) if m != float('inf')]
    if valid_losses:
        tasks_loss, losses = zip(*valid_losses)
        bars = ax2.bar(tasks_loss, losses, color='coral', alpha=0.7)
        ax2.set_xlabel('Task')
        ax2.set_ylabel('Best Validation Loss')
        ax2.set_title('Best Validation Loss')
        
        # Add value labels on bars
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.4f}',
                    ha='center', va='bottom')
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison to: {output_path}")
    plt.close()


def plot_routing_statistics(results_dir: str, output_dir: str):
    """Plot routing statistics if available."""
    # Try to load routing ratios from log files or results
    # This is a placeholder - you may need to extract routing ratios from training logs
    pass


def generate_summary_table(results: Dict[str, Dict], output_dir: str):
    """Generate a summary table of all results."""
    summary_data = []
    
    for task_name, data in results.items():
        summary_data.append({
            'Task': task_name,
            'Best Val Acc (%)': data.get('best_val_acc', 'N/A'),
            'Best Val Loss': data.get('best_val_loss', 'N/A'),
            'Total Epochs': len(data.get('train_losses', [])),
        })
    
    if not summary_data:
        print("No results found for summary table")
        return
    
    # Create a simple text summary
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Training Results Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for item in summary_data:
            f.write(f"Task: {item['Task']}\n")
            f.write(f"  Best Validation Accuracy: {item['Best Val Acc (%)']}\n")
            f.write(f"  Best Validation Loss: {item['Best Val Loss']}\n")
            f.write(f"  Total Epochs: {item['Total Epochs']}\n")
            f.write("\n")
    
    print(f"Saved summary table to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--results_dir', type=str, default='./checkpoints',
                       help='Directory containing training results JSON files')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualization plots')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print("Loading training results...")
    results = load_training_results(args.results_dir)
    
    if not results:
        print(f"No training results found in {args.results_dir}")
        print("Looking for files matching: **/training_results.json")
        return
    
    print(f"Found results for {len(results)} tasks: {list(results.keys())}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(results, args.output_dir)
    plot_metrics_comparison(results, args.output_dir)
    generate_summary_table(results, args.output_dir)
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()

