"""Generate summary visualization of attention heatmaps across tasks."""

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import argparse


def create_attention_summary(attention_dir: str, output_path: str, max_samples: int = 4):
    """
    Create a summary visualization of attention heatmaps.
    
    Args:
        attention_dir: Directory containing attention visualization images
        output_path: Output path for summary image
        max_samples: Maximum number of samples to include per task
    """
    # Find all attention heatmap images
    heatmap_pattern = os.path.join(attention_dir, "**/*_heatmap_*.png")
    heatmap_files = glob.glob(heatmap_pattern, recursive=True)
    
    if not heatmap_files:
        print(f"未找到attention热力图文件在 {attention_dir}")
        return
    
    # Group by task/dataset
    task_groups = {}
    for filepath in heatmap_files:
        # Extract task/dataset name from path
        path_parts = filepath.split(os.sep)
        # Look for recognizable patterns
        task_name = "unknown"
        for part in path_parts:
            if any(x in part.lower() for x in ['cifar', 'imagenet', 'coco', 'voc', 'classification', 'detection', 'segmentation']):
                task_name = part
                break
        
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append(filepath)
    
    # Sort files to get latest epochs
    for task_name in task_groups:
        task_groups[task_name].sort(reverse=True)  # Get latest first
        task_groups[task_name] = task_groups[task_name][:max_samples]
    
    # Create summary figure
    num_tasks = len(task_groups)
    if num_tasks == 0:
        print("未找到任何任务组")
        return
    
    samples_per_task = max(len(files) for files in task_groups.values())
    fig = plt.figure(figsize=(15, 5 * num_tasks))
    gs = gridspec.GridSpec(num_tasks, samples_per_task, figure=fig, hspace=0.3, wspace=0.3)
    
    row = 0
    for task_name, files in sorted(task_groups.items()):
        for col, filepath in enumerate(files):
            ax = fig.add_subplot(gs[row, col])
            
            # Load and display image
            img = Image.open(filepath)
            ax.imshow(img)
            ax.axis('off')
            
            # Set title for first column
            if col == 0:
                ax.set_title(f'{task_name}', fontsize=12, fontweight='bold', pad=10)
            
            # Add filename info
            filename = os.path.basename(filepath)
            ax.text(0.02, 0.98, filename, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        row += 1
    
    plt.suptitle('Attention Heatmaps Summary Across Tasks', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Attention热力图总结已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='生成attention热力图总结')
    parser.add_argument('--attention_dir', type=str, default='./visualizations',
                       help='包含attention可视化的目录')
    parser.add_argument('--output', type=str, default='./visualizations/attention_summary.png',
                       help='输出文件路径')
    parser.add_argument('--max_samples', type=int, default=4,
                       help='每个任务的最大样本数')
    
    args = parser.parse_args()
    
    create_attention_summary(args.attention_dir, args.output, args.max_samples)


if __name__ == '__main__':
    main()

