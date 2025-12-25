#!/bin/bash

# 多任务训练和评估脚本
# 用于训练不同任务并在训练完成后生成可视化指标图

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DATA_DIR="$PROJECT_ROOT/data"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
RESULTS_DIR="$PROJECT_ROOT/results"
VIS_DIR="$PROJECT_ROOT/visualizations"

# 创建必要的目录
mkdir -p "$RESULTS_DIR"
mkdir -p "$VIS_DIR"

# 激活conda环境（如果使用conda）
# conda activate your_env_name  # 取消注释并替换为你的环境名

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}多任务训练和评估脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查并下载数据集（如果需要）
echo -e "${YELLOW}检查数据集...${NC}"
python utils/download_datasets.py --dataset all --root "$DATA_DIR" 2>&1 | tee "$RESULTS_DIR/dataset_download.log" || {
    echo -e "${YELLOW}数据集下载脚本运行完成（某些数据集可能需要手动下载）${NC}"
}


# 定义要训练的任务列表
# 格式: "任务类型:数据集:配置文件"
TASKS=(
    "classification:cifar10:configs/default.yaml"
    "classification:imagenet:configs/imagenet.yaml"
    "detection:coco:configs/coco_detection.yaml"
    "detection:voc:configs/voc_detection.yaml"
    "segmentation:voc:configs/voc_segmentation.yaml"
    "instance_segmentation:coco:configs/coco_instance_seg.yaml"
)

# 函数：训练单个任务
train_task() {
    local task_type=$1
    local dataset=$2
    local config_file=$3
    
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}开始训练: $task_type - $dataset${NC}"
    echo -e "${YELLOW}配置文件: $config_file${NC}"
    echo -e "${YELLOW}========================================${NC}\n"
    
    # 检查配置文件是否存在
    if [ ! -f "$PROJECT_ROOT/$config_file" ]; then
        echo -e "${RED}错误: 配置文件 $config_file 不存在${NC}"
        return 1
    fi
    
    # 训练模型
    cd "$PROJECT_ROOT"
    python experiments/train_multi_task.py \
        --config "$config_file" \
        --task_type "$task_type" \
        2>&1 | tee "$RESULTS_DIR/train_${task_type}_${dataset}.log"
    
    # 检查训练是否成功
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}训练完成: $task_type - $dataset${NC}"
    else
        echo -e "${RED}训练失败: $task_type - $dataset${NC}"
        return 1
    fi
}

# 函数：评估单个任务
evaluate_task() {
    local task_type=$1
    local dataset=$2
    local config_file=$3
    
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}开始评估: $task_type - $dataset${NC}"
    echo -e "${YELLOW}========================================${NC}\n"
    
    cd "$PROJECT_ROOT"
    python experiments/evaluate.py \
        --config "$config_file" \
        --task_type "$task_type" \
        2>&1 | tee "$RESULTS_DIR/eval_${task_type}_${dataset}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}评估完成: $task_type - $dataset${NC}"
    else
        echo -e "${RED}评估失败: $task_type - $dataset${NC}"
    fi
}

# 函数：生成可视化图表
generate_visualizations() {
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}生成可视化图表${NC}"
    echo -e "${YELLOW}========================================${NC}\n"
    
    cd "$PROJECT_ROOT"
    
    # 检查是否有Python可视化脚本，如果没有则创建一个
    if [ ! -f "$PROJECT_ROOT/experiments/visualize_results.py" ]; then
        echo -e "${YELLOW}创建可视化脚本...${NC}"
        python << 'EOF'
# 这里会创建一个可视化脚本
EOF
        # 我们会在下面创建这个脚本
    fi
    
    # 运行可视化脚本
    python experiments/visualize_results.py \
        --results_dir "$RESULTS_DIR" \
        --output_dir "$VIS_DIR" \
        2>&1 | tee "$RESULTS_DIR/visualization.log"
    
    echo -e "${GREEN}可视化图表已生成到: $VIS_DIR${NC}"
}

# 主函数：训练所有任务
main() {
    local start_time=$(date +%s)
    
    # 解析命令行参数
    TRAIN_ONLY=false
    EVAL_ONLY=false
    SKIP_VIS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --train-only)
                TRAIN_ONLY=true
                shift
                ;;
            --eval-only)
                EVAL_ONLY=true
                shift
                ;;
            --skip-vis)
                SKIP_VIS=true
                shift
                ;;
            --tasks)
                shift
                # 使用自定义任务列表
                TASKS=("$@")
                break
                ;;
            *)
                echo -e "${RED}未知参数: $1${NC}"
                echo "用法: $0 [--train-only] [--eval-only] [--skip-vis] [--tasks task1 task2 ...]"
                exit 1
                ;;
        esac
    done
    
    # 训练阶段
    if [ "$EVAL_ONLY" = false ]; then
        echo -e "${GREEN}开始训练阶段...${NC}\n"
        
        for task_config in "${TASKS[@]}"; do
            IFS=':' read -r task_type dataset config_file <<< "$task_config"
            
            # 检查数据集是否存在（可选检查）
            # 这里可以根据需要添加数据集存在性检查
            
            train_task "$task_type" "$dataset" "$config_file"
            
            # 如果训练失败，记录但继续下一个任务
            if [ $? -ne 0 ]; then
                echo -e "${RED}任务 $task_type - $dataset 训练失败，跳过评估${NC}"
                continue
            fi
        done
        
        echo -e "\n${GREEN}所有训练任务完成！${NC}\n"
    fi
    
    # 评估阶段
    if [ "$TRAIN_ONLY" = false ]; then
        echo -e "${GREEN}开始评估阶段...${NC}\n"
        
        for task_config in "${TASKS[@]}"; do
            IFS=':' read -r task_type dataset config_file <<< "$task_config"
            evaluate_task "$task_type" "$dataset" "$config_file"
        done
        
        echo -e "\n${GREEN}所有评估任务完成！${NC}\n"
    fi
    
    # 可视化阶段
    if [ "$SKIP_VIS" = false ]; then
        generate_visualizations
        
        # 生成attention热力图总结
        echo -e "\n${YELLOW}生成attention热力图总结...${NC}"
        python experiments/visualize_attention_summary.py \
            --attention_dir "$VIS_DIR" \
            --output "$VIS_DIR/attention_summary.png" \
            --max_samples 4 \
            2>&1 | tee -a "$RESULTS_DIR/visualization.log" || {
            echo -e "${YELLOW}Attention热力图总结生成完成（某些任务可能没有attention输出）${NC}"
        }
    fi
    
    # 计算总时间
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}所有任务完成！${NC}"
    echo -e "${GREEN}总耗时: ${hours}小时 ${minutes}分钟 ${seconds}秒${NC}"
    echo -e "${GREEN}结果保存在: $RESULTS_DIR${NC}"
    echo -e "${GREEN}可视化图表在: $VIS_DIR${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

# 运行主函数
main "$@"

