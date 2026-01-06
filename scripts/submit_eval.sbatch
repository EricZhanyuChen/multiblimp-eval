#!/bin/bash

# ================= Slurm 配置 =================
#SBATCH --job-name=gemma3_eval        # 任务名称
#SBATCH --partition=gpumedium         # 使用 gpumedium 分区 (最高 24 小时)
#SBATCH --gres=gpu:a100:1             # 申请 1 块 A100 GPU (Gemma 3 12B 推荐)
#SBATCH --nodes=1                     # 申请 1 个节点
#SBATCH --ntasks=1                    # 运行 1 个任务
#SBATCH --cpus-per-task=8             # 申请 8 个 CPU 核心
#SBATCH --mem=64G                     # 申请 64GB 系统内存
#SBATCH --time=12:00:00               # 设置 12 小时运行时间限制
#SBATCH --output=logs/eval_%j.log     # 标准输出日志 (%j 会替换为作业 ID)
#SBATCH --error=logs/eval_%j.err      # 错误输出日志

# ================= 环境准备 =================
# 1. 创建日志目录 (如果不存在)
mkdir -p logs

# 2. 加载 CUDA 模块 (请根据 Habrok 的具体版本调整，通常为 12.x)
module load CUDA/12.1.1 

# 3. 激活您的 Conda 环境
# 注意：请将 'your_env_name' 替换为您实际的环境名称
source  venv/bin/activate

# 4. (可选) 设置 Hugging Face 缓存目录，防止 /home 空间不足
# export HF_HOME="/scratch/$USER/hf_cache"

# ================= 执行任务 =================
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"

# 执行您之前的主循环 Bash 脚本
# 确保 run.sh 文件在当前目录下，并且具有执行权限 (chmod +x run.sh)
bash scripts/run_experiment.sh

echo "Job finished at: $(date)"