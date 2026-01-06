import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to results")
    return parser.parse_args()

def load_data(filepath):
    """读取 Direct 方法的 Good/Bad 分数"""
    good_scores = []
    bad_scores = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # 优先读取 filtered_resps
                    resps = item.get('filtered_resps') or item.get('resps')
                    target = item.get('target')
                    
                    if resps is None or target is None or len(resps) < 2: continue
                    
                    # 兼容嵌套列表
                    val0 = resps[0][0] if isinstance(resps[0], list) else resps[0]
                    val1 = resps[1][0] if isinstance(resps[1], list) else resps[1]
                    
                    s0 = float(val0)
                    s1 = float(val1)
                    
                    if target == 0:
                        good_scores.append(s0)
                        bad_scores.append(s1)
                    else:
                        good_scores.append(s1)
                        bad_scores.append(s0)
                except: continue
    except Exception as e:
        print(f"  [!] Error reading {os.path.basename(filepath)}: {e}")
    return good_scores, bad_scores

def main():
    args = parse_args()
    
    target_langs = ["eng", "heb", "ita", "rus", "wol", "sah"]
    model_sizes = ["1b", "4b", "12b"]
    
    # 视觉配置
    configs = {
        "1b":  {"color": "#7f8c8d", "title": "Gemma-3-1B"},
        "4b":  {"color": "#2980b9", "title": "Gemma-3-4B"},
        "12b": {"color": "#c0392b", "title": "Gemma-3-12B"}
    }

    abs_run_dir = os.path.abspath(args.run_dir)
    
    # === 输出路径 ===
    figures_dir = os.path.join(abs_run_dir, "figures", "log_distribution")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    print(f"[*] Scanning for Subplot Log Plots in: {abs_run_dir}")
    print(f"[*] Output Dir: {figures_dir}")
    
    all_files = glob.glob(os.path.join(abs_run_dir, "*.jsonl"))
    
    for lang in target_langs:
        print(f"\n> Processing Language: {lang.upper()}...")
        
        # 1. 先把三个模型的数据都读出来
        lang_data = {}
        all_scores = [] 
        
        for size in model_sizes:
            target_file = None
            for f in all_files:
                fname = os.path.basename(f).lower()
                # 确保是 Direct 方法，且是对应语言和尺寸
                if "direct" in fname and lang in fname and size in fname:
                    target_file = f
                    break
            
            if target_file:
                good, bad = load_data(target_file)
                if good:
                    lang_data[size] = (bad, good) 
                    all_scores.extend(good + bad)
                    print(f"  [+] Loaded {size}: {len(good)} points")
        
        if not lang_data:
            print(f"  [!] No data found for {lang}, skipping.")
            continue

        # 2. 计算统一的坐标轴范围
        if all_scores:
            min_val = min(all_scores) - 5
            max_val = max(all_scores) + 5
        else:
            continue

        # 3. 创建画布 - 尺寸加大到 (24, 8)
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
        fig.suptitle(f"Model Scaling Analysis: Good vs Bad Log-Prob ({lang.upper()})", fontsize=24, y=1.05)
        
        for i, size in enumerate(model_sizes):
            ax = axes[i]
            
            if size in lang_data:
                bad, good = lang_data[size]
                conf = configs[size]
                
                # 画散点 (稍微把点调大一点 s=20)
                ax.scatter(bad, good, c=conf["color"], alpha=0.4, s=20, edgecolors='none')
                
                # 画对角线
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2.0, label="Random (y=x)")
                
                ax.set_title(conf["title"], fontsize=20, fontweight='bold')
            else:
                ax.set_title(f"{size} (No Data)", fontsize=20, color='gray')

            ax.set_xlabel("Log Prob (Bad Sentence)", fontsize=16)
            if i == 0:
                ax.set_ylabel("Log Prob (Good Sentence)", fontsize=16)
            
            # 刻度字体加大
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            # 强制使用相同的坐标轴范围
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)

        # === 文件名不含 scaling ===
        save_name = f"log_dist_{lang}.png"
        save_path = os.path.join(figures_dir, save_name)
        # DPI 提高到 200
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  [SUCCESS] Saved: {save_name}")
        plt.close()

if __name__ == "__main__":
    main()