import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Combined 4-Method Matrix Plot")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to results")
    parser.add_argument("--model_name", type=str, required=True, help="Model size string (e.g., 12b-it)")
    return parser.parse_args()

def load_method_data(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # 优先用 filtered_resps，如果没有则用 resps
                    resps = item.get('filtered_resps') or item.get('resps')
                    target = item.get('target')
                    doc_id = item.get('doc_id', 'unknown')
                    
                    if resps is None or target is None or len(resps) < 2: 
                        continue
                    
                    # === 核心修复：处理嵌套列表 [[score, bool], ...] ===
                    val0 = resps[0]
                    val1 = resps[1]
                    
                    # 如果元素是列表（例如 [-86.0, false]），取第一个值
                    if isinstance(val0, list): val0 = val0[0]
                    if isinstance(val1, list): val1 = val1[0]
                    
                    # 转为 float
                    s0 = float(val0)
                    s1 = float(val1)
                    
                    # 计算 Delta (Good - Bad)
                    # target 是正确答案的索引 (0 或 1)
                    delta = (s0 - s1) if target == 0 else (s1 - s0)
                    
                    data.append({'doc_id': doc_id, 'delta': delta})
                    
                except Exception as inner_e:
                    # 静默跳过坏行
                    continue
                    
    except Exception as e:
        print(f"      [!] Error reading {os.path.basename(filepath)}: {e}")
        
    return pd.DataFrame(data)

def main():
    args = parse_args()
    abs_run_dir = os.path.abspath(args.run_dir)
    figures_dir = os.path.join(abs_run_dir, "figures")
    if not os.path.exists(figures_dir): os.makedirs(figures_dir)

    print(f"[*] Running NESTED-FIX Version. Scanning: {abs_run_dir}")

    target_langs = ["eng", "heb", "ita", "rus", "wol", "sah"]
    all_files = glob.glob(os.path.join(abs_run_dir, "*.jsonl"))
    
    # 2. 建立索引 (Split Logic - 已验证有效)
    file_map = {}
    
    for f_path in all_files:
        fname = os.path.basename(f_path)
        if args.model_name.lower() not in fname.lower(): continue
            
        clean_name = fname.replace(".jsonl", "")
        parts = clean_name.split('_')
        
        found_lang = None
        lang_index = -1
        for i, part in enumerate(parts):
            if part.lower() in target_langs:
                found_lang = part.lower()
                lang_index = i
                break
        
        if found_lang and lang_index > 1:
            method_str = "_".join(parts[1:lang_index])
            if found_lang not in file_map: file_map[found_lang] = {}
            file_map[found_lang][method_str] = f_path

    # 3. 绘图流程
    display_map = {
        "direct": "Direct",
        "prompt": "Eng_Prompt",
        "native_prompt": "Native",
        "meta": "Meta"
    }

    print(f"[*] Parsing complete. Found files for: {list(file_map.keys())}")
    print("-" * 40)

    for lang in target_langs:
        print(f"\n> Processing: {lang.upper()}...")
        
        if lang not in file_map:
            print(f"  [!] No parsed files for {lang}.")
            continue
            
        lang_files = file_map[lang]
        method_dfs = []
        found_labels = []

        for m_key, m_label in display_map.items():
            if m_key in lang_files:
                f_path = lang_files[m_key]
                df = load_method_data(f_path)
                
                if not df.empty:
                    df = df.rename(columns={'delta': m_label})
                    method_dfs.append(df)
                    found_labels.append(m_label)
                else:
                    print(f"  [!] Data extraction failed for {m_label} (Empty DataFrame)")
        
        if len(method_dfs) < 2:
            print(f"  [STOP] Need at least 2 methods to plot, found {len(method_dfs)} ({found_labels})")
            continue

        # 合并数据
        final_df = method_dfs[0]
        for next_df in method_dfs[1:]:
            temp_merged = pd.merge(final_df, next_df, on='doc_id', how='inner')
            # 容错机制：ID对不上就硬拼
            if len(temp_merged) < (len(final_df) * 0.8):
                print(f"  [Warn] doc_id mismatch for {lang}. Fallback to index alignment.")
                final_df = final_df.reset_index(drop=True)
                next_df = next_df.reset_index(drop=True)
                final_df = pd.concat([final_df, next_df.drop(columns=['doc_id'], errors='ignore')], axis=1)
            else:
                final_df = temp_merged

        if 'doc_id' in final_df.columns: final_df = final_df.drop(columns=['doc_id'])
        final_df = final_df.dropna()
        
        # 绘图
        plt.figure(figsize=(12, 12))
        sns.set_theme(style="whitegrid", font_scale=1.1)
        
        # 定义颜色：对角线(蓝色填充)，散点(深蓝)，回归线(红色)
        g = sns.pairplot(
            final_df, 
            diag_kind="kde", 
            kind="reg", 
            plot_kws={'scatter_kws': {'alpha': 0.3, 's': 10, 'color': '#34495e'}, 'line_kws': {'color': '#e74c3c', 'lw': 1.5}},
            diag_kws={'fill': True, 'color': '#3498db'},
            corner=False
        )
        
        g.fig.suptitle(f"Consistency Matrix: {lang.upper()} ({args.model_name})", y=1.02, fontsize=16)
        save_name = f"{lang}_{args.model_name.replace('/', '_')}_matrix.png"
        g.savefig(os.path.join(figures_dir, save_name), bbox_inches='tight', dpi=150)
        print(f"  [SUCCESS] Saved: {save_name} (Methods: {found_labels}, Samples: {len(final_df)})")
        plt.close('all')

if __name__ == "__main__":
    main()