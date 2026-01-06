import argparse
import os
import json
import csv
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Summary CSV from JSON results")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory containing result .json files")
    return parser.parse_args()

def main():
    args = parse_args()
    target_dir = args.target_dir
    
    # Output CSV path
    output_csv = os.path.join(target_dir, "final_summary.csv")
    
    # List to hold all records
    records = []
    
    print(f"[*] Scanning directory: {target_dir}")
    
    # Find all result json files (exclude the summary itself)
    json_files = glob.glob(os.path.join(target_dir, "results_*.json"))
    
    if not json_files:
        print("[!] No results_*.json files found.")
        return

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract Metadata
            meta = data.get("custom_metadata", {})
            lang = meta.get("lang", "unknown")
            method = meta.get("method", "unknown")
            model_name = meta.get("model_name", "unknown").split("/")[-1] # Shorten model name
            
            # Extract Score
            # Structure usually: results -> task_name -> acc
            results = data.get("results", {})
            if not results:
                continue
                
            # Find the first key in results (usually the task name)
            task_name = next(iter(results))
            metrics = results[task_name]
            
            # Prioritize acc_norm, fallback to acc
            acc = metrics.get("acc,none", metrics.get("acc", 0.0))
            if isinstance(acc, str): # Handle formatting edge cases
                acc = float(acc)
                
            records.append({
                "Language": lang,
                "Model": model_name,
                "Parameters": "12B" if "12b" in model_name else ("4B" if "4b" in model_name else "1B"),
                "Method": method,
                "Accuracy": round(acc, 4)
            })
            
        except Exception as e:
            print(f"[!] Failed to parse {json_file}: {e}")

    # Sort records for better readability
    # Order by: Lang -> Model (Size) -> Method
    records.sort(key=lambda x: (x["Language"], x["Model"], x["Method"]))

    # Write to CSV
    if records:
        keys = ["Language", "Model", "Parameters", "Method", "Accuracy"]
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        
        print(f"[*] Summary table generated successfully: {output_csv}")
    else:
        print("[!] No valid records found to write.")

if __name__ == "__main__":
    main()