import argparse
import lm_eval
from lm_eval import tasks, utils
from lm_eval.api.registry import get_model 
import json
import os
import yaml 
from datetime import datetime
import random
import torch # Imported for debug prints

# Set random seed for reproducibility in Python logic
random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="MultiBLiMP Evaluation Tool")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    # [UPDATED] Added 'native_prompt' to choices
    parser.add_argument("--method", type=str, required=True, choices=["direct", "prompt", "meta", "native_prompt"])
    parser.add_argument("--batch_size", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom directory to save results")
    return parser.parse_args()

def generate_yaml_config(lang, method):
    lang_map = {
        "heb": "Hebrew", "eng": "English", "deu": "German", 
        "fra": "French", "ita": "Italian", "spa": "Spanish", 
        "rus": "Russian", "wol": "Wolof", "sah": "Yakut"
    }
    full_lang_name = lang_map.get(lang, lang)
    
    custom_task_name = f"multiblimp_{lang}_{method}_custom"
    
    # -------------------------------------------------------------
    # Maps ISO code -> Native Language Instruction
    # -------------------------------------------------------------
    native_prompts = {
        "eng": "The following is a sentence in English. ",
        "heb": "המשפט הבא הוא בעברית. ",
        "deu": "Der folgende Satz ist auf Deutsch. ",
        "fra": "Voici une phrase en français. ",
        "ita": "La seguente è una frase in italiano. ",
        "spa": "La siguiente es una oración en español. ",
        "rus": "Ниже приведено предложение на русском языке. ",
        # Wolof & Yakut are approximated. 
        # If the model hasn't seen these specific phrases, performance might vary.
        "wol": "Lii ab baat la ci Wolof. ", 
        "sah": "Бу саха тылынан этии. " 
    }

    # Logic: Use sentence length parity to randomize answer position.
    jinja_logic = (
        "{% set flip = (sen|length % 2) == 1 %}"
        "{% set choices = [wrong_sen, sen] if flip else [sen, wrong_sen] %}"
    )

    base_config = {
        "task": custom_task_name,
        "tag": "multiblimp_custom",
        "dataset_path": "jumelet/multiblimp",
        "dataset_name": lang,
        "output_type": "multiple_choice",
        "test_split": "train",
        "num_fewshot": 0,
        "metadata": {"version": 1.0},
        "metric_list": [
            {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
            {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True}
        ]
    }

    base_config["doc_to_target"] = jinja_logic + "{{1 if flip else 0}}"

    # --------------Method-specific configuration-------------
    if method == "direct":
        base_config["doc_to_text"] = ""
        base_config["doc_to_choice"] = jinja_logic + "{{choices}}"
    
    elif method == "prompt":
        # Standard English Prompt
        base_config["doc_to_text"] = f"The following is a sentence in {full_lang_name}. "
        base_config["doc_to_choice"] = jinja_logic + "{{choices}}"

    elif method == "native_prompt":
        # Fallback to English structure if language is not in dict
        prompt_text = native_prompts.get(lang, f"The following is a sentence in {full_lang_name}. ")
        base_config["doc_to_text"] = prompt_text
        base_config["doc_to_choice"] = jinja_logic + "{{choices}}"
    
    elif method == "meta":
        base_config["doc_to_text"] = (
            jinja_logic + 
            f"Which of the following sentences is grammatically correct in {full_lang_name}? "
            "Only respond with 1 or 2 as your answer.\n\n"
            "1. {{choices[0]}}\n"
            "2. {{choices[1]}}\n\n"
            "Answer: "
        )
        base_config["doc_to_choice"] = ["1", "2"]

    # Generate temporary YAML file
    yaml_filename = f"temp_task_{lang}_{method}.yaml"
    yaml_abs_path = os.path.abspath(yaml_filename)
    
    print(f"\n[DEBUG] Generating YAML content for {method}...")
    
    with open(yaml_abs_path, "w", encoding="utf-8") as f:
        yaml.dump(base_config, f, allow_unicode=True, sort_keys=False)
    
    return yaml_abs_path, custom_task_name

def main():
    args = parse_args()

    # [DEBUG] GPU Visibility Check
    if torch.cuda.is_available():
        print(f"[DEBUG] CUDA is available. GPU Count: {torch.cuda.device_count()}")
    else:
        print("[!] WARNING: CUDA is NOT available. Model will load on CPU (very slow).")
    
    # 1. Generate configuration
    yaml_abs_path, custom_task_name = generate_yaml_config(args.lang, args.method)
    yaml_dir = os.path.dirname(yaml_abs_path)
    
    # 2. Load task
    task_manager = tasks.TaskManager(include_path=yaml_dir)
    try:
        task_dict = task_manager.load_task_or_group([custom_task_name])
    except Exception as e:
        print(f"[!] Error loading task: {e}")
        if os.path.exists(yaml_abs_path):
            os.remove(yaml_abs_path)
        return

    # 3. Load model
    print(f"[*] Loading Model: {args.model} ...")
    try:
        HFModel = get_model("hf")
        lm_obj = HFModel(
            pretrained=args.model,
            dtype=args.dtype,
            trust_remote_code=True,
            batch_size=args.batch_size,
            # [CORRECTED] Use device_map="auto" for multi-GPU sharding.
            # Do NOT use device="cuda:0" here.
            device_map="auto"
        )
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        if os.path.exists(yaml_abs_path):
            os.remove(yaml_abs_path)
        return

    # 4. Evaluation
    print(f"[*] Starting Evaluation...")
    try:
        raw_results = lm_eval.evaluate(lm=lm_obj, task_dict=task_dict, limit=None)
    except Exception as e:
        print(f"[!] Error during evaluation: {e}")
        if os.path.exists(yaml_abs_path):
            os.remove(yaml_abs_path)
        return

    # 5. Result Management
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    
    if args.output_dir:
        base_results_dir = args.output_dir
    else:
        base_results_dir = os.path.join(project_root, "results")
    
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)

    safe_model_name = args.model.replace("/", "_")
    
    samples_filename = f"samples_{args.method}_{args.lang}_{safe_model_name}.jsonl"
    summary_filename = f"results_{args.method}_{args.lang}_{safe_model_name}.json"

    samples_path = os.path.join(base_results_dir, samples_filename)
    summary_path = os.path.join(base_results_dir, summary_filename)
    
    # Save Samples
    if "samples" in raw_results:
        with open(samples_path, "w", encoding="utf-8") as f:
            if custom_task_name in raw_results["samples"]:
                for sample in raw_results["samples"][custom_task_name]:
                    json.dump(sample, f, ensure_ascii=False)
                    f.write("\n")

    # Save Summary
    output_data = raw_results.copy()
    if "samples" in output_data:
        del output_data["samples"]
    
    output_data["custom_metadata"] = {
        "lang": args.lang,
        "method": args.method,
        "config_source": f"Generated YAML {yaml_abs_path}",
        "task_name": custom_task_name,
        "model_name": args.model,
        "note": "Randomized via length-parity check"
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False, default=str)

    print(f"[*] Results saved to: {summary_path}")
    
    if os.path.exists(yaml_abs_path):
        os.remove(yaml_abs_path)

if __name__ == "__main__":
    main()