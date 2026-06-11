# -*- coding: utf-8 -*-
import json
import os
import re
import ijson
import decimal
from collections import defaultdict
from tqdm import tqdm



# === Convert Decimal to float ===
def convert_decimal(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    return obj


# === Stream a large JSON object (dict: {path: {...}}) ===
def stream_load_big_dict_json(path):
    """
    Expected JSON format:
    {
        "path1": {...},
        "path2": {...},
        ...
    }
    Parse it item by item with ijson.kvitems.
    """
    with open(path, "r", encoding="utf-8") as f:
        for key, value in ijson.kvitems(f, ""):
            # Convert Decimal values to float
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, decimal.Decimal):
                        value[k] = float(v)
                    elif isinstance(v, dict):
                        # Nested dict Decimal values
                        for kk, vv in v.items():
                            if isinstance(vv, decimal.Decimal):
                                v[kk] = float(vv)

            yield key, value

def main(args):
    # === Settings ===
    # threshold = 0.30
    threshold = args.normal_filter["common"]["threshold"]
    # max_per_label = 2
    max_per_label = args.normal_filter["common"]["max_per_label"]
    # max_total_per_case = 30
    max_total_per_case = args.normal_filter["common"]["max_total_per_case"]

    # INPUT_JSON = "2_1_classification_results_with_probabilities_224_eval.json"
    INPUT_JSON = args.normal_cls["out_json"]

    OUT_CASE_IDS = args.normal_filter["out_case_ids"]
    OUT_MISSING = args.normal_filter["out_missing"]

    # === Initialize structures ===
    export_results = {}
    used_images = set()
    all_case_label_candidates = defaultdict(lambda: defaultdict(list))


    # === Extract case ID ===
    def extract_case_id(image_path):
        match = re.search(r'(PatientID\d+_ExamID\d+)', image_path)
        if match:
            return match.group(1)
        return os.path.basename(os.path.dirname(image_path))


    # === Main flow: stream parse and select ===
    print("🚀 Streaming huge JSON...")
    total_loaded = 0

    for image_path, result in tqdm(stream_load_big_dict_json(INPUT_JSON), desc="Reading JSON"):
        total_loaded += 1

        probabilities = result.get("probabilities", {})
        if not probabilities:
            continue

        # Normalize Decimal values
        probabilities = {k: (float(v) if isinstance(v, decimal.Decimal) else v)
                         for k, v in probabilities.items()}

        case_id = extract_case_id(image_path)
        if not case_id:
            continue

        neg_prob = probabilities.get("neg", 0.0)

        valid_labels = [
            (label, prob)
            for label, prob in probabilities.items()
            if label != "neg" and isinstance(prob, (float, int)) and prob > threshold
        ]

        if not valid_labels:
            continue

        valid_labels.sort(key=lambda x: x[1], reverse=True)
        label, prob = valid_labels[0]
        all_case_label_candidates[case_id][label].append((image_path, float(prob)))

    print(f"📦 Total images processed: {total_loaded}")


    # === Round 1: select 1 image per label ===
    for case_id, label_dict in tqdm(all_case_label_candidates.items(), desc="Stage 1"):
        export_results[case_id] = defaultdict(list)

        for label, images in label_dict.items():
            for path, prob in sorted(images, key=lambda x: x[1], reverse=True):
                if path in used_images:
                    continue
                if sum(len(v) for v in export_results[case_id].values()) >= max_total_per_case:
                    break

                export_results[case_id][label].append({"image_path": path, "probability": float(prob)})
                used_images.add(path)
                break


    # === Round 2: fill up to 2 images per label ===
    for case_id, label_dict in tqdm(all_case_label_candidates.items(), desc="Stage 2"):
        for label, images in label_dict.items():
            if len(export_results[case_id][label]) >= max_per_label:
                continue

            for path, prob in sorted(images, key=lambda x: x[1], reverse=True):
                if path in used_images:
                    continue
                if sum(len(v) for v in export_results[case_id].values()) >= max_total_per_case:
                    break

                export_results[case_id][label].append({"image_path": path, "probability": float(prob)})
                used_images.add(path)
                break


    # === Save JSON ===
    export_results = {cid: dict(label_info) for cid, label_info in export_results.items()}

    # with open("best_images_per_case_top_2_hubeirenming.json", "w", encoding="utf-8") as f:
    with open(args.normal_filter["out_json"], "w", encoding="utf-8") as f:
        json.dump(export_results, f, ensure_ascii=False, indent=2, default=convert_decimal)

    print(f"💾 Saved successfully: {args.normal_filter['out_json']}")


    # === Statistics ===
    selected_counts = [
        sum(len(v) for v in label_info.values())
        for label_info in export_results.values()
    ]
    if selected_counts:
        print("Average images per case:", sum(selected_counts) / len(selected_counts))
    else:
        print("Warning: No images selected")


    # === Write case IDs ===
    with open(OUT_CASE_IDS, "w", encoding="utf-8") as f:
        for cid in export_results.keys():
            f.write(cid + "\n")

    print(f"📄 Case IDs saved to {OUT_CASE_IDS}")


    # === Missing cases ===
    copied_root = args.image_root
    folder_names = [
        d for d in os.listdir(copied_root)
        if os.path.isdir(os.path.join(copied_root, d))
    ]

    dir_case_ids = set(folder_names)
    final_case_ids = set(export_results.keys())
    missing_cases = dir_case_ids - final_case_ids

    print("📉 Missing case count:", len(missing_cases))
    print("Examples:", list(missing_cases)[:10])

    with open(OUT_MISSING, "w", encoding="utf-8") as f:
        for cid in sorted(missing_cases):
            f.write(cid + "\n")

    print(f"📄 Missing case list saved to {OUT_MISSING}")

