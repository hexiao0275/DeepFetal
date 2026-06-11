# -*- coding: utf-8 -*-
import json
import os
import re
from collections import defaultdict
from tqdm import tqdm


# ============== Compatibility loader ==============
def load_records(path):
    """
    Return a normalized list[dict]:
      { "image_path": str, "all_confidences": dict or None,
        "top5_classes": list or None, "classified_as": str or None }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    def norm_one(d):
        return {
            "image_path": d.get("original_path") or d.get("image_path"),
            "all_confidences": d.get("all_confidences"),
            "top5_classes": d.get("top5_classes"),
            "classified_as": d.get("classified_as"),
        }

    if isinstance(data, dict):
        if "details" in data and isinstance(data["details"], list):
            for d in data["details"]:
                rec = norm_one(d)
                if not rec["image_path"]:
                    rec["image_path"] = d.get("path") or d.get("file") or ""
                records.append(rec)
        else:
            # Assume { image_path: result_dict }
            for k, v in data.items():
                rec = {
                    "image_path": v.get("original_path") or v.get("image_path") or k,
                    "all_confidences": v.get("all_confidences"),
                    "top5_classes": v.get("top5_classes"),
                    "classified_as": v.get("classified_as"),
                }
                records.append(rec)
    elif isinstance(data, list):
        for d in data:
            rec = norm_one(d)
            records.append(rec)
    else:
        raise ValueError("Unsupported JSON structure.")

    # Filter out invalid records without paths
    records = [r for r in records if r["image_path"]]
    return records

# ============== Extract case ID ==============
def extract_case_id(image_path):
    """
    Target format example:
      PatientID445682_ExamID1281825_trimester3_FieldID10373
    Compatible with ".../PatientID...FieldIDxxxx/..." path structures.
    Falls back to parent directory name for generic paths.
    """
    image_path = image_path.replace("\\", "/")
    # Try strict FieldID matching first
    m = re.search(r'(PatientID\d+_ExamID\d+_[^/]*?FieldID\d+)', image_path)
    if m:
        return m.group(1)
    # Fallback: match the path segment up to the first '/'
    m2 = re.search(r'(PatientID\d+_ExamID\d+_[^/]+)', image_path)
    if m2:
        return m2.group(1)
    # Generic fallback: use parent directory name
    return os.path.basename(os.path.dirname(image_path))

# ============== Build all_confidences from top5 (fallback) ==============
def build_probs_from_top5(top5):
    """
    top5: [{"class": "NT", "confidence": 0.9}, ...]
    Return dict[str, float] containing only the provided classes.
    """
    if not isinstance(top5, list):
        return None
    out = {}
    for it in top5:
        c = it.get("class")
        p = it.get("confidence")
        if isinstance(c, str) and isinstance(p, (int, float)):
            out[c] = float(p)
    return out or None

def main(args):
    # === Hyperparameters ===
    # threshold = 0.60
    threshold = args.early_filter["common"]["threshold"]
    # max_per_label = 2
    max_per_label = args.early_filter["common"]["max_per_label"]
    # max_total_per_case = 30
    max_total_per_case = args.early_filter["common"]["max_total_per_case"]

    # IN_PATH = "eval_multi_model_parallel/classification_results.json"
    # IN_PATH = "eval_hubeirenming_zhaoyun_classification_results_zhao/classification_results.json"
    IN_PATH = os.path.join(args.early_cls["output_folder"], args.early_cls["out_json"])
    # OUT_JSON = "best_images_per_case_top_2_hubeirenming_zhao.json"
    OUT_JSON = args.early_filter["out_json"]
    # OUT_CASE_IDS = "case_ids_zhao.txt"
    OUT_CASE_IDS = args.early_filter["out_case_ids"]
    # OUT_MISSING = "missing_cases_zhao.txt"
    OUT_MISSING = args.early_filter["out_missing"]
    # copied_root = "./copied_reports_only_eval_224"  # replace with your actual directory if needed
    copied_root = args.image_root

    # ============== Main flow ==============
    records = load_records(IN_PATH)
    print(f"📦 Total records loaded: {len(records)}")

    export_results = {}
    used_images = set()
    skipped_no_probs = 0

    # Collect candidates by case -> label -> (image, probability)
    all_case_label_candidates = defaultdict(lambda: defaultdict(list))

    for rec in tqdm(records, desc="Collecting candidates"):
        image_path = rec["image_path"]
        probs = rec.get("all_confidences")
        if probs is None:
            probs = build_probs_from_top5(rec.get("top5_classes"))

        if not probs:
            skipped_no_probs += 1
            continue

        # Append the _zhao suffix to each key
        probs = {f"{k}_zhao": v for k, v in probs.items()}

        case_id = extract_case_id(image_path)
        if not case_id:
            skipped_no_probs += 1
            continue

        neg_prob = probs.get("neg_zhao", 0.0)

        valid = [
            (label, p)
            for label, p in probs.items()
            if label != "neg_zhao" and isinstance(p, (int, float)) and p > threshold and p > neg_prob
        ]
        if not valid:
            continue

        label, p = max(valid, key=lambda x: x[1])
        all_case_label_candidates[case_id][label].append((image_path, float(p)))

    print(f"ℹ️ Skipped entries (no probs or no case_id): {skipped_no_probs}")

    # Round 1: take one image per label
    for case_id, label_dict in tqdm(all_case_label_candidates.items(), desc="Stage 1: one per label"):
        export_results[case_id] = defaultdict(list)
        for label, images in label_dict.items():
            for path, prob in sorted(images, key=lambda x: x[1], reverse=True):
                if path in used_images:
                    continue
                total_count = sum(len(v) for v in export_results[case_id].values())
                if total_count >= max_total_per_case:
                    break
                export_results[case_id][label].append({"image_path": path, "probability": prob})
                used_images.add(path)
                break

    # Round 2: fill up to 2 per label and at most 30 per case
    for case_id, label_dict in tqdm(all_case_label_candidates.items(), desc="Stage 2: fill up to 2"):
        if case_id not in export_results:
            export_results[case_id] = defaultdict(list)
        for label, images in label_dict.items():
            if len(export_results[case_id][label]) >= max_per_label:
                continue
            for path, prob in sorted(images, key=lambda x: x[1], reverse=True):
                if path in used_images:
                    continue
                total_count = sum(len(v) for v in export_results[case_id].values())
                if total_count >= max_total_per_case:
                    break
                export_results[case_id][label].append({"image_path": path, "probability": prob})
                used_images.add(path)
                break

    # Convert to a plain dict
    export_results = {cid: dict(label_dict) for cid, label_dict in export_results.items()}

    # Validate case image limits
    over_limit = []
    for cid, label_info in export_results.items():
        total = sum(len(v) for v in label_info.values())
        if total > max_total_per_case:
            over_limit.append((cid, total))
            print(f"❌ {cid} has {total} images (> {max_total_per_case})")

    # Summary
    print(f"\n📊 Total cases: {len(export_results)}")
    print(f"✅ Cases within {max_total_per_case}: {len(export_results) - len(over_limit)}")
    print(f"❗ Cases over limit: {len(over_limit)}")

    # Save output
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(export_results, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {OUT_JSON}")

    # Selection statistics
    if export_results:
        selected_counts = [sum(len(images) for images in label_dict.values()) for label_dict in export_results.values()]
        avg_selected = sum(selected_counts) / len(selected_counts)
        max_selected = max(selected_counts)
        min_selected = min(selected_counts)
    else:
        avg_selected = max_selected = min_selected = 0

    print(f"\n📊 Average images per case: {avg_selected:.2f}")
    print(f"🔺 Max images: {max_selected}")
    print(f"🔻 Min images: {min_selected}")
    print(f"🧾 Total cases: {len(export_results)}")

    # Export case IDs
    with open(OUT_CASE_IDS, "w", encoding="utf-8") as f:
        for cid in export_results.keys():
            f.write(cid + "\n")
    print(f"📄 All case IDs saved to {OUT_CASE_IDS}")

    # Track cases present on disk but not selected
    if os.path.isdir(copied_root):
        folder_names = [d for d in os.listdir(copied_root) if os.path.isdir(os.path.join(copied_root, d))]
        print(f"📁 Case folders under {copied_root}: {len(folder_names)}")

        dir_case_ids = set(folder_names)
        final_case_ids = set(export_results.keys())
        missing_cases = dir_case_ids - final_case_ids

        print(f"📉 Missing cases (present in {copied_root} but not retained): {len(missing_cases)}")
        print(f"📋 Missing case examples (up to 10): {list(sorted(missing_cases))[:10]}")

        with open(OUT_MISSING, "w", encoding="utf-8") as f:
            for cid in sorted(missing_cases):
                f.write(cid + "\n")
        print(f"📄 Missing case list saved to {OUT_MISSING}")
    else:
        print(f"⚠️ Directory does not exist: {copied_root} (skipping missing-case stats)")
