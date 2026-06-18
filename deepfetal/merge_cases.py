# -*- coding: utf-8 -*-
import json
from collections import defaultdict


def main(args):
    # === Paths ===
    # json_normal_path = "best_images_per_case_top_2.json"
    # json_normal_path = args.merge["json_normal_path"]
    json_normal_path = args.normal_filter["out_json"]
    # json_early_path  = "best_images_per_case_top_2.json"
    # json_early_path  = args.merge["json_early_path"]
    json_early_path  = args.early_filter["out_json"]
    # output_path      = "filtered_best_images.json"
    output_path      = args.merge["output_json"]

    # === Parameters ===
    # max_per_label = 2
    max_per_label = args.merge["common"]["max_per_label"]
    # max_total_per_case = 30
    max_total_per_case = args.merge["common"]["max_total_per_case"]

    # === Read JSON ===
    with open(json_normal_path, "r", encoding="utf-8") as f:
        data_normal = json.load(f)
    with open(json_early_path, "r", encoding="utf-8") as f:
        data_early = json.load(f)

    is_early_case = bool(args.is_early)

    # === Selection logic: no image-number filtering ===
    used_images = set()
    result = {}

    def add_from_source(case_id: str, source: dict, case_images: defaultdict):
        """Fill case_images from source[case_id], keeping <=2 per label and <=30 per case with deduplication."""
        if case_id not in source:
            return
        def total_now():
            return sum(len(v) for v in case_images.values())

        for label, img_list in source[case_id].items():
            if len(case_images[label]) >= max_per_label:
                continue
            # Sort by probability descending
            for img in sorted(img_list, key=lambda x: x.get("probability", 0), reverse=True):
                path = img.get("image_path", "")
                if path in used_images:
                    continue
                if len(case_images[label]) < max_per_label and total_now() < max_total_per_case:
                    case_images[label].append(img)
                    used_images.add(path)
                if len(case_images[label]) >= max_per_label or total_now() >= max_total_per_case:
                    break
            if total_now() >= max_total_per_case:
                break

    # === Main loop: iterate over normal case IDs ===
    for case_id in data_normal.keys():
        case_images = defaultdict(list)
        if is_early_case:
            add_from_source(case_id, data_early, case_images)
            add_from_source(case_id, data_normal, case_images)
        else:
            add_from_source(case_id, data_normal, case_images)

        if any(len(v) > 0 for v in case_images.values()):
            result[case_id] = dict(case_images)

    # === Save ===
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # === Statistics ===
    selected_counts = [sum(len(v) for v in label_dict.values()) for label_dict in result.values()]
    avg_selected = (sum(selected_counts) / len(selected_counts)) if selected_counts else 0
    max_selected = max(selected_counts) if selected_counts else 0
    min_selected = min(selected_counts) if selected_counts else 0

    early_cases = len(result) if is_early_case else 0
    normal_cases = 0 if is_early_case else len(result)

    print(f"\n📊 Avg selected per patient: {avg_selected:.2f}")
    print(f"🔺 Max selected: {max_selected}")
    print(f"🔻 Min selected: {min_selected}")
    print(f"🟡 Total patients: {len(result)}")
    print(f"🟢 Early pregnancy patients: {early_cases}")
    print(f"🔵 Mid/late pregnancy patients: {normal_cases}")
    print(f"✅ Done. Saved to: {output_path}")

    # === Diagnose normal cases that ended up with 0 images after merge ===
    print("\n⚠️ The following cases are present in normal JSON but ended up with 0 images:")
    empty_list = []
    for case_id, label_dict in data_normal.items():
        if case_id not in result or sum(len(v) for v in result[case_id].values()) == 0:
            empty_list.append(case_id)
    print(f"Total: {len(empty_list)}")
    for cid in empty_list[:30]:
        print(cid)
    if len(empty_list) > 30:
        print(f"... and {len(empty_list) - 30} more")
