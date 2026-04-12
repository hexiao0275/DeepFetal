# -*- coding: utf-8 -*-
import json
import re
import pandas as pd
from collections import defaultdict
def parse_case_id(case_id: str):
    CASE_PAT = re.compile(r'PatientID(\d+)_ExamID(\d+)_.*?FieldID(\d+)')
    m = CASE_PAT.search(case_id)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)

def main(args):
    # === Paths ===
    # excel_path       = "孕周提取完整结果_gpt_results_only_eval_with_stage.xlsx"
    excel_path = args.excel_path
    # json_normal_path = "best_images_per_case_top_2_hubeirenming.json"
    # json_normal_path = args.merge["json_normal_path"]
    json_normal_path = args.normal_filter["out_json"]
    # json_early_path  = "best_images_per_case_top_2_hubeirenming_zhao.json"
    # json_early_path  = args.merge["json_early_path"]
    json_early_path  = args.early_filter["out_json"]
    # output_path      = "filtered_best_images_hubeirenming.json"
    output_path      = args.merge["output_json"]

    # === Parameters ===
    # max_per_label = 2
    max_per_label = args.merge["common"]["max_per_label"]
    # max_total_per_case = 30
    max_total_per_case = args.merge["common"]["max_total_per_case"]

    # === Parse case_id -> (patient_id, exam_id, field_id) ===



    # === Read JSON ===
    with open(json_normal_path, "r", encoding="utf-8") as f:
        data_normal = json.load(f)
    with open(json_early_path, "r", encoding="utf-8") as f:
        data_early = json.load(f)

    # === Read Excel: three-key lookup + early-pregnancy flag ===
    df = pd.read_excel(excel_path)

    col_patient = "患者ID"
    col_exam    = "检查ID"
    col_field   = "字段ID"
    col_early   = "是否早孕"   # 0/1; change this if your column name differs

    for c in [col_patient, col_exam, col_field]:
        if c not in df.columns:
            raise ValueError(f"Excel is missing required column: {c}")

    has_early_col = col_early in df.columns

    # (patient_id, exam_id, field_id) -> is_early(bool)
    early_map = {}
    for _, row in df.iterrows():
        pid = str(row[col_patient]).strip() if pd.notna(row[col_patient]) else ""
        eid = str(row[col_exam]).strip()    if pd.notna(row[col_exam]) else ""
        fid = str(row[col_field]).strip()   if pd.notna(row[col_field]) else ""
        if not pid or not eid or not fid:
            continue
        key = (pid, eid, fid)
        if has_early_col and pd.notna(row[col_early]):
            flag = bool(int(row[col_early]))
        else:
            flag = False
        # Any True value makes the final flag True
        early_map[key] = early_map.get(key, False) or flag

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
        pid, eid, fid = parse_case_id(case_id)
        is_early = False
        if all([pid, eid, fid]):
            is_early = early_map.get((pid, eid, fid), False)

        # Debug note: forcing is_early=True was used during testing

        case_images = defaultdict(list)
        if is_early:
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

    # Early vs. mid/late pregnancy stats (based on the three-key mapping)
    early_cases = 0
    normal_cases = 0
    for case_id in result.keys():
        pid, eid, fid = parse_case_id(case_id)
        if all([pid, eid, fid]) and early_map.get((pid, eid, fid), False):
            early_cases += 1
        else:
            normal_cases += 1

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



    # === Save matched Excel rows ===
    matched_keys = set()
    for case_id in result.keys():
        pid, eid, fid = parse_case_id(case_id)
        if all([pid, eid, fid]):
            matched_keys.add((pid, eid, fid))

    # Key columns used for filtering
    col_patient = "患者ID"
    col_exam    = "检查ID"
    col_field   = "字段ID"

    # Filter rows
    df_matched = df[df.apply(
        lambda r: (str(r[col_patient]).strip(), str(r[col_exam]).strip(), str(r[col_field]).strip()) in matched_keys,
        axis=1
    )]

    # Save output
    xlsx_out_path = output_path.replace(".json", "_matched.xlsx")
    df_matched.to_excel(xlsx_out_path, index=False)

    print(f"📑 Saved matched Excel rows to: {xlsx_out_path}")
    print(f"Total matched rows: {len(df_matched)}")
