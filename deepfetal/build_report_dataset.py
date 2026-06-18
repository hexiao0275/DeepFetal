import json
import os
import random
from pathlib import Path
from tqdm import tqdm

from .plane_texts import (
    DIAGNOSIS_PROMPTS_EN,
    DIAGNOSIS_PROMPTS_ZH,
    LABEL_MAPPING_EN,
    LABEL_MAPPING_ZH,
    REPORT_PROMPTS_EN,
    REPORT_PROMPTS_ZH,
)


def find_case_xlsx_files(args):
    original_case_name = getattr(args, "original_case_name", "")
    root_mapping = getattr(args, "root_mapping", None)

    candidates = []
    if root_mapping and original_case_name:
        source_parent = Path(root_mapping[0])
        candidates.append(source_parent / original_case_name)
        candidates.append(source_parent)
    if getattr(args, "image_root", None):
        image_root = Path(args.image_root)
        candidates.append(image_root)
        candidates.append(image_root.parent)

    xlsx_files = []
    seen = set()
    for case_dir in candidates:
        if not case_dir.exists():
            continue
        for xlsx_path in sorted(case_dir.glob("*.xlsx")) + sorted(case_dir.glob("*.xls")):
            if xlsx_path in seen:
                continue
            seen.add(xlsx_path)
            xlsx_files.append(xlsx_path)
    return xlsx_files


def get_preferred_value(row, columns):
    preferred_columns = [
        "checklist",
        "examination_items",
        "exam_items",
        "items",
        "some examination items include",
    ]

    normalized_columns = {str(col).strip().lower(): col for col in columns}
    for key in preferred_columns:
        col = normalized_columns.get(key)
        if col is None:
            continue
        value = row.get(col)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def read_examination_items(args):
    original_case_name = str(getattr(args, "original_case_name", "")).strip()
    if not original_case_name:
        return ""

    try:
        import pandas as pd
    except Exception:
        return ""

    for xlsx_path in find_case_xlsx_files(args):
        try:
            df = pd.read_excel(xlsx_path)
        except Exception:
            continue

        if original_case_name and len(df.columns) > 0:
            for _, row in df.iterrows():
                row_values = [str(v).strip() for v in row.tolist() if str(v).strip() and str(v).strip().lower() != "nan"]
                if original_case_name in row_values:
                    value = get_preferred_value(row, df.columns)
                    if value:
                        return value

    return ""


def main(args):
    label_mapping = LABEL_MAPPING_ZH if args.is_zh else LABEL_MAPPING_EN

    if args.infer_task_is_report:
        prompt_list = REPORT_PROMPTS_ZH if args.is_zh else REPORT_PROMPTS_EN
    else:
        prompt_list = DIAGNOSIS_PROMPTS_ZH if args.is_zh else DIAGNOSIS_PROMPTS_EN

    examination_items = read_examination_items(args)

    # -----------------------------
    # Read filtered_best_images_hubeirenming.json
    # -----------------------------
    with open(args.merge["output_json"], 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    output = []

    # -----------------------------
    # Process each case
    # -----------------------------
    for case_id, labels in tqdm(json_data.items(), desc="Processing cases"):
        # -----------------------------
        # Build image description
        # -----------------------------
        image_paths = [v[0]['image_path'] for v in labels.values() if v]
        labels_list = list(labels.keys())

        image_description = ""
        for i, (label, img_path) in enumerate(zip(labels_list, image_paths)):
            plane_label = label_mapping.get(label, label)
            image_description += f"{i+1}. {plane_label}\n<image>\n"

        # Build prompt
        prompt_text = random.choice(prompt_list)
        if examination_items:
            prompt_text += f"\nSome examination items include: {examination_items}"

        entry = {
            "id": case_id,
            "image": image_paths,
            "conversations": [
                {
                    "from": "human",
                    "content": image_description + prompt_text
                },
                {
                    "from": "gpt",
                    "content": ""
                }
            ]
        }

        output.append(entry)

    # -----------------------------
    # Write JSON
    # -----------------------------

    os.makedirs(os.path.dirname(args.report["out_json"]), exist_ok=True)
    with open(args.report["out_json"], "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Generated {args.report['out_json']}")
