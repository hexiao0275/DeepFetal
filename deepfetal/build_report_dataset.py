import json
import os
import random
import pandas as pd
import re
from tqdm import tqdm

from .plane_texts import (
    DIAGNOSIS_PROMPTS_EN,
    DIAGNOSIS_PROMPTS_ZH,
    LABEL_MAPPING_EN,
    LABEL_MAPPING_ZH,
    REPORT_PROMPTS_EN,
    REPORT_PROMPTS_ZH,
)

def main(args):
    # -----------------------------
    # Read Excel to obtain report text
    # -----------------------------
    # excel_path = "孕周提取完整结果_gpt_results_only_eval_with_stage.xlsx"
    excel_path = args.excel_path
    df_excel = pd.read_excel(excel_path)

    # Ensure patient and exam IDs are strings
    df_excel["患者ID"] = df_excel["患者ID"].astype(str)
    df_excel["检查ID"] = df_excel["检查ID"].astype(str)

    # Build {(patient_id, exam_id): report}
    excel_report_dict = {
        (row["患者ID"], row["检查ID"]): row["report"]
        for _, row in df_excel.iterrows()
    }


    label_mapping = LABEL_MAPPING_ZH if args.is_zh else LABEL_MAPPING_EN

    if args.infer_task_is_report:
        prompt_list = REPORT_PROMPTS_ZH if args.is_zh else REPORT_PROMPTS_EN
    else:
        prompt_list = DIAGNOSIS_PROMPTS_ZH if args.is_zh else DIAGNOSIS_PROMPTS_EN

    # -----------------------------
    # Read filtered_best_images_hubeirenming.json
    # -----------------------------
    # with open('filtered_best_images_hubeirenming.json', 'r', encoding='utf-8') as f:
    with open(args.merge["output_json"], 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    output = []

    missing_report_cases = []

    # -----------------------------
    # Process each case
    # -----------------------------
    for case_id, labels in tqdm(json_data.items(), desc="Processing cases"):

        # -----------------------------
        # Extract patient ID and exam ID from case_id
        # Example format: PatientID223731_ExamID674172_trimester2
        # -----------------------------
        m = re.search(r"PatientID(\d+)_ExamID(\d+)", case_id)
        if not m:
            print(f"Unable to parse ID: {case_id}")
            continue

        patient_id, exam_id = m.group(1), m.group(2)

        # -----------------------------
        # Get report text from Excel
        # -----------------------------
        report = excel_report_dict.get((patient_id, exam_id), "Report not found in Excel")

        # -----------------------------
        # Build image description
        # -----------------------------
        image_paths = [v[0]['image_path'] for v in labels.values() if v]
        labels_list = list(labels.keys())

        image_description = ""
        for i, (label, img_path) in enumerate(zip(labels_list, image_paths)):
            plane_label = label_mapping.get(label, label)
            image_description += f"{i+1}. {plane_label}\n<image>\n"

        entry = {
            "id": case_id,
            "image": image_paths,
            "conversations": [
                {
                    "from": "human",
                    "content": image_description + random.choice(prompt_list)
                },
                {
                    "from": "gpt",
                    "content": report
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
