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


def load_patient_info(patient_info_path):
    """Load patient information Excel and return a dict keyed by patient name."""
    if not patient_info_path or not os.path.exists(patient_info_path):
        return {}
    df = pd.read_excel(patient_info_path)
    name_col = df.columns[0]
    info_dict = {}
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        info = {}
        if "checklist" in df.columns:
            val = row.get("checklist")
            if pd.notna(val):
                info["checklist"] = str(val).strip()
        if "type" in df.columns:
            val = row.get("type")
            if pd.notna(val):
                info["type"] = str(val).strip()
        if "agent API结果" in df.columns:
            val = row.get("agent API结果")
            if pd.notna(val):
                info["agent_api_result"] = str(val).strip()
        info_dict[name] = info
    return info_dict


def main(args):
    # -----------------------------
    # Read Excel to obtain report text
    # -----------------------------
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
    # Load patient info (checklist, type, agent API result)
    # -----------------------------
    patient_info_path = getattr(args, "patient_info_path", None)
    patient_info = load_patient_info(patient_info_path)
    original_case_name = getattr(args, "original_case_name", "")

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
        # Extract patient ID and exam ID from case_id
        # -----------------------------
        m = re.search(r"PatientID(\d+)_ExamID(\d+)", case_id)
        if m:
            patient_id, exam_id = m.group(1), m.group(2)
        else:
            patient_id, exam_id = case_id, case_id

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

        # Build prompt with optional checklist and API result
        prompt_text = random.choice(prompt_list)

        info = patient_info.get(original_case_name, {})
        checklist = info.get("checklist", "")
        agent_api_result = info.get("agent_api_result", "")

        if checklist:
            prompt_text += f"\nSome examination items include: {checklist}"
        if agent_api_result:
            prompt_text += f"\n\nThe following are the relevant imaging descriptions for this case, provided for reference: {agent_api_result}"

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
