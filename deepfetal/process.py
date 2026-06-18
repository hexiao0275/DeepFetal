import argparse
import json
import os
import shutil
import sys
from importlib import import_module

from .check_paths import check_and_prepare_paths

DEFAULT_WORKSPACE_DIR = "./workspace"


def load_yaml_module():
    try:
        return import_module("yaml")
    except ModuleNotFoundError as exc:
        python_exe = sys.executable or "python"
        raise ModuleNotFoundError(
            "Missing required dependency 'PyYAML'.\n"
            f"Current Python executable: {python_exe}\n"
            "This usually means the package was installed into a different conda environment.\n"
            f"Install it into the current interpreter with:\n"
            f"  {python_exe} -m pip install PyYAML"
        ) from exc


def set_up_script(args, config_key):
    config_dict = getattr(args, config_key)
    script_path = config_dict["script"]
    script = getattr(import_module(script_path), "main")
    script(args)
    print(f"{config_key} finished")


def early_process(args):
    set_up_script(args, "early_cls")
    set_up_script(args, "normal_cls")
    set_up_script(args, "early_filter")
    set_up_script(args, "normal_filter")
    set_up_script(args, "merge")
    set_up_script(args, "report")


def normal_process(args):
    os.makedirs(os.path.dirname(args.early_filter["out_json"]), exist_ok=True)
    with open(args.early_filter["out_json"], "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)

    set_up_script(args, "normal_cls")
    set_up_script(args, "normal_filter")
    set_up_script(args, "merge")
    set_up_script(args, "report")


def handle_singleton(case_dir, is_early, tem_early, tem_normal):
    case_name = os.path.basename(case_dir.rstrip("/"))

    dst_root = tem_early if is_early else tem_normal
    dst_dir = os.path.join(dst_root, case_name)
    shutil.rmtree(dst_dir, ignore_errors=True)
    shutil.copytree(case_dir, dst_dir, dirs_exist_ok=True)

    src_parent = os.path.dirname(case_dir.rstrip("/"))
    dst_parent = os.path.dirname(dst_dir.rstrip("/"))
    return src_parent, dst_parent, int(bool(is_early))


def main(args):
    check_and_prepare_paths(args)
    tem_root = os.path.join(args.workspace_dir, "temp")
    tem_early = os.path.join(tem_root, "early_image_root")
    tem_normal = os.path.join(tem_root, "normal_image_root")
    shutil.rmtree(tem_early, ignore_errors=True)
    os.makedirs(tem_early, exist_ok=True)
    shutil.rmtree(tem_normal, ignore_errors=True)
    os.makedirs(tem_normal, exist_ok=True)

    if args.batch_prediction:
        raise ValueError("Only single-case inference is supported.")

    args.original_case_name = os.path.basename(args.image_root.rstrip("/"))
    src_root, dst_root, _ = handle_singleton(args.image_root, args.is_early, tem_early, tem_normal)
    args.root_mapping = [src_root, dst_root]

    if len(os.listdir(tem_early)) != 0:
        print("Running first-trimester pipeline...")
        args.image_root = tem_early
        early_process(args)

    if len(os.listdir(tem_normal)) != 0:
        print("Running second- and third-trimester pipeline...")
        args.image_root = tem_normal
        normal_process(args)

    print("Converting output format...")
    set_up_script(args, "convert")
    set_up_script(args, "image_prompt")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None, help="Override run.name")
    parser.add_argument("--workspace_dir", default=DEFAULT_WORKSPACE_DIR, help="Workspace directory for intermediate outputs")
    parser.add_argument("--image_root", default="./data/samples/PatientID704_ExamID10143_trimester2", help="Case folder path")
    parser.add_argument("--config_path", default="./config/config.yaml", help="")
    parser.add_argument("--batch_prediction", action="store_true", help="Batch mode is not supported.")
    parser.add_argument("--is_early", type=int, choices=[0, 1], default=0, help="Trimester pipeline: 1 for first-trimester, 0 for second- and third-trimester")
    parser.add_argument(
        "--convert_report_path",
        default=f"{DEFAULT_WORKSPACE_DIR}/infer/ultrasound_prompt_result.jsonl",
        help="Converted report output path",
    )
    parser.add_argument("--visit_type_is_screening", type=int, choices=[0, 1], default=1, help="Visit type: 1 for screening, 0 for clinical")
    parser.add_argument(
        "--center",
        type=str,
        choices=["CENTER_1_RED_HOUSE", "CENTER_2_RENMING", "CENTER_5_XIEHE"],
        default="CENTER_1_RED_HOUSE",
        help="Center name, used only when infer_task_is_report is enabled",
    )
    parser.add_argument(
        "--infer_task_is_report",
        type=int,
        choices=[0, 1],
        default=1,
        help="Inference task: 1 for report generation, 0 for diagnosis generation",
    )
    parser.add_argument("--is_zh", type=int, choices=[0, 1], default=1, help="Language: 1 for Chinese, 0 for English")
    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    yaml = load_yaml_module()
    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in cfg.items():
        setattr(args, k, v)
    return args


def cli():
    args = parse_args()
    main(args)


if __name__ == "__main__":
    cli()
