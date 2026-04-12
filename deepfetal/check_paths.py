import os
from pathlib import Path


def _require_exists(path_str: str, desc: str):
    """Validate that an input path exists."""
    if not path_str:
        raise ValueError(f"{desc} is empty")
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{desc} does not exist: {path}")
    return path


def _remove_if_exists(path_str: str, desc: str):
    """Delete an output file if it already exists."""
    if not path_str:
        return
    path = Path(path_str)
    if path.exists():
        if path.is_file():
            path.unlink()
            print(f"[Clean] Removed existing output file: {desc} -> {path}")
        elif path.is_dir():
            raise IsADirectoryError(
                f"{desc} was expected to be a file but is a directory: {path}"
            )


def _ensure_parent_dir(path_str: str):
    """Ensure that the parent directory for an output file exists."""
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)


def check_and_prepare_paths(args):
    """
    Validate required inputs, remove stale output files, and create parent directories.
    """
    # ========= Top-level inputs =========
    _require_exists(args.config_path, "config_path")
    _require_exists(args.image_root, "image_root")
    _require_exists(args.excel_path, "excel_path")

    # convert_report_path is usually the final jsonl consumed by swift infer.
    # When regenerating outputs in this pipeline, clean the target path first.
    _ensure_parent_dir(args.convert_report_path)
    _remove_if_exists(args.convert_report_path, "convert_report_path")

    # ========= early_cls =========
    if hasattr(args, "early_cls") and isinstance(args.early_cls, dict):
        sec = args.early_cls
        model = sec.get("model", {})
        _require_exists(model.get("weights_path"), "early_cls.model.weights_path")

        out_json = sec.get("out_json")
        _ensure_parent_dir(out_json)
        _remove_if_exists(out_json, "early_cls.out_json")

    # ========= normal_cls =========
    if hasattr(args, "normal_cls") and isinstance(args.normal_cls, dict):
        sec = args.normal_cls
        model = sec.get("model", {})
        _require_exists(model.get("weights_path"), "normal_cls.model.weights_path")
        _require_exists(model.get("class_indices_path"), "normal_cls.model.class_indices_path")

        out_json = sec.get("out_json")
        _ensure_parent_dir(out_json)
        _remove_if_exists(out_json, "normal_cls.out_json")

    # ========= early_filter =========
    if hasattr(args, "early_filter") and isinstance(args.early_filter, dict):
        sec = args.early_filter
        for key in ["out_json", "out_case_ids", "out_missing"]:
            out_path = sec.get(key)
            _ensure_parent_dir(out_path)
            _remove_if_exists(out_path, f"early_filter.{key}")

    # ========= normal_filter =========
    if hasattr(args, "normal_filter") and isinstance(args.normal_filter, dict):
        sec = args.normal_filter
        for key in ["out_json", "out_case_ids", "out_missing"]:
            out_path = sec.get(key)
            _ensure_parent_dir(out_path)
            _remove_if_exists(out_path, f"normal_filter.{key}")

    # ========= merge =========
    if hasattr(args, "merge") and isinstance(args.merge, dict):
        sec = args.merge
        out_path = sec.get("output_json")
        _ensure_parent_dir(out_path)
        _remove_if_exists(out_path, "merge.output_json")

    # ========= report =========
    if hasattr(args, "report") and isinstance(args.report, dict):
        sec = args.report
        out_path = sec.get("out_json")
        _ensure_parent_dir(out_path)
        _remove_if_exists(out_path, "report.out_json")

    # ========= convert =========
    if hasattr(args, "convert") and isinstance(args.convert, dict):
        sec = args.convert
        _require_exists(sec.get("en_excel"), "convert.en_excel")

        out_path = sec.get("out_json")
        _ensure_parent_dir(out_path)
        _remove_if_exists(out_path, "convert.out_json")

    print("[Check] Input validation complete, old outputs have been cleaned.")
