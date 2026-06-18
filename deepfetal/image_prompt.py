# -*- coding: utf-8 -*-
"""
Enhanced version with resume support.
After generating constrained image-evidence text, append it to
messages[0]["content"] as downstream guidance and write the result to OUTPUT_ROOT.
"""

import os
import json
import time
import base64
import mimetypes
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from openai import OpenAI
from tqdm import tqdm

# ========= Core configuration =========
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
TEST_COUNT = None

# INPUT_PATHS = [
#     Path("ultrasound_reports_convert.jsonl"),
# ]

# COMMON_BASE = Path("no_china")
# OUTPUT_ROOT = Path("./output")

# ========= Performance configuration =========
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 3.0
REQUEST_TIMEOUT = 240.0
MAX_WORKERS = 10
IN_FLIGHT_LIMIT = 50

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY. Set it in .env.local or export it before running the semantic agent module.")

client_kwargs = {"api_key": api_key, "timeout": REQUEST_TIMEOUT}
if os.getenv("OPENAI_BASE_URL"):
    client_kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
client = OpenAI(**client_kwargs)

IMAGE_ANALYSIS_SYSTEM_PROMPT = (
    "You are a professional fetal ultrasound doctor. "
    "Describe fetal ultrasound images clearly and stay grounded in the visible evidence."
)

SEMANTIC_AGENT_REQUEST = (
    "Review all provided fetal ultrasound images and describe the visible imaging findings in English. "
    "Include visible measurements, numbers, and units when they can be read. "
    "Preserve uncertainty when the still images are incomplete or equivocal. "
    "Do not invent findings that are not supported by the images."
)

IMAGING_DESCRIPTION_PREFIX = (
    "\n\nThe following are the relevant imaging descriptions for this case, provided for reference: "
)


# ========= Helper functions =========

def to_data_url(image_path: str) -> Optional[str]:
    p = Path(image_path).expanduser()
    if not p.exists() or not p.is_file():
        return None

    mime, _ = mimetypes.guess_type(str(p))
    if mime is None:
        ext = p.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif ext == ".png":
            mime = "image/png"
        else:
            mime = "application/octet-stream"

    try:
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def prepare_pure_user_payload(messages: List[Dict[str, Any]], images: List[str]) -> List[Dict[str, Any]]:
    """
    Build model input without mutating the original messages.
    """
    system_message = {
        "role": "system",
        "content": IMAGE_ANALYSIS_SYSTEM_PROMPT,
    }

    user_msgs = [deepcopy(m) for m in messages if m.get("role") == "user"]
    if not user_msgs:
        user_msgs = [{"role": "user", "content": ""}]

    target_msg = user_msgs[-1]
    orig_content = target_msg.get("content", "")
    content_blocks = []

    if isinstance(orig_content, str):
        prompt_text = orig_content.rstrip()
        if prompt_text:
            prompt_text += "\n\n"
        prompt_text += SEMANTIC_AGENT_REQUEST
        content_blocks.append({"type": "text", "text": prompt_text})
    elif isinstance(orig_content, list):
        content_blocks = deepcopy(orig_content)
        found_text = False
        for block in content_blocks:
            if block.get("type") == "text":
                base_text = block.get("text", "").rstrip()
                if base_text:
                    base_text += "\n\n"
                block["text"] = base_text + SEMANTIC_AGENT_REQUEST
                found_text = True
                break
        if not found_text:
            content_blocks.append({"type": "text", "text": SEMANTIC_AGENT_REQUEST})
    else:
        content_blocks.append({"type": "text", "text": SEMANTIC_AGENT_REQUEST})

    for img_path in images:
        data_url = to_data_url(img_path)
        if data_url:
            content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            content_blocks.append({"type": "text", "text": f"[Error: Cannot load image {img_path}]"})

    target_msg["content"] = content_blocks
    return [system_message] + user_msgs


def do_completion(payload_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    backoff = INITIAL_BACKOFF_SEC
    last_err = None
    for i in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(model=MODEL, messages=payload_messages)
            dur = time.time() - t0
            return {
                "completion": resp.choices[0].message.content or "",
                "usage": resp.usage.model_dump() if resp.usage else None,
                "latency_sec": round(dur, 2)
            }
        except Exception as e:
            last_err = str(e)
            if "rate_limit" in last_err.lower():
                time.sleep(backoff * 2)
            else:
                time.sleep(backoff)
            backoff *= 1.5
    return {"completion": "", "usage": None, "error": f"Max retries reached: {last_err}"}


def strip_image_blocks(messages: List[Dict[str, Any]]) -> None:
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            m["content"] = [b for b in content if b.get("type") not in ("image_url", "input_image")]


def append_imaging_description_to_first_message(messages: List[Dict[str, Any]], completion_text: str) -> None:
    """
    Append semantic-agent imaging descriptions to the first message.
    """
    if not messages or not completion_text.strip():
        return

    first_msg = messages[0]
    original_content = first_msg.get("content", "")
    addition = IMAGING_DESCRIPTION_PREFIX + completion_text.strip()

    if isinstance(original_content, str):
        first_msg["content"] = original_content + addition
    elif isinstance(original_content, list):
        first_msg["content"].append({"type": "text", "text": addition})
    else:
        first_msg["content"] = addition


def process_one_record(idx: int, obj: Dict[str, Any], sem: threading.Semaphore) -> Tuple[int, Dict[str, Any]]:
    with sem:
        try:
            raw_images = obj.get("images", [])
            payload = prepare_pure_user_payload(obj.get("messages", []), raw_images)
            result = do_completion(payload)

            out = deepcopy(obj)
            strip_image_blocks(out.get("messages", []))

            completion_text = result.get("completion", "") or ""
            append_imaging_description_to_first_message(out.get("messages", []), completion_text)
            out["image_constraint_text"] = completion_text

            if result.get("error"):
                out["error"] = result["error"]
            else:
                out["image_constraint_usage"] = result.get("usage")
                out["image_constraint_latency_sec"] = result.get("latency_sec")

            return idx, out

        except Exception as e:
            return idx, {"error": f"Thread Error: {str(e)}"}


def count_lines(file_path: Path) -> int:
    """Count existing lines in a file for resume support."""
    if not file_path.exists():
        return 0
    with file_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def process_single_file(input_path: Path, sem: threading.Semaphore):
    output_path = OUTPUT_PATHS
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                records.append((i, line.strip()))

    results_buffer = {}
    next_to_write = records[0][0]

    with output_path.open("a", encoding="utf-8") as fout:
        with tqdm(total=len(records), desc=f"📄 {input_path.name[:20]}", leave=False) as pbar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_idx = {
                    executor.submit(process_one_record, idx, json.loads(line), sem): idx
                    for idx, line in records
                }

                for fut in as_completed(future_to_idx):
                    idx, out_obj = fut.result()
                    results_buffer[idx] = out_obj
                    pbar.update(1)

                    while next_to_write in results_buffer:
                        res = results_buffer.pop(next_to_write)
                        fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                        fout.flush()
                        next_to_write += 1


def main(args):

    global INPUT_PATHS, OUTPUT_PATHS
    INPUT_PATHS = [Path(args.convert["out_json"])]

    OUTPUT_PATHS = Path(args.convert_report_path)

    # INPUT_PATHS = [Path("./output_final/5_1_ultrasound_reports_convert.json")]
    # OUTPUT_PATHS = Path("./ultrasound_prompt_result.jsonl")
    valid_files = [p for p in INPUT_PATHS if p.exists()]
    if not valid_files:
        print("No valid input files were found. Check INPUT_PATHS.")
        return

    print(f"Task started | workers: {MAX_WORKERS} | mode: {'test' if TEST_COUNT else 'full'}")

    sem = threading.Semaphore(IN_FLIGHT_LIMIT)

    for jsonl_file in tqdm(valid_files, desc="Overall progress", unit="file"):
        try:
            process_single_file(jsonl_file, sem)
        except Exception as e:
            print(f"Fatal error while processing file {jsonl_file}: {e}")

    print(f"\nFinished. Results saved to: {OUTPUT_PATHS}")


if __name__ == "__main__":
    main()
