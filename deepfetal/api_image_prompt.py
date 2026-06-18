# -*- coding: utf-8 -*-
"""
Call an OpenAI-compatible API to generate constrained image-evidence text for multi-image cases.

Default input:
  ./workspace/preprocess/5_1_ultrasound_reports_convert.jsonl

Default output:
  ./workspace/infer/api_image_prompt_output.jsonl
"""

import argparse
import base64
import json
import mimetypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 3.0
REQUEST_TIMEOUT = 240.0

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


def build_parser():
    parser = argparse.ArgumentParser(description="Generate supplemental case image-analysis text through an OpenAI-compatible API")
    parser.add_argument(
        "--input",
        default="./workspace/preprocess/5_1_ultrasound_reports_convert.jsonl",
        help="Input jsonl path",
    )
    parser.add_argument(
        "--output",
        default="./workspace/infer/api_image_prompt_output.jsonl",
        help="Output jsonl path",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), help="Model name")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), help="Compatible API base URL")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="API key")
    parser.add_argument("--max-workers", type=int, default=8, help="Worker count")
    parser.add_argument("--in-flight-limit", type=int, default=16, help="Maximum concurrent in-flight requests")
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT, help="Single-request timeout in seconds")
    parser.add_argument("--test-count", type=int, default=None, help="Only process the first N records")
    return parser


def to_data_url(image_path: str) -> Optional[str]:
    path = Path(image_path).expanduser()
    if not path.exists() or not path.is_file():
        return None

    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif ext == ".png":
            mime = "image/png"
        else:
            mime = "application/octet-stream"

    try:
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def prepare_payload(messages: List[Dict[str, Any]], images: List[str]) -> List[Dict[str, Any]]:
    system_message = {"role": "system", "content": IMAGE_ANALYSIS_SYSTEM_PROMPT}

    user_messages = [deepcopy(m) for m in messages if m.get("role") == "user"]
    if not user_messages:
        user_messages = [{"role": "user", "content": ""}]

    target = user_messages[-1]
    content = target.get("content", "")
    blocks: List[Dict[str, Any]] = []

    if isinstance(content, str):
        prompt_text = content.rstrip()
        if prompt_text:
            prompt_text += "\n\n"
        prompt_text += SEMANTIC_AGENT_REQUEST
        blocks.append({"type": "text", "text": prompt_text})
    elif isinstance(content, list):
        blocks = deepcopy(content)
        found_text = False
        for block in blocks:
            if block.get("type") == "text":
                base_text = block.get("text", "").rstrip()
                if base_text:
                    base_text += "\n\n"
                block["text"] = base_text + SEMANTIC_AGENT_REQUEST
                found_text = True
                break
        if not found_text:
            blocks.append({"type": "text", "text": SEMANTIC_AGENT_REQUEST})
    else:
        blocks.append({"type": "text", "text": SEMANTIC_AGENT_REQUEST})

    for image_path in images:
        data_url = to_data_url(image_path)
        if data_url:
            blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            blocks.append({"type": "text", "text": f"[Error: Cannot load image {image_path}]"})

    target["content"] = blocks
    return [system_message] + user_messages


def do_completion(client: OpenAI, model: str, payload_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    backoff = INITIAL_BACKOFF_SEC
    last_error = None
    for _ in range(MAX_RETRIES):
        try:
            start = time.time()
            response = client.chat.completions.create(model=model, messages=payload_messages)
            latency = time.time() - start
            return {
                "completion": response.choices[0].message.content or "",
                "usage": response.usage.model_dump() if response.usage else None,
                "latency_sec": round(latency, 2),
            }
        except Exception as exc:
            last_error = str(exc)
            if "rate_limit" in last_error.lower():
                time.sleep(backoff * 2)
            else:
                time.sleep(backoff)
            backoff *= 1.5
    return {"completion": "", "usage": None, "error": f"Max retries reached: {last_error}"}


def strip_image_blocks(messages: List[Dict[str, Any]]) -> None:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [b for b in content if b.get("type") not in ("image_url", "input_image")]


def append_imaging_description_to_first_message(messages: List[Dict[str, Any]], completion_text: str) -> None:
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


def process_one_record(
    idx: int,
    obj: Dict[str, Any],
    sem: threading.Semaphore,
    client: OpenAI,
    model: str,
) -> Tuple[int, Dict[str, Any]]:
    with sem:
        try:
            images = obj.get("images", [])
            payload = prepare_payload(obj.get("messages", []), images)
            result = do_completion(client, model, payload)
            out = deepcopy(obj)
            strip_image_blocks(out.get("messages", []))
            completion_text = result.get("completion", "") or ""
            append_imaging_description_to_first_message(out.get("messages", []), completion_text)
            out.update(
                {
                    "completion": completion_text,
                    "image_constraint_text": completion_text,
                    "usage": result.get("usage"),
                    "latency_sec": result.get("latency_sec"),
                }
            )
            if result.get("error"):
                out["error"] = result["error"]
            return idx, out
        except Exception as exc:
            return idx, {"error": f"Thread Error: {exc}"}


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def process_file(args, client: OpenAI):
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    processed_count = count_lines(output_path)
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i <= processed_count:
                continue
            if line.strip():
                records.append((i, line.strip()))
            if args.test_count and len(records) >= args.test_count:
                break

    if not records:
        print(f"{input_path.name} is already fully processed, skipping.")
        return

    print(f"Continuing {input_path.name}: remaining {len(records)} records (skipped {processed_count})")

    sem = threading.Semaphore(args.in_flight_limit)
    results_buffer: Dict[int, Dict[str, Any]] = {}
    next_to_write = records[0][0]

    with output_path.open("a", encoding="utf-8") as fout:
        with tqdm(total=len(records), desc=f"📄 {input_path.name[:20]}", leave=False) as pbar:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_idx = {
                    executor.submit(process_one_record, idx, json.loads(line), sem, client, args.model): idx
                    for idx, line in records
                }
                for future in as_completed(future_to_idx):
                    idx, out_obj = future.result()
                    results_buffer[idx] = out_obj
                    pbar.update(1)
                    while next_to_write in results_buffer:
                        res = results_buffer.pop(next_to_write)
                        fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                        fout.flush()
                        next_to_write += 1


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Provide it with --api-key or OPENAI_API_KEY.")

    client_kwargs = {"api_key": args.api_key, "timeout": args.timeout}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    print(f"Task started | model={args.model} | max_workers={args.max_workers}")
    process_file(args, client)
    print(f"Finished. Output file: {args.output}")


if __name__ == "__main__":
    main()
