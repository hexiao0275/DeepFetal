#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API inference entry point for stage 2.

Supported modes:
1. Direct OpenAI API
2. Any OpenAI-compatible API
3. A local service exposing an OpenAI-compatible protocol

Default input:
  ./workspace/infer/ultrasound_prompt_result.jsonl

Default output:
  ./workspace/infer/final_result_api.jsonl
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


DEFAULT_INPUT = "./workspace/infer/ultrasound_prompt_result.jsonl"
DEFAULT_OUTPUT = "./workspace/infer/final_result_api.jsonl"
DEFAULT_SYSTEM_PROMPT = "You are a professional fetal ultrasound doctor."
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TIMEOUT = 240.0
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 3.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stage-2 final inference through an OpenAI-compatible API")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input jsonl path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output jsonl path")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), help="Model name")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="API key")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Maximum output tokens")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Single-request timeout in seconds")
    parser.add_argument("--max-workers", type=int, default=4, help="Worker thread count")
    parser.add_argument("--in-flight-limit", type=int, default=8, help="Maximum in-flight requests")
    parser.add_argument("--test-count", type=int, default=None, help="Only process the first N records")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt, defaulting to a fetal ultrasound expert role",
    )
    return parser


def to_data_url(image_path: str) -> Optional[str]:
    path = Path(image_path).expanduser()
    if not path.exists() or not path.is_file():
        return None

    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".png":
            mime = "image/png"
        else:
            mime = "application/octet-stream"

    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def normalize_user_message(messages: List[Dict[str, Any]]) -> str:
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return ""

    content = user_messages[-1].get("content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if block.get("type") == "text":
                text_blocks.append(block.get("text", ""))
        return "\n".join(text_blocks).strip()

    return str(content)


def build_api_messages(
    messages: List[Dict[str, Any]],
    images: List[str],
    system_prompt: str,
) -> List[Dict[str, Any]]:
    user_text = normalize_user_message(messages)
    content_blocks: List[Dict[str, Any]] = []

    if user_text:
        content_blocks.append({"type": "text", "text": user_text})

    for image_path in images:
        data_url = to_data_url(image_path)
        if data_url:
            content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            content_blocks.append({"type": "text", "text": f"[Missing image: {image_path}]"})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_blocks or [{"type": "text", "text": user_text}]},
    ]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def chat_completion(
    client: OpenAI,
    model: str,
    payload_messages: List[Dict[str, Any]],
    temperature: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    backoff = INITIAL_BACKOFF_SEC
    last_error = None

    for _ in range(MAX_RETRIES):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=payload_messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            latency = round(time.time() - start, 2)
            return {
                "completion": response.choices[0].message.content or "",
                "usage": response.usage.model_dump() if response.usage else None,
                "latency_sec": latency,
            }
        except Exception as exc:
            last_error = str(exc)
            time.sleep(backoff)
            backoff *= 1.5

    return {
        "completion": "",
        "usage": None,
        "error": f"Max retries reached: {last_error}",
    }


def process_one_record(
    idx: int,
    obj: Dict[str, Any],
    sem: threading.Semaphore,
    client: OpenAI,
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    with sem:
        try:
            payload_messages = build_api_messages(
                messages=obj.get("messages", []),
                images=obj.get("images", []),
                system_prompt=args.system_prompt,
            )
            result = chat_completion(
                client=client,
                model=args.model,
                payload_messages=payload_messages,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

            out = deepcopy(obj)
            out["backend"] = "api"
            out["model"] = args.model
            out["completion"] = result.get("completion", "")
            out["usage"] = result.get("usage")
            out["latency_sec"] = result.get("latency_sec")
            if result.get("error"):
                out["error"] = result["error"]
            return idx, out
        except Exception as exc:
            return idx, {"error": f"Thread Error: {exc}"}


def process_file(args: argparse.Namespace, client: OpenAI) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    processed_count = count_lines(output_path)
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if idx <= processed_count:
                continue
            if line.strip():
                records.append((idx, line.strip()))
            if args.test_count and len(records) >= args.test_count:
                break

    if not records:
        print(f"{input_path.name} is already fully processed, skipping.")
        return

    sem = threading.Semaphore(args.in_flight_limit)
    results_buffer: Dict[int, Dict[str, Any]] = {}
    next_to_write = records[0][0]

    print(f"Starting API inference: {input_path} -> {output_path}")
    print(f"model={args.model}, max_workers={args.max_workers}, total={len(records)}")

    with output_path.open("a", encoding="utf-8") as fout:
        with tqdm(total=len(records), desc=input_path.name, leave=False) as pbar:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_map = {
                    executor.submit(process_one_record, idx, json.loads(line), sem, client, args): idx
                    for idx, line in records
                }
                for future in as_completed(future_map):
                    idx, out_obj = future.result()
                    results_buffer[idx] = out_obj
                    pbar.update(1)
                    while next_to_write in results_buffer:
                        fout.write(json.dumps(results_buffer.pop(next_to_write), ensure_ascii=False) + "\n")
                        fout.flush()
                        next_to_write += 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Provide it with --api-key or OPENAI_API_KEY.")

    client_kwargs = {"api_key": args.api_key, "timeout": args.timeout}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    process_file(args, client)
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
