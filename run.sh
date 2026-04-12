#!/usr/bin/env bash

set -euo pipefail

load_env_file() {
  local file_path="$1"
  local line key value

  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    if [ -z "$line" ] || [[ "$line" == \#* ]]; then
      continue
    fi

    key="${line%%=*}"
    value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    if [ "${value#\"}" != "$value" ] && [ "${value%\"}" != "$value" ]; then
      value="${value#\"}"
      value="${value%\"}"
    elif [ "${value#\'}" != "$value" ] && [ "${value%\'}" != "$value" ]; then
      value="${value#\'}"
      value="${value%\'}"
    fi

    if [ -z "$key" ]; then
      continue
    fi

    if [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$file_path"
}

env_file="${ENV_FILE:-}"
if [ -z "$env_file" ]; then
  if [ -f ./.env.local ]; then
    env_file=./.env.local
  elif [ -f ./.env ]; then
    env_file=./.env
  fi
fi

if [ -n "$env_file" ]; then
  load_env_file "$env_file"
fi

mode="${MODE:-all}"
infer_backend="${INFER_BACKEND:-api}"

workspace_dir="${WORKSPACE_DIR:-./workspace}"
image_root="${IMAGE_ROOT:-./data/samples/PatientID704_ExamID10143_trimester2}"
config_path="${CONFIG_PATH:-./config/config.yaml}"
excel_path="${EXCEL_PATH:-./data/metadata/pregnancy_stage.xlsx}"
input_jsonl="${INPUT_JSONL:-$workspace_dir/infer/ultrasound_prompt_result.jsonl}"

visit_type_is_screening="${VISIT_TYPE_IS_SCREENING:-1}"
infer_task_is_report="${INFER_TASK_IS_REPORT:-1}"
center="${CENTER:-CENTER_1_RED_HOUSE}"
is_zh="${IS_ZH:-1}"
use_openai_constraint="${USE_OPENAI_CONSTRAINT:-${USE_IMAGE_PROMPT:-0}}"

resolve_python_bin() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/$PYTHON_BIN" ]; then
      printf '%s\n' "$CONDA_PREFIX/bin/$PYTHON_BIN"
      return 0
    fi

    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/$PYTHON_BIN" ]; then
      printf '%s\n' "$VIRTUAL_ENV/bin/$PYTHON_BIN"
      return 0
    fi

    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      command -v "$PYTHON_BIN"
      return 0
    fi

    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi

  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    printf '%s\n' "$CONDA_PREFIX/bin/python"
    return 0
  fi

  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    printf '%s\n' "$VIRTUAL_ENV/bin/python"
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  return 1
}

python_bin="$(resolve_python_bin)" || {
  echo "No usable Python interpreter was found."
  echo "Set PYTHON_BIN explicitly or activate your conda/virtual environment first."
  exit 1
}

echo "Using Python interpreter: $python_bin"

ensure_python_module() {
  local module_name="$1"
  local package_name="$2"

  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python interpreter not found: $python_bin"
    exit 1
  fi

  if ! "$python_bin" -c "import $module_name" >/dev/null 2>&1; then
    echo "The Python interpreter '$python_bin' does not have the '$package_name' package installed."
    if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
      echo "Active conda environment: $CONDA_DEFAULT_ENV"
    fi
    echo "This often means the package was installed into a different conda or virtual environment."
    echo "Install it with: $python_bin -m pip install $package_name"
    exit 1
  fi
}

run_process() {
  ensure_python_module yaml PyYAML
  "$python_bin" -m deepfetal.process \
    --workspace_dir "$workspace_dir" \
    --image_root "$image_root" \
    --config_path "$config_path" \
    --excel_path "$excel_path" \
    --convert_report_path "$input_jsonl" \
    --visit_type_is_screening "$visit_type_is_screening" \
      --center "$center" \
      --infer_task_is_report "$infer_task_is_report" \
      --is_zh "$is_zh" \
      --use_openai_constraint "$use_openai_constraint" \
      --use_image_prompt "$use_openai_constraint"
}

run_infer() {
  mkdir -p "$workspace_dir/infer"

  if [ "$infer_backend" = "swift" ]; then
    if [ "$use_openai_constraint" -eq 1 ]; then
      model_dir="${MODEL_DIR:-./checkpoints/checkpoint-600-merged}"
    else
      model_dir="${MODEL_DIR:-./checkpoints/checkpoint-2450-merged}"
    fi

    output_path="${OUTPUT_PATH:-$workspace_dir/infer/final_result.jsonl}"

    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export SWIFT_PATCH_CONV3D=1

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" swift infer \
      --model "$model_dir" \
      --model_type qwen3_vl \
      --infer_backend transformers \
      --temperature 0 \
      --max_new_tokens 2048 \
      --result_path "$output_path" \
      --val_dataset "$input_jsonl"
    return
  fi

  if [ "$infer_backend" = "api" ]; then
    output_path="${OUTPUT_PATH:-$workspace_dir/infer/final_result_api.jsonl}"
    if [ -z "${OPENAI_API_KEY:-}" ]; then
      echo "Missing OPENAI_API_KEY. Set it in .env.local / .env, or export it in the shell."
      exit 1
    fi
    ensure_python_module openai openai
    cmd=(
      "$python_bin" -m deepfetal.api_infer
      --input "$input_jsonl"
      --output "$output_path"
      --model "${OPENAI_MODEL:-gpt-4.1-mini}"
    )
    if [ -n "${OPENAI_BASE_URL:-}" ]; then
      cmd+=(--base-url "$OPENAI_BASE_URL")
    fi
    if [ -n "${OPENAI_API_KEY:-}" ]; then
      cmd+=(--api-key "$OPENAI_API_KEY")
    fi
    "${cmd[@]}"
    return
  fi

  echo "Unsupported INFER_BACKEND: $infer_backend"
  echo "Available values: swift | api"
  exit 1
}

case "$mode" in
  process)
    run_process
    ;;
  infer)
    run_infer
    ;;
  all)
    run_process
    run_infer
    ;;
  *)
    echo "Unsupported MODE: $mode"
    echo "Available values: process | infer | all"
    exit 1
    ;;
esac
