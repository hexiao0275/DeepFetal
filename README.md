# A Deconfounded Multimodal AI System for Fetal Ultrasound Interpretation

 [Website](http://deepfetal.com/)  | [ Model](https://huggingface.co/natureteam/DeepFetal)

> *DeepFetal is an deconfounded multimodal AI system for full-gestation fetal ultrasound interpretation, enabling traceable diagnostic reasoning and robust clinical decision support.*


For more detailed about our pipeline, please refer to our paper.


# Installation

This document provides a concise setup guide for running `deepfetal_code` for preprocessing and inference.

## Step 1: Clone the Repository

```bash
git clone https://github.com/hexiao0275/DeepFetal.git
cd DeepFetal
```

## Step 2: Create the Python Environment

We recommend using Conda with Python 3.10.

```bash
conda create -n deepfetal python=3.10 -y
conda activate deepfetal
```

## Step 3: Install Dependencies

Install the project requirements into the active environment:

```bash
pip install -r requirements.txt
```

If you only plan to use the API backend and do not need local Swift inference, you may still keep the same environment for simplicity.

## Step 4: Prepare Required Files

Make sure the following directories and files are available:

```text
deepfetal_code/
├── run.sh
├── deepfetal/
├── config/
│   └── config.yaml
├── data/
│   ├── metadata/
│   └── samples/
├── checkpoints/
└── workspace/
```

Required assets:

- `data/metadata/pregnancy_stage.xlsx`
- `data/metadata/plane_translation.xlsx`
- your input ultrasound case folder under `data/samples/` or another custom path
- model checkpoints under `checkpoints/` if you use the local `swift` backend

## Step 5: Configure Environment Variables

Create a local environment file named `.env.local` in the project root.

Example configuration:

```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini

MODE=all
INFER_BACKEND=api
USE_OPENAI_CONSTRAINT=0
```

Notes:

- `INFER_BACKEND=api` uses an OpenAI-compatible API for the final reasoning step.
- `INFER_BACKEND=swift` uses a local model from `checkpoints/`.
- `USE_OPENAI_CONSTRAINT=1` enables the optional image-constraint generation step before final inference.
- `USE_OPENAI_CONSTRAINT=0` skips that step.

## Step 6: Run the Pipeline

### Full Pipeline

```bash
conda activate deepfetal
bash run.sh
```

### Preprocess Only

```bash
MODE=process bash run.sh
```

### Inference Only

```bash
MODE=infer bash run.sh
```

### API Inference

```bash
MODE=infer \
INFER_BACKEND=api \
bash run.sh
```

### Local Swift Inference

```bash
MODE=infer \
INFER_BACKEND=swift \
CUDA_VISIBLE_DEVICES=0 \
bash run.sh
```

### Enable Optional OpenAI Constraint Injection

```bash
USE_OPENAI_CONSTRAINT=1 \
MODE=all \
INFER_BACKEND=api \
bash run.sh
```

## Step 7: Expected Outputs

Main intermediate and output files:

```text
workspace/
├── preprocess/
│   └── 5_1_ultrasound_reports_convert.jsonl
└── infer/
    ├── ultrasound_prompt_result.jsonl
    ├── final_result_api.jsonl
    └── final_result.jsonl
```

Output description:

- `workspace/infer/ultrasound_prompt_result.jsonl`: prompt file used for the second-stage inference
- `workspace/infer/final_result_api.jsonl`: final output from the API backend
- `workspace/infer/final_result.jsonl`: final output from the local Swift backend

## Step 8: Common Options

You can override the default paths and behavior with environment variables:

```bash
IMAGE_ROOT=./data/samples/PatientID704_ExamID10143_trimester2
WORKSPACE_DIR=./workspace
CONFIG_PATH=./config/config.yaml
EXCEL_PATH=./data/metadata/pregnancy_stage.xlsx
USE_OPENAI_CONSTRAINT=0
INFER_BACKEND=api
MODE=all
```

Example:

```bash
IMAGE_ROOT=./data/samples/your_case \
USE_OPENAI_CONSTRAINT=1 \
INFER_BACKEND=api \
MODE=all \
bash run.sh
```


## Minimal Quick Start

```bash
conda create -n deepfetal python=3.10 -y
conda activate deepfetal
pip install -r requirements.txt

cat > .env.local <<'EOF'
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
MODE=all
INFER_BACKEND=api
USE_OPENAI_CONSTRAINT=0
EOF

bash run.sh
```


## Acknowledgement
We thank all the collaborators who supported the development and evaluation of DeepFetal. We also acknowledge the open-source community and prior research in medical imaging, multimodal learning, and large language models, which provided important foundations for this work.
