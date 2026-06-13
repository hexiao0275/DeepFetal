# A Deconfounded Multimodal AI System for Fetal Ultrasound Interpretation

 [Website](http://deepfetal.com/)  | [Model](https://huggingface.co/natureteam/DeepFetal)

> *DeepFetal is a deconfounded multimodal AI system for full-gestation fetal ultrasound interpretation, enabling traceable diagnostic reasoning and robust clinical decision support.*
> *Due to privacy and API access restrictions, online case testing is enabled only upon deployment. We kindly encourage users to download the model from Hugging Face and perform inference locally.*

For more details about our pipeline, please refer to our paper.


## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/hexiao0275/DeepFetal.git
cd DeepFetal
```

### Step 2: Create the Python Environment

We recommend using Conda with Python 3.10.

```bash
conda create -n deepfetal python=3.10 -y
conda activate deepfetal
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Required Files

```text
DeepFetal/
├── run.sh
├── deepfetal/
├── config/
│   └── config.yaml
├── data/
│   ├── metadata/
│   │   ├── pregnancy_stage.xlsx
│   │   ├── plane_translation.xlsx
│   │   └── select_patient_information.xlsx
│   └── samples/
│       ├── patient_1/
│       ├── patient_2/
│       └── ...
├── checkpoints/
│   ├── 1_1/1_cls_model.pt
│   ├── 1_2/2_3_cls_model.pth
│   └── checkpoint-merged/    
└── workspace/
```

### Step 5: Configure Environment Variables

Create `.env.local` in the project root:

```bash
PYTHON_BIN=./venv/bin/python

MODE=all
INFER_BACKEND=swift
INFER_TASK_IS_REPORT=2
USE_OPENAI_CONSTRAINT=0

# Only needed when INFER_BACKEND=api
# OPENAI_API_KEY=your_api_key
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-5.2
```

Key variables:

| Variable | Values | Description |
|----------|--------|-------------|
| `MODE` | `all`, `process`, `infer`| Pipeline mode |
| `INFER_BACKEND` | `swift`, `api` | `swift` for local GPU inference, `api` for OpenAI-compatible API |
| `INFER_TASK_IS_REPORT` | `0`, `1`, `2` | `0` = diagnosis only, `1` = report only, `2` = both |
| `USE_OPENAI_CONSTRAINT` | `0`, `1` | Enable image-constraint generation step before inference |
| `IMAGE_ROOT` | path | Input case folder (single case) or parent folder (batch mode) |
| `WORKSPACE_DIR` | path | Output workspace directory |
| `CUDA_VISIBLE_DEVICES` | `0`, `0,1`, ... | GPU selection for swift backend |


## Usage

### Single Case: Report + Diagnosis

```bash
IMAGE_ROOT=./data/samples/patient_1 \
INFER_TASK_IS_REPORT=2 \
INFER_BACKEND=swift \
MODE=all \
bash run.sh
```

### Single Task Modes

```bash
# Report generation only
INFER_TASK_IS_REPORT=1 MODE=all bash run.sh

# Diagnosis generation only
INFER_TASK_IS_REPORT=0 MODE=all bash run.sh

# Preprocess only (no inference)
MODE=process bash run.sh

# Inference only (assumes preprocessing is done)
MODE=infer bash run.sh
```


## Output Structure

### Single case (`MODE=all`)

```text
workspace/
├── preprocess/                           (intermediate files)
└── infer/
    ├── ultrasound_prompt_result.jsonl    (when INFER_TASK_IS_REPORT=0 or 1)
    ├── ultrasound_prompt_report.jsonl    (when INFER_TASK_IS_REPORT=2)
    ├── ultrasound_prompt_diagnosis.jsonl (when INFER_TASK_IS_REPORT=2)
    ├── final_result.jsonl               (swift output, single task)
    ├── final_result_report.jsonl        (swift output, report)
    └── final_result_diagnosis.jsonl     (swift output, diagnosis)
```



### Output Format

Each line in `final_result_*.jsonl` is a JSON object containing the model's response:

**Report** (`TASK_ULTRASOUND_REPORT`): Full structured ultrasound report with FINDINGS and IMPRESSION sections, including fetal measurements, anatomical observations, and clinical assessment.

**Diagnosis** (`TASK_ULTRASOUND_DIAGNOSIS`): Concise diagnostic impression summarizing key findings and abnormalities.


## Model Weights

| Path | Size | Description |
|------|------|-------------|
| `checkpoints/1_1/1_cls_model.pt` | 3.1M | Quality detection/filtering model |
| `checkpoints/1_2/2_3_cls_model.pth` | 107M | 2_3-trimester-class fetal ultrasound plane classifier |
| `checkpoints/checkpoint-600-merged/` | 17G | multimodal model (36 layers) |


## Acknowledgement

We thank all the collaborators who supported the development and evaluation of DeepFetal. We also acknowledge the open-source community and prior research in medical imaging, multimodal learning, and large language models, which provided important foundations for this work.
