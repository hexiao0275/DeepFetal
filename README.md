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
│   │   └── plane_translation.xlsx
│   └── samples/
│       ├── clinical_information.xlsx    (optional)
│       ├── patient_1_inpatient/
│       ├── patient_2_inpatient/
│       └── ...
├── checkpoints/
│   ├── 1_1/1_cls_model.pt
│   ├── 1_2/2_3_cls_model.pth
│   └── checkpoint-merged/    
└── workspace/
```

`clinical_information.xlsx` is optional. If an `.xlsx` file exists under the case folder or under `data/samples/`, the pipeline will look for a row that exactly matches the case folder name and read its `checklist` value. In older sample naming, `inpatient` indicates an inpatient visit; for screening visits, use the outpatient/screening visit setting.

### Step 5: Configure Environment Variables

Create `.env.local` in the project root:

```bash
PYTHON_BIN=./venv/bin/python

# Required by the semantic agent module
export OPENAI_API_KEY="your_openai_official_api_key"
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-5.2
```


Key variables:

| Variable | Values | Description |
|----------|--------|-------------|
| `MODE` | `all`, `process`, `infer`| Pipeline mode |
| `IS_EARLY` | `0`, `1` | `0` = second- and third-trimester pipeline, `1` = first-trimester pipeline |
| `VISIT_TYPE_IS_SCREENING` | `0`, `1` | `0` = inpatient/clinical visit, `1` = outpatient/screening visit |
| `INFER_TASK_IS_REPORT` | `0`, `1`| `0` = diagnosis only, `1` = report only|
| `OPENAI_API_KEY` | string | API key for the required semantic agent module |
| `OPENAI_BASE_URL` | URL | OpenAI-compatible API endpoint for the semantic agent module |
| `OPENAI_MODEL` | model name | Model used by the semantic agent module. Default: `gpt-5.2` |
| `IMAGE_ROOT` | path | Input case folder, single case only |
| `WORKSPACE_DIR` | path | Output workspace directory |
| `CUDA_VISIBLE_DEVICES` | `0`, `0,1`, ... | GPU selection for Swift inference |


## Usage

### Single Case: Diagnosis

```bash
IMAGE_ROOT=./data/samples/patient_1_inpatient \
IS_EARLY=0 \
VISIT_TYPE_IS_SCREENING=0 \
INFER_TASK_IS_REPORT=0 \
MODE=all \
bash run.sh
```


### Single Case: Report

```bash
IMAGE_ROOT=./data/samples/patient_1_inpatient \
IS_EARLY=0 \
VISIT_TYPE_IS_SCREENING=0 \
INFER_TASK_IS_REPORT=1 \
MODE=all \
bash run.sh
```

### Process / Infer Separately

```bash
MODE=process bash run.sh

MODE=infer bash run.sh
```


## Output Structure

### Single case (`MODE=all`)

```text
workspace/
├── preprocess/                           (intermediate files)
└── infer/
    ├── ultrasound_prompt_result.jsonl    (semantic agent output; generated for both diagnosis and report modes)
    └── final_result.jsonl               (final output)
```



### Output Format

Each line in `final_result_*.jsonl` is a JSON object containing the model's response:

**Report** (`TASK_ULTRASOUND_REPORT`): Full structured ultrasound report with FINDINGS and IMPRESSION sections, including fetal measurements, anatomical observations, and clinical assessment.

**Diagnosis** (`TASK_ULTRASOUND_DIAGNOSIS`): Concise diagnostic impression summarizing key findings and abnormalities.


## Model Weights

| Path | Size | Description |
|------|------|-------------|
| `checkpoints/1_1/1_cls_model.pt` | Quality detection/filtering model |
| `checkpoints/1_2/2_3_cls_model.pth` | 2_3-trimester-class fetal ultrasound plane classifier |
| `checkpoints/checkpoint-merged/` | multimodal model (36 layers) |


## Acknowledgement

We thank all the collaborators who supported the development and evaluation of DeepFetal. We also acknowledge the open-source community and prior research in medical imaging, multimodal learning, and large language models, which provided important foundations for this work.
