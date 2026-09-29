# MultimodalAI — TREAT-MMTB 2026 Task 1

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/Deepnoid/TREAT-MMTB-2026-Task1-MultimodalAI) [![GitHub](https://img.shields.io/badge/GitHub-Code-181717?logo=github)](https://github.com/deepnoid-ai/TREAT-MMTB-2026-Task1-MultimodalAI) [![Leaderboard](https://img.shields.io/badge/Leaderboard-Rank%203-2EA44F)](https://github.com/mi2rl-challenge/treat-mmtb.miccai2026/blob/main/leader_board_point.json) [![TREAT-MMTB 2026](https://img.shields.io/badge/MICCAI%202026-TREAT--MMTB-1F6FEB)](https://treat-mmtb.mi2rl.co/)

Cavity detection and segmentation from chest X-ray DICOMs for [TREAT-MMTB 2026](https://treat-mmtb.mi2rl.co/), with optional classification-only inference.

## Challenge result

MultimodalAI's [official final external leaderboard](https://github.com/mi2rl-challenge/treat-mmtb.miccai2026/blob/main/leader_board_point.json) result:

| Rank | Final score | Detection accuracy | Dice |
| --- | --- | --- | --- |
| 3 | 0.6001 | 0.8112 | 0.1077 |

## Installation

Python 3.10, with PyTorch for CUDA 11.8 and dependencies from [requirements.txt](requirements.txt):

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

## AutoModel inference

Code and weights load automatically from Hugging Face.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Deepnoid/TREAT-MMTB-2026-Task1-MultimodalAI",
    trust_remote_code=True,
    mode="seg",
).to("cuda").eval()

result = model.predict("/path/to/image.dcm", output_dir=None)[0]
mask = result["mask"]
cavity = result["cavity"]

results = model.predict("/path/to/case_folders", output_dir="output/task1")
```

`mode="seg"` is the default. Set `mode="cls"` in `from_pretrained` to load only the classifier. Use `.to("cpu")` for CPU inference.

- **Input:** a `.dcm`/`.dicom` file, a flat DICOM folder, or a root containing case folders. Extensions are case-insensitive. `our_id` is the file stem or case-folder name; each case uses its first sorted DICOM. Mixed flat files and case folders are rejected.
- **Return:** always a list of results containing `our_id` and `cavity` (`0` or `1`). With `output_dir=None` (default), no output is written; `mask` is a NumPy array on the original image grid, or `None` in classification-only mode.
- **Save:** setting `output_dir` writes `prediction.csv` (`our_id,cavity`). Segmentation also writes binary `uint8` masks as `<our_id>.nii.gz`, preserving image geometry, and returns `mask_path` instead of `mask`.

## CLI inference

Place checkpoints in `weights/` beside `predict.py` (or set `--weights`). Architecture and threshold settings are read from `config.json` (or set `--config`).

```bash
python predict.py --input /path/to/case_folders --output output/seg
python predict.py --input /path/to/image.dcm --output output/cls --mode cls
```

## Models and method

| Checkpoint in `weights/` | Architecture |
| --- | --- |
| `model_0.safetensors` | DINOv3 ViT-L/16 cavity classifier with attention pooling |
| `model_1.safetensors`–`model_4.safetensors` | Four DINOv3 ConvNeXt-L segmenters with pyramid pooling and U-Net decoders |

Input combines histogram equalization, CLAHE, and grayscale channels at 1024 × 1024. Classifier probability **≥ 0.70** indicates a cavity. The four segmentation probability maps are averaged, restored to the original image grid, and thresholded at **> 0.79**. Negative cases receive empty masks; positive cases with empty masks use a top-1% fallback.

## Docker

Requires an NVIDIA GPU, driver, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Place all five [checkpoints](https://huggingface.co/Deepnoid/TREAT-MMTB-2026-Task1-MultimodalAI/tree/main/weights) in `weights/` beside the Dockerfile. Build with network access; inference runs offline.

```bash
docker build -t multimodalai-task1:latest .
mkdir -p output
docker run --rm --gpus all --network none \
  -v /absolute/path/to/case_folders:/input:ro \
  -v "$PWD/output":/output multimodalai-task1:latest
```

Append `--input /input/image.dcm` for a single file or `--mode cls` for classification only. Challenge submission uses the default segmentation mode with `/input/<our_id>/*.dcm`.

## Builds on

Vision-language pretraining followed [GLINT](https://arxiv.org/abs/2606.03180) (Park et al., 2026), using Meta AI / FAIR's [DINOv3](https://arxiv.org/abs/2508.10104) encoders (Siméoni et al., 2025), [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) sentence embeddings (Song et al., 2020), and report labels using [Qwen3.6-35B-A3B](https://qwen.ai/blog?id=qwen3.6-35b-a3b).

| Dataset | Use |
| --- | --- |
| [TREAT-MMTB 2026](https://doi.org/10.5281/zenodo.19732124) | Classification and segmentation training |
| [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) | Vision-language pretraining |
| [TB Portals](https://tbportals.niaid.nih.gov/) | Cavity classification training |

## Acknowledgements

This work was supported by the Technology Innovation Program (RS-2025-02221011, Development of Medical-Specialized Multimodal Hyperscale Generative AI Technology for Global Integration) funded by the Ministry of Trade Industry & Energy (MOTIE, South Korea), and by the “Advanced GPU Utilization Support Program” funded by the Government of the Republic of Korea (Ministry of Science and ICT).

Data were obtained from the [TB Portals](https://tbportals.niaid.nih.gov), which is an open-access TB data resource supported by the National Institute of Allergy and Infectious Diseases (NIAID) Office of Cyber Infrastructure and Computational Biology (OCICB) in Bethesda, MD. These data were collected and submitted by members of the [TB Portals Consortium](https://tbportals.niaid.nih.gov/Partners). Investigators and other data contributors that originally submitted the data to the TB Portals did not participate in the design or analysis of this study (Rosenthal et al., 2017).
