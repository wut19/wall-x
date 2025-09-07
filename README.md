# Wall-X

## Overview

Wall-X is a multimodal foundation model designed for robotics applications, combining vision, language, and action capabilities. The model architecture is built upon Qwen2.5-3B-VL with specialized adaptations for robotic control tasks.

## Environment Setup

Create and activate conda environment:
```bash
conda create --name wallx python=3.10
conda activate wallx
```

Install requirements:
```bash
pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Install lerobot:
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

Install wall_x:
```bash
git submodule update --init --recursive
MAX_JOBS=4 pip install --no-build-isolation --verbose .
```

## Training

### Finetune on LeRobot Datasets

Before training, please refer to `workspace/README.md` for detailed configuration instructions including:

Training script path configuration

- GPU setup
- Model and data paths
- Robot DOF configuration
- Training hyperparameters

```bash
bash ./workspace/lerobot_example/run.sh
```

## Inference

For model inference, please refer to:

```bash
python ./scripts/fake_inference.py
```

This script demonstrates how to:
- Load the Wall-OSS model using `Qwen2_5_VLMoEForAction.from_pretrained()`
- Prepare input data including proprioceptive information, attention masks, and dataset specifications
- Run inference in validation mode with proper data types (bfloat16)
- Validate model outputs and check for numerical stability

To generate an open-loop comparison plot, please follow:

```bash
python ./scripts/draw_openloop_plot.py
```