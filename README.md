# LLM Knowledge Distillation

This repository contains code for knowledge distillation of Large Language Models (LLMs), specifically focusing on T5-based models.

## Features
- Multi-teacher knowledge distillation
- Support for various LLM models (T5-based, Llama-based, etc.)
- Flexible data processing pipeline
- Customizable training configurations


## Installation

```bash
git clone https://github.com/yourusername/llm_kd.git
cd llm_kd
pip install -r requirements.txt
```

## Quick Start

### Train

To train a model, firstly you need to prepare the dataset -- default directory is `.data/`. Then setup the `train_config.yaml` file, which contains the parameters for dataset, training, etc.

Finally, you can use the following command:

```bash
python train_tinyllm.py --config_dir train_config.yaml
```

# Reference 
- [YikunHan42/TinyLLM](https://github.com/YikunHan42/TinyLLM)