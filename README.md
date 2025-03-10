# Comparison of LoRa and MeZO+LoRa algorithms

## Installation

Please install the PyTorch (`pytorch` following [https://pytorch.org](https://pytorch.org)) and Transformers (`transformers`). This code is tested on `torch==2.1.0.dev20230514+cu118` and `transformers==4.28.1` with Python 3.9.7. It is recommended to use these versions.

## Prepare the data

We pack the datasets [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). Please download it and extract the files to `./data/original`, or run the following commands:

```bash
cd data
bash download_dataset.sh
```

Then use the following command to generate the data we need:

```bash
for K in 16 512; do
    # Generate k-shot splits for seeds 13,21,42,87,100 with a maximum of 1k test examples in data/k-shot-1k-test,
    # where k is the number of training/validation examples per label
    python tools/generate_k_shot_data.py --mode k-shot-1k-test --k $K
done
```

See `tools/generate_k_shot_data.py` for more options. For results in the paper, we use the default options: we take `K=16` and `K=512` and take 5 different seeds of 13, 21, 42, 87, 100. The few-shot data will be generated to `data/k-shot-1k-test`. In the directory of each dataset, there will be folders named as `$K-$SEED` indicating different dataset samples.

## Usage

Use `run.py` for all functions and refer to `run.py` for the usage of all arguments.
```bash
python run.py {ARGUMENTS}
```

The main goal is to compare LoRa and MeZO+LoRa fine-tuning algorithms.

MeZO (https://arxiv.org/pdf/2305.17333) is the state-of-the-art gradient-free optimization method for Large Language Models (LLMs) fine-tuning. The MeZO algorithm is a way to fine-tune LLMs using only forward passes, which means it doesn't need as much memory as backpropagation. The LoRA is a Parameter-Efficient-Fine-Tuning (PEFT) approach that freezes a pre-trained model and applies additional trainable parameters (weights) that are factorized with small decomposition rank r.

This part of the code is for experiments on RoBERTa-large model. It is based on [LM-Kernel-FT](https://github.com/princeton-nlp/LM-Kernel-FT) and [LM-BFF](https://github.com/princeton-nlp/LM-BFF).

The model will be trained on the [MRPC](https://huggingface.co/datasets/SetFit/mrpc) dataset.

To train the model on our dataset, enter the following commands.

For the LoRa algorithm:

```bash
# LoRA fine-tuning
TASK=MRPC K=16 SEED=42 BS=64 LR=1e-4 EPS=1e-3 MODEL=roberta-large MODE=lora STEP=1000 EVAL_STEP=100 bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16 --output_dir ./result/MRPC-roberta-large-prompt-standard-k16-roberta-large-lora-seed42-bs64-lr1e-4-eps1e-3-wd0-step1000-evalstep100
```

For the MeZO + LoRa algorithm:

```
# MeZO + LoRA
TASK=MRPC K=16 SEED=42 BS=64 LR=1e-4 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora STEP=1000 EVAL_STEP=100 bash mezo.sh --apply_lora --lora_r 8 --lora_alpha 16 --output_dir ./result/MRPC-roberta-large-prompt-standard-k16-roberta-large-mezo-lora-seed42-bs64-lr1e-4-eps1e-3-wd0-step1000-evalstep100
```

## Gather results

To analyze the results, you can go to the `.result` folder and the corresponding algorithm folder. It will contain `txt` files with basic quality metrics.

## Research part


