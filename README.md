# Comparison of LoRA and MeZO+LoRA algorithms

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

The main goal is to compare LoRA and MeZO+LoRA fine-tuning algorithms.

This part of the code is for experiments on RoBERTa-large model. It is based on [LM-Kernel-FT](https://github.com/princeton-nlp/LM-Kernel-FT) and [LM-BFF](https://github.com/princeton-nlp/LM-BFF).

The model will be trained on the [MRPC](https://huggingface.co/datasets/SetFit/mrpc) dataset.

To train the model on our dataset, enter the following commands.

For the LoRA algorithm:

```bash
# LoRA fine-tuning
TASK=MRPC K=16 SEED=42 BS=64 LR=1e-4 EPS=1e-3 MODEL=roberta-large MODE=lora STEP=1000 EVAL_STEP=100 bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16 --output_dir ./result/MRPC-roberta-large-prompt-standard-k16-roberta-large-lora-seed42-bs64-lr1e-4-eps1e-3-wd0-step1000-evalstep100
```

For the MeZO + LoRA algorithm:

```
# MeZO + LoRA
TASK=MRPC K=16 SEED=42 BS=64 LR=1e-4 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora STEP=1000 EVAL_STEP=100 bash mezo.sh --apply_lora --lora_r 8 --lora_alpha 16 --output_dir ./result/MRPC-roberta-large-prompt-standard-k16-roberta-large-mezo-lora-seed42-bs64-lr1e-4-eps1e-3-wd0-step1000-evalstep100
```

## Gather results

To analyze the results, you can go to the `.result` folder and the corresponding algorithm folder. It will contain `txt` files with basic quality metrics.

## Research part

### Overview of algorithms

MeZO (https://arxiv.org/pdf/2305.17333) is the state-of-the-art gradient-free optimization method for Large Language Models (LLMs) fine-tuning. The MeZO algorithm is a way to fine-tune LLMs using only forward passes, which means it doesn't need as much memory as backpropagation. 

The LoRA is a Parameter-Efficient-Fine-Tuning (PEFT) approach that freezes a pre-trained model and applies additional trainable parameters (weights) that are factorized with small decomposition rank r.

### Model Architecture:

Base Model: RoBERTa-large (355 million parameters).

Task: Sequence classification (binary paraphrase detection).

### Dataset:

MRPC (Microsoft Research Paraphrase Corpus):

Objective: Determine if two sentences are semantically equivalent.

Size: 3,668 training examples, 408 validation examples, 1,725 test examples.

Class Balance: ~65% positive (paraphrases), ~35% negative (non-paraphrases).

### Hyperparameters:

Batch size: 64

Learning rate: 1e-4

Training steps: 1,000

Optimizer: Adam (LoRA), MeZO-SGD (MeZO+LoRA)

### Performance Comparison

|  Metric  | LoRA (Validation) | MeZO+LoRA (Validation) | LoRA (Test) | MeZO+LoRA (Test) |
|----------|-------------------|------------------------|-------------|------------------|
| Loss     |        0.94       |       **0.62**         |     1.37    |      **0.65**    |
| Accuracy |     **78.1%**     |         68.7%          |   **72.8%** |        63.9%     |
| F1-score |     **81.1%**     |         70.6%          |   **80.1%** |        71.8%     |
| Acc + F1 |     **79.6%**     |         69.7%          |   **76.5%** |        67.9%     |

### Training Time

LoRA: 835 sec

MeZO+LoRA: 778 sec

Speed advantage: MeZO+LoRA is ~7% faster due to the absence of backpropagation.

### Memory Consumption

LoRA: 5.31 GB.

MeZO+LoRA: 2.3 GB (57% reduction).

###  Interpretation of Results

The sharp increase in LoRA’s test Loss (0.94 → 1.37) suggests potential overfitting. In contrast, MeZO+LoRA maintains stable Loss (0.62 → 0.65), indicating better generalization.

LoRA prioritizes quality but requires double the memory. MeZO+LoRA sacrifices ~9% test Accuracy for significant memory and time savings. 

The key feature of MeZO+LoRA: The drastic memory savings make it feasible for resource-constrained environments, or it is better to train larger models with a much larger number of parameters.

### Conclusions

The results highlight the classic trade-off between quality and resource efficiency. LoRA is optimal for accuracy-critical tasks, while MeZO+LoRA enables deployment in constrained environments.

Potentially, the MeZO+LoRA algorithm's gains will be more visible for large models with billions of parameters, where the training time and memory usage will be several times less than that of the LoRA algorithm, but Accuracy/F1 will not be much less.




