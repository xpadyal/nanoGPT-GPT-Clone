# GPT Model from Scratch

Welcome to the repository for building a GPT (Generative Pre-trained Transformer) model from scratch. This project was developed as part of the "Zero to Hero in Neural Networks" course by Andrej Karpathy. The goal is to provide a hands-on understanding of how modern language models like GPT are constructed and trained.

## Check out post here: 
https://www.linkedin.com/posts/sahil-padyal_ai-machinelearning-gpt-activity-7206758862731812864-BJ3b?utm_source=share&utm_medium=member_desktop

## Table of Contents
- [Overview](#overview)
- [Implementation Details](#implementation-details)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Tokenization](#tokenization)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Key Learnings](#key-learnings)
- [Acknowledgements](#acknowledgements)

## Overview
This project involves implementing a GPT model from scratch using Python and PyTorch. The model is trained on the tiny Shakespeare dataset, which includes all works of Shakespeare concatenated into a single file. The goal is to generate text that mimics Shakespearean language.

## Implementation Details
- **Programming Language**: Python
- **Framework**: PyTorch
- **Development Environment**: Jupyter Notebook
- **Training Data**: Tiny Shakespeare dataset (~1MB)
- **Parameters**: The model is trained with 124 million parameters, similar to the early GPT-2 model.

## Model Architecture
The model is based on the Transformer architecture introduced in the "Attention is All You Need" paper. Key components include:
- Position encodings
- Token encodings
- Transformer blocks
- Layer normalization
- Final linear layer

## Training Process
The training process involves two main stages:
1. **Pre-training**: The model is pre-trained on the tiny Shakespeare dataset to model the sequence of characters.
2. **Fine-tuning**: The pre-trained model is further fine-tuned for specific tasks to improve its performance.

## Tokenization
A character-level tokenizer is used to convert the text into integers. This approach keeps the implementation simple and effective.

## Dataset
The tiny Shakespeare dataset is used for training. It consists of all works of Shakespeare concatenated into a single file, totaling about 1MB in size.

## Requirements
To run this project, you will need the following:
- Python 3.7+
- PyTorch
- Jupyter Notebook

Install the required packages using pip:
```bash
pip install torch jupyter
