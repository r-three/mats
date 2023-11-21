#!/bin/bash

mkdir -p /fruitbasket/models/google/t5-large-lm-adapt
cd /fruitbasket/models/google/t5-large-lm-adapt

wget "https://huggingface.co/google/t5-large-lm-adapt/raw/main/config.json"
wget "https://huggingface.co/google/t5-large-lm-adapt/raw/main/generation_config.json"
wget "https://huggingface.co/google/t5-large-lm-adapt/resolve/main/pytorch_model.bin"
wget "https://huggingface.co/google/t5-large-lm-adapt/raw/main/special_tokens_map.json"
wget "https://huggingface.co/google/t5-large-lm-adapt/resolve/main/spiece.model"
wget "https://huggingface.co/google/t5-large-lm-adapt/raw/main/tokenizer.json"
wget "https://huggingface.co/google/t5-large-lm-adapt/raw/main/tokenizer_config.json"