#!/usr/bin/env bash

export HF_HOME="${PWD}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export WANDB_DISABLED=true

mkdir -p "${HUGGINGFACE_HUB_CACHE}"
