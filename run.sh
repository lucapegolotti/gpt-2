#!/bin/bash
torchrun --standalone --nproc-per-node 8 train_gpt2.py