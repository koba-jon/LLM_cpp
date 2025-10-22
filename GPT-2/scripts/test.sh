#!/bin/bash

DATA='the-verdict'

./GPT-2 \
    --test true \
    --dataset ${DATA} \
    --tokenizer "dist/tokenizer.json" \
    --vocab_size 50277 \
    --endoftext 0 \
    --padding 1 \
    --gpu_id 0
