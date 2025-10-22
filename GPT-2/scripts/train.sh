#!/bin/bash

DATA='the-verdict'

./GPT-2 \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --tokenizer "dist/tokenizer.json" \
    --vocab_size 50277 \
    --endoftext 0 \
    --padding 1 \
    --batch_size 8 \
    --gpu_id 0
