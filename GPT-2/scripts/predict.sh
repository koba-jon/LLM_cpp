#!/bin/bash

DATA='the-verdict'

./GPT-2 \
    --predict true \
    --dataset ${DATA} \
    --tokenizer "dist/tokenizer.json" \
    --vocab_size 50277 \
    --endoftext 0 \
    --padding 1 \
    --seed 0 \
    --gpu_id 0
