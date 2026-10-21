#!/bin/bash

python pipeline_temporal.py  --qat --convert both --loops 2 --force --encoders "1 1" --cand 0.25 \
    --test-clips cabinet_D65_100_local_0 cabinet_D50_36_static_0
