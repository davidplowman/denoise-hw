#!/bin/bash

python pipeline_temporal.py  --ch 32 --wide-low --qat --convert both --loops 3 --force --encoders "1 2" --cand 0.25 \
    --test-clips cabinet_D65_100_local_0 cabinet_D50_36_static_0
python pipeline_temporal.py  --ch 32 --wide-low --qat --convert both --loops 3 --force --encoders "2 2" --cand 0.25 \
    --test-clips cabinet_D65_100_local_0 cabinet_D50_36_static_0
python pipeline_temporal.py  --ch 32 --wide-low --qat --convert both --loops 3 --force --encoders "1 3" --cand 0.25 \
    --test-clips cabinet_D65_100_local_0 cabinet_D50_36_static_0
python pipeline_temporal.py  --ch 32 --wide-low --qat --convert both --loops 3 --force --encoders "2 3" --cand 0.25 \
    --test-clips cabinet_D65_100_local_0 cabinet_D50_36_static_0
