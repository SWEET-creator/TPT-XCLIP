#!/bin/bash

data_root='./dataset'
testsets='target'
arch=RN50
# arch=ViT-B/16
bs=8
ctx_init=In_this_video_he_is

CUDA_LAUNCH_BLOCKING=1 python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --ctx_init ${ctx_init}