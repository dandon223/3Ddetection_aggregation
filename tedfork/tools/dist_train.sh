#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=6,7 nohup python3 -m torch.distributed.launch --nproc_per_node=2 train.py --cfg_file cfgs/models/nuscenes/TED-S.yaml --launcher pytorch > log.txt&
