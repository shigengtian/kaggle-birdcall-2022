#!/bin/bash
python train.py --fold 0 --debug false --duration 30 --exp_no 2 --model tf_efficientnet_b0_ns --lr 1e-3 --epochs 30  --output tf_efficientnet_b0_ns
# python train.py --fold 1 --debug false --exp_no 1 --model tf_efficientnet_b0_ns --lr 1e-3 --epochs 40  --output tf_efficientnet_b0_ns_f1
# python train.py --fold 2 --debug false --exp_no 1 --model tf_efficientnet_b0_ns --lr 1e-3 --epochs 40  --output tf_efficientnet_b0_ns_f2
# python train.py --fold 3 --debug false --exp_no 1 --model tf_efficientnet_b0_ns --lr 1e-3 --epochs 40  --output tf_efficientnet_b0_ns_f3
# python train.py --fold 4 --debug false --exp_no 1 --model tf_efficientnet_b0_ns --lr 1e-3 --epochs 40  --output tf_efficientnet_b0_ns_f4