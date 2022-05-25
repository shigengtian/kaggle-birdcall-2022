#!/bin/bash

#python train.py --fold 0 --debug false --weight weights/exp_2/tf_efficientnet_b0_ns/fold-0.bin --duration 5 --exp_no 5 --model tf_efficientnet_b0_ns --lr 1e-4 --epochs 20  --output tf_efficientnet_b0_ns
# python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model tf_efficientnetv2_s  --epochs 30  --output tf_efficientnetv2_s
# python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model tf_efficientnetv2_s_in21k  --epochs 30  --output tf_efficientnetv2_s_in21k

#python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model tf_efficientnetv2_m  --epochs 30  --output tf_efficientnetv2_m
#python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model tf_efficientnetv2_m_in21k  --epochs 30  --output tf_efficientnetv2_m_in21k
# python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model tf_efficientnet_b1_ns --epochs 30  --output tf_efficientnet_b1_ns
# python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model tf_efficientnet_b2_ns --epochs 30  --output tf_efficientnet_b2_ns
# python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model tf_efficientnet_b3_ns --epochs 30  --output tf_efficientnet_b3_ns


python train2.py --fold 0 --weight weights/exp_8/tf_efficientnet_b3_ns/fold-0.bin --debug false  --duration 5 --exp_no 10 --model tf_efficientnet_b3_ns --epochs 30  --output tf_efficientnet_b3_ns

# python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model mixnet_m --epochs 30  --output mixnet_m
# # python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model repvgg_b0 --epochs 30  --output repvgg_b0
# # python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model resnest26d --epochs 30  --output resnest26d
# python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model resnest50d_1s4x24d --epochs 30  --output resnest50d_1s4x24d
# python train.py --fold 0 --debug false  --duration 30 --exp_no 5 --model resnext50d_32x4d --epochs 40  --output resnext50d_32x4d
# python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model efficientnet_b0 --epochs 30  --output efficientnet_b0
# python train.py --fold 0 --debug false  --duration 30 --exp_no 8 --model ecaresnet26t --epochs 30  --output ecaresnet26t

#python train.py --fold 0 --debug false  --duration 5 --exp_no 5 --model tf_efficientnet_b0_ns --lr 1e-4 --epochs 20  --output tf_efficientnet_b0_ns
# python train_mix.py --fold 0 --debug false --exp_no 1 --model tf_efficientnet_b0_ns --lr 1e-3 --epochs 30  --output tf_efficientnet_b0_ns_f4

# python train2.py --fold 0 --weight weights/exp_8/resnest50d_1s4x24d/fold-0.bin --debug false  --duration 5 --exp_no 9 --model resnest50d_1s4x24d --epochs 20  --output resnest50d_1s4x24d
# python train2.py --fold 0 --weight weights/exp_8/efficientnet_b0/fold-0.bin --debug false  --duration 5 --exp_no 9 --model efficientnet_b0 --epochs 20  --output efficientnet_b0
# python train2.py --fold 0 --weight weights/exp_8/ecaresnet26t/fold-0.bin --debug false  --duration 5 --exp_no 9 --model ecaresnet26t --epochs 20  --output ecaresnet26t
