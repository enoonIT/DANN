#!/usr/bin/env bash
python new_main.py --epoch 600 --source mnist_m mnist synth --target svhn --data_aug_mode simple-tuned --source_limit 20000 --target_limit 20000 --use_deco --generalization --classifier multi --deco_incremental $1
