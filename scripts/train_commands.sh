#!/usr/bin/env bash

# Train
python train.py --name facades --model flowgan --gpu_ids "[0, 1]"
python train.py --name facades --model cyclegan --gpu_ids "[0, 1]"
python train.py --name cityscapes --model flowgan --gpu_ids "[0, 1]"
python train.py --name cityscapes --model cyclegan --gpu_ids "[0, 1]"

# Test
python train.py --name facades --model flowgan --gpu_ids "[0, 1]" --is_train 0
python train.py --name facades --model cyclegan --gpu_ids "[0, 1]" --is_train 0
python train.py --name cityscapes --model flowgan --gpu_ids "[0, 1]" --is_train 0
python train.py --name cityscapes --model cyclegan --gpu_ids "[0, 1]" --is_train 0

# FID Scores
#python fid_score.py ../myproject/from_gcloud/facades_flowgan_fakeB_46 ../myproject/from_gcloud/facades_flowgan_realA_46
#python fid_score.py ../myproject/from_gcloud/facades_flowgan_fakeA_46 ../myproject/from_gcloud/facades_flowgan_realB_46
#python fid_score.py ../myproject/from_gcloud/facades_cyclegan_fakeB_50 ../myproject/from_gcloud/facades_cyclegan_realA_50
#python fid_score.py ../myproject/from_gcloud/facades_cyclegan_fakeA_50 ../myproject/from_gcloud/facades_cyclegan_realB_50
#
#python fid_score.py ../myproject/from_gcloud/cityscapes_flowgan_fakeB_16 ../myproject/from_gcloud/cityscapes_flowgan_realA_16
#python fid_score.py ../myproject/from_gcloud/cityscapes_flowgan_fakeA_16 ../myproject/from_gcloud/cityscapes_flowgan_realB_16
#python fid_score.py ../myproject/from_gcloud/cityscapes_cyclegan_fakeB_25 ../myproject/from_gcloud/cityscapes_cyclegan_realA_25
#python fid_score.py ../myproject/from_gcloud/cityscapes_cyclegan_fakeA_25 ../myproject/from_gcloud/cityscapes_cyclegan_realB_25
