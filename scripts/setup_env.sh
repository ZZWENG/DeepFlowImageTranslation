#!/usr/bin/env bash

# setting up environment on GPU instance
sudo apt-get install python-pip
pip install tqdm enum34 functools32
pip install torch torchvision
mkdir sampling
mkdir logs
mkdir logs/checkpoints
