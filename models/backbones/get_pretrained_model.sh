#!/bin/bash

FILENAME="models/backbones/pretrained/3x3resnet50-imagenet.pth"

mkdir -p models/backbones/pretrained
wget https://github.com/yassouali/CCT/releases/download/v0.1/3x3resnet50-imagenet.pth -O $FILENAME
