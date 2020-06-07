#!/bin/bash

FILENAME="models/backbones/pretrained/3x3resnet50-imagenet.pth"
FILEID="1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx"

mkdir -p models/backbones/pretrained
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O $FILENAME