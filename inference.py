import argparse
import scipy, math
from scipy import ndimage
import cv2
import numpy as np
import sys
import json
import models
import dataloaders
from utils.helpers import colorize_mask
from utils.pallete import get_voc_pallete
from utils import metrics
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from math import ceil
from PIL import Image
from pathlib import Path


class testDataset(Dataset):
    def __init__(self, images):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        images_path = Path(images)
        self.filelist = list(images_path.glob("*.jpg"))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        image_path = self.filelist[index]
        image_id = str(image_path).split("/")[-1].split(".")[0]
        image = Image.open(image_path)
        image = self.normalize(self.to_tensor(image))
        return image, image_id

def multi_scale_predict(model, image, scales, num_classes, flip=True):
    H, W = (image.size(2), image.size(3))
    upsize = (ceil(H / 8) * 8, ceil(W / 8) * 8)
    upsample = nn.Upsample(size=upsize, mode='bilinear', align_corners=True)
    pad_h, pad_w = upsize[0] - H, upsize[1] - W
    image = F.pad(image, pad=(0, pad_w, 0, pad_h), mode='reflect')

    total_predictions = np.zeros((num_classes, image.shape[2], image.shape[3]))

    for scale in scales:
        scaled_img = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_prediction = upsample(model(scaled_img))

        if flip:
            fliped_img = scaled_img.flip(-1)
            fliped_predictions = upsample(model(fliped_img))
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions[:, :H, :W]

def main():
    args = parse_arguments()

    # CONFIG
    assert args.config
    config = json.load(open(args.config))
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    # DATA
    testdataset = testDataset(args.images)
    loader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=1)
    num_classes = 21
    palette = get_voc_pallete(num_classes)

    # MODEL
    config['model']['supervised'] = True; config['model']['semi'] = False
    model = models.CCT(num_classes=num_classes,
                        conf=config['model'], testing=True)
    checkpoint = torch.load(args.model)
    model = torch.nn.DataParallel(model)
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    model.cuda()

    if args.save and not os.path.exists('outputs'):
        os.makedirs('outputs')

    # LOOP OVER THE DATA
    tbar = tqdm(loader, ncols=100)
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    labels, predictions = [], []

    for index, data in enumerate(tbar):
        image, image_id = data
        image = image.cuda()

        # PREDICT
        with torch.no_grad():
            output = multi_scale_predict(model, image, scales, num_classes)
        prediction = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)

        # SAVE RESULTS
        prediction_im = colorize_mask(prediction, palette)
        prediction_im.save('outputs/'+image_id[0]+'.png')

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='configs/config.json',type=str,
                        help='Path to the config file')
    parser.add_argument( '--model', default=None, type=str,
                        help='Path to the trained .pth model')
    parser.add_argument('--images', default="/home/yassine/Datasets/vision/PascalVoc/VOC/VOCdevkit/VOC2012/test_images", type=str,
                        help='Test images for Pascal VOC')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

