from base import BaseDataSet, BaseDataLoader
from utils import pallete
import pathlib
import numpy as np
import os
import scipy
from glob import glob
import torch
from PIL import Image, ImageOps
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json


class VOCDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 4

        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split == "val":
            print('Init val files')
            files = sorted(glob(os.path.join(self.root, 'labelled', 'el_images_5_val', '*png')))
            labels = [os.path.join(self.root, 'labelled', 'el_masks_1', os.path.basename(e)) for e in files]
        elif self.split == "train_supervised":
            print('Init supervised files')
            files = sorted(glob(os.path.join(self.root, 'labelled', 'el_images_4_train', '*png')))
            files = [e for e in files if 'mirror' not in e and 'flip' not in e and 'rotate' not in e]
            labels = [os.path.join(self.root, 'labelled', 'el_masks_1', os.path.basename(e)) for e in files]
        elif self.split == "train_unsupervised":
            print('Init unsup files')
            files = list(pathlib.Path(os.path.join(self.root, 'all_data')).rglob('*.png'))
            files = [str(e) for e in files]
            print('Finished init unsup files')
            labels = ['/mnt/data/home/david/datasets/el/dataset/labelled/el_masks_1/ARTS_00001_r4_c4.png'] * len(files)
        else:
            raise ValueError(f"Invalid split name {self.split}")

        print(f'Num images: {len(files)}, Num labels: {len(labels)}')
        self.files, self.labels = files, labels

    def _load_data(self, index):
        image_path = self.files[index]
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]

        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id + ".png")
        else:
            label_path = self.labels[index]

        if label_path:
            label = np.asarray(Image.open(label_path), dtype=np.int32)
        else:
            label = None

        return image, label, image_id


class VOCDataset_bak(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 21

        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')
        if self.split == "val":
            file_list = os.path.join("dataloaders/voc_splits", f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join("dataloaders/voc_splits", f"{self.n_labeled_examples}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id + ".png")
        else:
            label_path = os.path.join(self.root, self.labels[index][1:])
        label = np.asarray(ImageOps.grayscale(Image.open(label_path)), dtype=np.int32)
        return image, label, image_id


class VOC(BaseDataLoader):
    def __init__(self, kwargs):

        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')

        self.dataset = VOCDataset(**kwargs)

        super(VOC, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)
