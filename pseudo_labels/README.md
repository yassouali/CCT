

## Generating Pseudo-Labels

This is a 3rd party code, which was adapted for our case, we thank the original authors for 
providing the implementation for their work, please check it out if you are interested:
* Paper: [Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations](https://arxiv.org/abs/1904.05044)
* Code: [Jiwoon Ahn's irn](https://github.com/jiwoon-ahn/irn)

This code is used for generating pseudo pixel-level from class labels. This is done in three steps:

* `train_cam.py`: first we fine-tune a pretrained resnet50 (on imagenet from torchvision) on Pascal Voc for image classification
with 21 classes. In this case, for fast training, the batch norm layers are frozen, and we only use high learning rate for the last classification
layer after an average pool.
* `make_cam.py`: Using the pretrained resnet on Pascal Voc, we follows the traditional
([paper](https://arxiv.org/pdf/1512.04150.pdf)) approach to generate localization maps, this is done
by simply weighting the activations of the last block of resnet by the learned weight of the classification weight.
We then only consider the maps of the ground-truth classes.
* `cam_to_pseudo_labels.py`: The last step is a refinement step to only consider the highly confident regions, and the non-confident regions
are ignored. A CRF refinement step is also applied before saving the pseudo-labels.



To generate the pseudo-labels, simply run:

```bash
python run.py --voc12_root DATA_PATH
```

`DATA_PATH` must point to the folder containing `JPEGImages` in Pascal Voc dataset.

The results will be saved in `result/pseudo_labels` as PNG files, which will be used to train the auxiliary decoders of CCT
in weakly semi-supervised setting.

If you find this code useful, please consider citing the original [paper]((https://arxiv.org/abs/1904.05044)).