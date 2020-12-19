import numpy as np

import torch

class CutMix:
    #copy from: https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py

    def __init__(self, cutmix_param):
    
        self.cutmix_param = cutmix_param
        


    def _create_rand_box(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def generate_cutmix_images(self, input):

        r = np.random.rand(1)
        if self.cutmix_param["beta"] > 0 and r < self.cutmix_param["cutmix_prob"]:

            # generate mixed sample
            lam = np.random.beta(self.cutmix_param["beta"], self.cutmix_param["beta"])
            rand_index = torch.randperm(input.size()[0])
            #target_a = target
            #target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = self._create_rand_box(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            #lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

        return input
