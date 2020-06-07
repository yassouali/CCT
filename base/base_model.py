import logging
import torch.nn as nn
import numpy as np

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = int(sum([np.prod(p.size()) for p in model_parameters]))
        return f'\nNbr of trainable parameters: {nbr_params}'
        #return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'
