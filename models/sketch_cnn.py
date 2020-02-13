from __future__ import division
from __future__ import print_function
from .basemodel import BaseModel
import torch
import torch.nn as nn


class SketchCNN(BaseModel):

    def __init__(self, model_fn, num_categories, pretrained_conv_layers=False, train_conv_layers=True, train_fc_layer=True):
        super().__init__()

        self.conv = model_fn(pretrained=pretrained_conv_layers, requires_grad=train_conv_layers)
        self.fc = nn.Linear(self.conv.num_out_features, num_categories)
        if not train_fc_layer:
            for param in self.fc.parameters():
                param.requires_grad = False

        self.register_nets([self.conv, self.fc], ['conv', 'fc'], [train_conv_layers, train_fc_layer])

    def __call__(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
