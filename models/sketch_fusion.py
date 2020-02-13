from __future__ import division
from __future__ import print_function
from .basemodel import BaseModel
from torch.nn.utils.rnn import pack_padded_sequence
import os.path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_project_folder_ = os.path.realpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)

from models.modelzoo import ResNet50EarlyFusionBackbone
from models.sketch_r2cnn import SeqEncoder
from neuralline.rasterize import RasterIntensityFunc, Raster


class SketchLateFusion(BaseModel):
    """
    Similar to SketchMate
    """

    def __init__(self,
                 cnn_fn,
                 rnn_input_size,
                 rnn_dropout,
                 img_size,
                 thickness,
                 num_categories,
                 intensity_channels=1,
                 rnn_hidden_size=512,
                 rnn_num_layers=2,
                 rnn_batch_first=True,
                 rnn_bidirect=True,
                 train_cnn=True,
                 device=None):
        super().__init__()

        self.img_size = img_size
        self.thickness = thickness
        self.intensity_channels = intensity_channels
        self.rnn_batch_first = rnn_batch_first
        self.eps = 1e-4
        self.device = device

        # CNN for 2D analysis
        self.cnn = cnn_fn(pretrained=True, requires_grad=train_cnn)

        # RNN for analyzing vector sketches
        self.rnn = nn.LSTM(input_size=rnn_input_size,
                           hidden_size=rnn_hidden_size,
                           num_layers=rnn_num_layers,
                           batch_first=rnn_batch_first,
                           bidirectional=rnn_bidirect,
                           dropout=rnn_dropout)

        num_directs = 2 if rnn_bidirect else 1
        self.fc1 = nn.Linear(num_directs * rnn_num_layers * rnn_hidden_size + self.cnn.num_out_features, 2048)
        self.fc2 = nn.Linear(2048, num_categories)

        nets = list()
        names = list()
        train_flags = list()

        nets.extend([self.cnn, self.rnn, self.fc1, self.fc2])
        names.extend(['conv', 'rnn', 'fc1', 'fc2'])
        train_flags.extend([True] * 4)

        self.register_nets(nets, names, train_flags)
        self.to(device)

    def __call__(self, points, points_offset, lengths):
        # === RNN ===
        batch_size = points_offset.shape[0]  # [batch_size, num_points, 3]

        # Pack
        points_offset_packed = pack_padded_sequence(points_offset, lengths, batch_first=self.rnn_batch_first)
        hiddens_packed, (last_hidden, _) = self.rnn(points_offset_packed)
        last_hidden = last_hidden.view(batch_size, -1)

        # === CNN ===
        images = Raster.to_image(points, 1.0, self.img_size, self.thickness, device=self.device)
        cnnfeat = self.cnn(images.repeat(1, 3, 1, 1))

        # === FC ===
        logits = self.fc2(F.relu(self.fc1(torch.cat((last_hidden, cnnfeat), 1))))

        return logits, images


# =======================================================================
# A simple variant of our Sketch-R2CNN


class SketchEarlyFusion(BaseModel):

    def __init__(self,
                 cnn_fn,
                 rnn_input_size,
                 rnn_dropout,
                 img_size,
                 thickness,
                 num_categories,
                 intensity_channels=1,
                 train_cnn=True,
                 device=None):
        super().__init__()

        self.img_size = img_size
        self.thickness = thickness
        # Should be 1 or 256
        self.intensity_channels = intensity_channels
        self.eps = 1e-4
        self.device = device

        nets = list()
        names = list()
        train_flags = list()

        # RNN for analyzing stroke-based sketches
        self.rnn = SeqEncoder(rnn_input_size, out_channels=intensity_channels, dropout=rnn_dropout)

        # CNN for 2D analysis
        self.cnn = ResNet50EarlyFusionBackbone(pretrained=False, requires_grad=train_cnn)

        # Last layer for classification
        num_fc_in_features = self.cnn.num_out_features
        self.fc = nn.Linear(num_fc_in_features, num_categories)

        nets.extend([self.rnn, self.cnn, self.fc])
        names.extend(['rnn', 'conv', 'fc'])
        train_flags.extend([True, train_cnn, True])

        self.register_nets(nets, names, train_flags)
        self.to(device)

    def __call__(self, points, points_offset, lengths):
        # === RNN ===
        # Compute point-wise attention
        intensities, seqfeat = self.rnn(points_offset, lengths)
        # Rasterization and inject into CNN after stage 2
        attention = RasterIntensityFunc.apply(points, intensities, 56, 0.5, self.eps, self.device)
        # === CNN ===
        images = Raster.to_image(points, 1.0, self.img_size, self.thickness, device=self.device)
        cnnfeat = self.cnn(images.repeat(1, 3, 1, 1), attention)
        logits = self.fc(cnnfeat)

        return logits, (images, attention)
