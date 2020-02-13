from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pretrainedmodels


class CNNBackbone(torch.nn.Module):

    def __init__(self, pretrained=True, requires_grad=True, in_channels=3):
        """
        :param pretrained:
        :param requires_grad:
        :param in_channels:
        """
        super().__init__()

        self.pretrained = pretrained
        self.requires_grad = requires_grad
        self.in_channels = in_channels

        self.num_out_features = self._init()

        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def _init(self):
        raise NotImplementedError

    def forward(self, *x):
        raise NotImplementedError


# =======================================================================


class Vgg16Backbone(CNNBackbone):
    """
    https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
    """

    def _init(self):
        self.features = torchvision.models.vgg16(pretrained=self.pretrained).features

        num_out_features = 512 * 7 * 7
        return num_out_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class DenseNet161Backbone(CNNBackbone):

    def _init(self):
        cnn = torchvision.models.densenet161(pretrained=self.pretrained)

        if self.in_channels in [1, 3]:
            # Normal version
            self.features = cnn.features  # A nn.Sequential module
            print('[*] DenseNet161Backbone: use pretrained conv0 with {} input channels'.format(
                self.features.conv0.in_channels))
        else:
            # Replace conv0
            from collections import OrderedDict
            print('[*] DenseNet161Backbone: cnn.features -', cnn.features.__class__.__name__)
            module_dict = OrderedDict()
            for name, module in cnn.features.named_children():
                if name == 'conv0':
                    module_dict[name + '_new'] = nn.Conv2d(self.in_channels,
                                                           module.out_channels,
                                                           kernel_size=module.kernel_size,
                                                           stride=module.stride,
                                                           padding=module.padding,
                                                           bias=False)
                else:
                    module_dict[name] = module
            self.features = nn.Sequential(module_dict)
            print('[*] DenseNet161Backbone: use a new conv0 with {} input channels'.format(self.in_channels))

        num_out_features = cnn.classifier.in_features
        return num_out_features

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        return out


class ResNeXt101Backbone(CNNBackbone):

    def _init(self):
        if self.pretrained:
            cnn = pretrainedmodels.__dict__['resnext101_32x4d'](pretrained='imagenet')
        else:
            cnn = pretrainedmodels.__dict__['resnext101_32x4d'](pretrained=None)
        self.features = cnn.features
        self.avg_pool = cnn.avg_pool

        num_out_features = 2048
        return num_out_features

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


class ResNet101Backbone(CNNBackbone):

    def _init(self):
        cnn = torchvision.models.resnet101(pretrained=self.pretrained)

        if self.in_channels in [1, 3]:
            # Normal version
            self.conv1 = cnn.conv1
            self.conv1_new = None
            print('[*] ResNet101Backbone: use pretrained conv1 with {} input channels'.format(cnn.conv1.in_channels))
        else:
            self.conv1_new = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1 = None
            print('[*] ResNet101Backbone: use a new conv1 with {} input channels'.format(self.in_channels))
        self.bn1 = cnn.bn1
        self.relu = cnn.relu
        self.maxpool = cnn.maxpool
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4
        self.avgpool = cnn.avgpool

        num_out_features = 512 * 4
        return num_out_features

    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)
        else:
            x = self.conv1_new(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class ResNet50Backbone(CNNBackbone):
    """
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def _init(self):
        cnn = torchvision.models.resnet50(pretrained=self.pretrained)
        if self.in_channels in [1, 3]:
            # Normal version
            self.conv1 = cnn.conv1
            self.conv1_new = None
            print('[*] ResNet50Backbone: use pretrained conv1 with {} input channels'.format(cnn.conv1.in_channels))
        else:
            self.conv1_new = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1 = None
            print('[*] ResNet50Backbone: use a new conv1 with {} input channels'.format(self.in_channels))
        self.bn1 = cnn.bn1
        self.relu = cnn.relu
        self.maxpool = cnn.maxpool
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4
        self.avgpool = cnn.avgpool

        num_out_features = 512 * 4
        return num_out_features

    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)
        else:
            x = self.conv1_new(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten
        x = x.view(x.size(0), -1)

        return x


class AttentiveResNet50Backbone(CNNBackbone):
    """
    Attentive-ResNet50
    Ref:
    Deep Spatial-Semantic Attention for Fine-Grained Sketch-Based Image Retrieval
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def _init(self):
        cnn = torchvision.models.resnet50(pretrained=self.pretrained)
        self.conv1 = cnn.conv1
        self.bn1 = cnn.bn1
        self.relu = cnn.relu
        self.maxpool = cnn.maxpool
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4
        self.avgpool = cnn.avgpool

        print('[*] RestNet50: Add attention module.')
        self.attn_conv1 = nn.Conv2d(256, 256, 1)
        self.attn_conv2 = nn.Conv2d(256, 1, 1)

        num_out_features = 512 * 4
        return num_out_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # Attention module
        # https://github.com/yuchuochuo1023/Deep_SBIR_tf/blob/master/triplet_sbir_train.py
        attn = self.attn_conv1(x)
        attn = F.relu(attn)
        attn = self.attn_conv2(attn)
        # Spatial softmax
        N, C, H, W = attn.size(0), attn.size(1), attn.size(2), attn.size(3)
        attn = F.softmax(attn.view(-1, H * W), dim=1)
        attn = attn.view(N, C, H, W)
        x = x + torch.mul(x, attn)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten
        x = x.view(x.size(0), -1)

        return x


class ResNet50EarlyFusionBackbone(CNNBackbone):
    """
    For two-branch early fusion
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def _init(self):
        cnn = torchvision.models.resnet50(pretrained=self.pretrained)
        self.conv1 = cnn.conv1
        self.bn1 = cnn.bn1
        self.relu = cnn.relu
        self.maxpool = cnn.maxpool
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4
        self.avgpool = cnn.avgpool

        print('[*] RestNet50: Fuse attention from RNN.')
        num_out_features = 512 * 4
        return num_out_features

    def forward(self, x, attn):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # (None, 256, 56, 56)
        # Inject attention
        # print('x:', x.size(), 'attn:', attn.size())
        x = torch.mul(x, attn)
        x = self.layer2(x)  # (None, 512, 28, 28)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten
        x = x.view(x.size(0), -1)

        return x


class SketchANetBackbone(CNNBackbone):
    """
    Sketch-a-Net: A Deep Neural Network that Beats Humans
    """

    def _init(self):
        if self.in_channels in [1, 3]:
            # Normal version
            self.conv1 = nn.Conv2d(3, 64, 15, stride=3, padding=0)
            self.conv1_new = None
            print('[*] SketchANetBackbone: use conv1 with 3 input channels')
        else:
            self.conv1_new = nn.Conv2d(self.in_channels, 64, 15, stride=3, padding=0)
            self.conv1 = None
            print('[*] SketchANetBackbone: use a new conv1 with {} input channels'.format(self.in_channels))
        self.conv2 = nn.Conv2d(64, 128, 5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 256, 512)
        self.fc2 = nn.Linear(512, 512)

        num_out_features = 512
        return num_out_features

    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)
        else:
            x = self.conv1_new(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x


class AttentiveSketchANetBackbone(SketchANetBackbone):
    """
    Attentive-SketchANet
    Ref:
    Deep Spatial-Semantic Attention for Fine-Grained Sketch-Based Image Retrieval
    https://github.com/yuchuochuo1023/Deep_SBIR_tf/blob/master/triplet_sbir_train.py
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def _init(self):
        if self.in_channels in [1, 3]:
            # Normal version
            self.conv1 = nn.Conv2d(3, 64, 15, stride=3, padding=0)
            self.conv1_new = None
            print('[*] AttentiveSketchANetBackbone: use conv1 with 3 input channels')
        else:
            self.conv1_new = nn.Conv2d(self.in_channels, 64, 15, stride=3, padding=0)
            self.conv1 = None
            print('[*] AttentiveSketchANetBackbone: use a new conv1 with {} input channels'.format(self.in_channels))
        self.conv2 = nn.Conv2d(64, 128, 5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        print('[*] AttentiveSketchANetBackbone: Add attention module.')
        self.attn_conv1 = nn.Conv2d(256, 256, 1)
        self.attn_conv2 = nn.Conv2d(256, 1, 1)

        self.fc1 = nn.Linear(7 * 7 * 256, 512)
        self.fc2 = nn.Linear(512, 512)

        num_out_features = 512
        return num_out_features

    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)
        else:
            x = self.conv1_new(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        # Attention module
        attn = self.attn_conv1(x)
        attn = F.relu(attn)
        attn = self.attn_conv2(attn)
        # Spatial softmax
        N, C, H, W = attn.size(0), attn.size(1), attn.size(2), attn.size(3)
        attn = F.softmax(attn.view(-1, H * W), dim=1)
        attn = attn.view(N, C, H, W)
        attn = torch.mul(x, attn)

        x = attn.view(attn.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x


class AttentiveSketchANetBackboneV1(SketchANetBackbone):
    """
    Attentive-SketchANet
    Ref:
    Deep Spatial-Semantic Attention for Fine-Grained Sketch-Based Image Retrieval
    https://github.com/yuchuochuo1023/Deep_SBIR_tf/blob/master/triplet_sbir_train.py
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def _init(self):
        if self.in_channels in [1, 3]:
            # Normal version
            self.conv1 = nn.Conv2d(3, 64, 15, stride=3, padding=0)
            self.conv1_new = None
            print('[*] AttentiveSketchANetBackbone: use conv1 with 3 input channels')
        else:
            self.conv1_new = nn.Conv2d(self.in_channels, 64, 15, stride=3, padding=0)
            self.conv1 = None
            print('[*] AttentiveSketchANetBackbone: use a new conv1 with {} input channels'.format(self.in_channels))
        self.conv2 = nn.Conv2d(64, 128, 5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        print('[*] AttentiveSketchANetBackbone: Add attention module.')
        self.attn_conv1 = nn.Conv2d(256, 256, 1)
        self.attn_conv2 = nn.Conv2d(256, 1, 1)

        self.fc1 = nn.Linear(7 * 7 * 256, 512)
        self.fc2 = nn.Linear(512, 256)

        num_out_features = 256 + 256
        return num_out_features

    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)
        else:
            x = self.conv1_new(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        # Attention module
        attn = self.attn_conv1(x)
        attn = F.relu(attn)
        attn = self.attn_conv2(attn)
        # Spatial softmax
        N, C, H, W = attn.size(0), attn.size(1), attn.size(2), attn.size(3)
        attn = F.softmax(attn.view(-1, H * W), dim=1)
        attn = attn.view(N, C, H, W)
        attn = x + torch.mul(x, attn)
        # Coarse to fine
        c2f = torch.sum(attn, dim=(2, 3))
        c2f = F.normalize(c2f, dim=1)

        x = attn.view(attn.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = F.normalize(x, dim=1)

        x = torch.cat((x, c2f), dim=1)
        return x


CNN_MODELS = {
    'densenet161': DenseNet161Backbone,
    'resnext101': ResNeXt101Backbone,
    'resnet101': ResNet101Backbone,
    'resnet50': ResNet50Backbone,
    'resnet50attn': AttentiveResNet50Backbone,
    'sketchanet': SketchANetBackbone,
    'sketchanetattn': AttentiveSketchANetBackbone,
}

CNN_IMAGE_SIZES = {
    'densenet161': 224,
    'resnext101': 224,
    'resnet101': 224,
    'resnet50': 224,
    'resnet50attn': 224,
    'sketchanet': 225,
    'sketchanetattn': 225,
}
