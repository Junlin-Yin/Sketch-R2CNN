from __future__ import division
from __future__ import print_function

import argparse
import cv2
import json
import numpy as np
import os.path
import pickle
import random
import sys
import time
import torch
import tqdm

from pathlib import Path
from torch.utils.data import DataLoader

_project_folder_ = os.path.realpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)

from data.quickdraw_dataset import QuickDrawDataset
from data.sketch_util import SketchUtil
from data.tuberlin_dataset import TUBerlinDataset
from models.modelzoo import CNN_MODELS, CNN_IMAGE_SIZES
from models.sketch_cnn import SketchCNN
from models.sketch_fusion import SketchEarlyFusion, SketchLateFusion
from models.sketch_r2cnn import SketchR2CNN
from neuralline.rasterize import Raster
from scripts.base_train import train_data_collate

DATASETS = {'tuberlin': TUBerlinDataset, 'quickdraw': QuickDrawDataset}

FUSION_MODELS = {'early': SketchEarlyFusion, 'late': SketchLateFusion}

DRAWING_RATIOS = [1, ]
# DRAWING_RATIOS = [0.25, 0.5, 0.75, 1]


def write_row(file, item_list):
    for idx, item in enumerate(item_list):
        if idx == 0:
            file.write(str(item))
        else:
            file.write(',{}'.format(item))
    file.write('\n')


def save_grayscale_sketch(file_path, image):
    inverted_image = 255 - np.array(np.around(image * 255), np.uint8)
    cv2.imwrite(file_path, inverted_image)


def scale_attention(attn, min_attn=None, max_attn=None):
    old_max_attn = np.amax(attn)
    old_min_attn = np.amin(attn)
    normalized_attn = (attn - old_min_attn) / (old_max_attn - old_min_attn)
    if min_attn is not None and max_attn is not None:
        return min_attn + normalized_attn * (max_attn - min_attn)
    else:
        return normalized_attn


def eval_data_collate(batch):
    """
    :param batch: [{'points3': , 'category': }, ...]
    :return:
    """

    assert len(batch) == 1

    res = list()
    for drawing_ratio in DRAWING_RATIOS:
        points3 = np.copy(batch[0]['points3'])
        category = batch[0]['category']

        if drawing_ratio < 1:
            # Construct partial sketches
            strokes = SketchUtil.to_stroke_list(points3)
            # Current drawn strokes according to drawing ratio
            num_drawn_strokes = int(max(round(len(strokes) * drawing_ratio), 1))
            points3 = np.concatenate(strokes[:num_drawn_strokes], axis=0)

        # No padding
        points3_length = len(points3)
        # Compute points3 offset
        points3_offset = np.copy(points3)
        points3_offset[1:points3_length, 0:2] = points3[1:, 0:2] - points3[:points3_length - 1, 0:2]

        # Temporal encoding
        intensities = 1.0 - np.arange(points3_length, dtype=np.float32) / float(points3_length - 1)

        batch_new = {
            'points3': [points3],
            'points3_offset': [points3_offset],
            'points3_length': [points3_length],
            'intensities': [intensities],
            'category': [category]
        }
        batch_collate = dict()
        for k, v in batch_new.items():
            batch_collate[k] = torch.from_numpy(np.array(v))

        # Data in torch format
        res.append((drawing_ratio, batch_collate))
    return res


class BaseEval(object):

    def __init__(self):
        self.config = self._parse_args()

        self.device = torch.device('cuda:{}'.format(self.config['gpu']) if torch.cuda.is_available() else 'cpu')
        print('[*] Using device: {}'.format(self.device))

        self.collect_stats = None
        self.dataset = None
        self.drawing_ratios = DRAWING_RATIOS

    def _parse_args(self):
        """Get arguments from command line"""
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--batch_size', type=int, default=1)
        arg_parser.add_argument('--checkpoint', type=str, default=None)
        arg_parser.add_argument('--ckpt_step_freq', type=int, default=0)
        arg_parser.add_argument('--dataset_fn', type=str, default=None)
        arg_parser.add_argument('--dataset_root', type=str, default=None)
        arg_parser.add_argument('--gpu', type=int, default=0)
        arg_parser.add_argument('--imgsize', type=int, default=224)
        arg_parser.add_argument('--log_dir', type=str, default=None)
        arg_parser.add_argument('--max_ckpt_step', type=int, default=0)
        arg_parser.add_argument('--max_points', type=int, default=321)
        arg_parser.add_argument('--min_ckpt_step', type=int, default=0)
        arg_parser.add_argument('--mode', type=str, default=None)  # ['valid', 'test']
        arg_parser.add_argument('--model_fn', type=str, default=None)
        arg_parser.add_argument('--note', type=str, default=None)
        arg_parser.add_argument('--seed', type=int, default=10)
        arg_parser.add_argument('--thickness', type=float, default=1.0)

        arg_parser = self.add_args(arg_parser)
        config = vars(arg_parser.parse_args())

        # Add some default arguments
        config['imgsize'] = CNN_IMAGE_SIZES[config['model_fn']]
        if config['dataset_fn'] == 'quickdraw':
            config['max_points'] = 321
            config['mode'] = 'test'
        elif config['dataset_fn'] == 'tuberlin':
            config['ckpt_step_freq'] = 20
            config['max_ckpt_step'] = 100
            config['max_points'] = 448
            config['min_ckpt_step'] = 20
            config['mode'] = 'valid'
        else:
            raise Exception('Not valid dataset name!')

        if config['log_dir'] is None:
            raise Exception('No log_dir specified!')
        else:
            if not os.path.exists(config['log_dir']):
                os.makedirs(config['log_dir'], 0o777)

        if config['dataset_root'] is None:
            raise Exception('No dataset_root specified!')

        if config['checkpoint'] is not None:
            if len(config['checkpoint']) < 1:
                config['checkpoint'] = None

        # Reset random seed
        if config['seed'] is None:
            config['seed'] = random.randint(0, 2**31 - 1)
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])

        # Save arguments
        with open(os.path.join(config['log_dir'], 'options.json'), 'w') as fh:
            fh.write(json.dumps(config, sort_keys=True, indent=4))

        return config

    def add_args(self, arg_parser):
        """
        Add additional arguments
        May be implemented in subclass
        :param arg_parser:
        :return:
        """
        return arg_parser

    def checkpoint_prefix(self):
        return self.config['checkpoint']

    def prepare_dataset(self, dataset):
        """
        May be implemented in subclass
        :param dataset:
        :return:
        """
        pass

    def create_model(self, num_categories):
        """
        Should be implemented in subclass
        :param num_categories:
        :return:
        """
        raise NotImplementedError

    def forward_sample(self, model, batch_data, index, drawing_ratio):
        raise NotImplementedError

    def run(self):
        """
        Train loop
        :return:
        """
        batch_size = self.config['batch_size']
        dataset_fn = self.config['dataset_fn']
        dataset_root = self.config['dataset_root']
        log_dir = self.config['log_dir']
        mode = self.config['mode']

        # Load eval data
        self.dataset = DATASETS[dataset_fn](dataset_root, mode, rand_stroke_order=False, rand_point_order=False)
        self.prepare_dataset(self.dataset)
        num_categories = self.dataset.num_categories()

        print('[*] Number of categories:', num_categories)

        # Define model
        net = self.create_model(num_categories)
        net.print_params()
        net.eval_mode()

        # Create a dota loader
        data_loader = DataLoader(self.dataset,
                                 batch_size=batch_size,
                                 num_workers=3,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=eval_data_collate,
                                 pin_memory=True)

        # ============================== Iteration Loop ==================================
        checkpoint = self.checkpoint_prefix()
        if checkpoint is None:
            raise Exception('[!] No valid checkpoint!')
        # Load pretrained models
        loaded_paths = net.load(checkpoint)
        print('[*] Loaded pretrained model from {}'.format(loaded_paths))

        # Running results
        self.collect_stats = list()
        running_corrects = [0] * len(self.drawing_ratios)
        num_samples = [0] * len(self.drawing_ratios)
        running_time = list()
        pbar = tqdm.tqdm(total=len(data_loader))
        for bid, batch_data in enumerate(data_loader):
            for drid in range(len(self.drawing_ratios)):
                dr = batch_data[drid][0]
                batch_data_dr = batch_data[drid][1]

                with torch.set_grad_enabled(False):
                    logits, gt_category, duration = self.forward_sample(net, batch_data_dr, bid, dr)
                    _, predicts = torch.max(logits, 1)

                    predicts_accu = torch.sum(predicts == gt_category)
                    running_corrects[drid] += predicts_accu.item()
                    num_samples[drid] += gt_category.size(0)
                    if dr == 1:
                        running_time.append(duration)
            pbar.update()
        pbar.close()

        accuracies = list()
        for drid, dr in enumerate(self.drawing_ratios):
            accu = float(running_corrects[drid]) / float(num_samples[drid])
            accuracies.append(accu)
            print('[*] {}@{}\t drawing progress: {} / {} are correctly recognized.'.format(
                mode, dr, running_corrects[drid], num_samples[drid]))

        avg_timing = np.mean(np.array(running_time, dtype=np.float32))
        print('[*] {}@1\t Average running time: {}s'.format(mode, avg_timing))
        print('-' * 20)

        self.dataset.dispose()
        return accuracies, self.collect_stats


class SketchCNNEval(BaseEval):

    def add_args(self, arg_parser):
        """
        Add additional arguments
        :param arg_parser:
        :return:
        """
        arg_parser.add_argument('--temporal_encode', dest='temporal_encode', action='store_true')
        arg_parser.set_defaults(temporal_encode=False)
        return arg_parser

    def create_model(self, num_categories):
        """
        :param num_categories:
        :return:
        """
        model_fn = self.config['model_fn']

        # Define model
        model = SketchCNN(CNN_MODELS[model_fn], num_categories, False, False, False)
        model.to(self.device)
        return model

    def forward_sample(self, model, batch_data, index, drawing_ratio):
        imgsize = self.config['imgsize']
        thickness = self.config['thickness']

        points = batch_data['points3'].to(self.device)
        category = batch_data['category'].to(self.device)

        if self.config['temporal_encode']:
            intensities = batch_data['intensities']
        else:
            intensities = 1.0

        start_time = time.time()

        # Rasterization
        sketches_image = Raster.to_image(points, intensities, imgsize, thickness, device=self.device)
        logits = model(sketches_image.repeat(1, 3, 1, 1))

        duration = time.time() - start_time

        # _, predicts = torch.max(logits, 1)
        # if drawing_ratio == 1:
        #    self.collect_stats.append((index, predicts.cpu().numpy()[0], batch_data['category'][0]))

        return logits, category, duration


class SketchR2CNNEval(BaseEval):

    def add_args(self, arg_parser):
        """
        Add additional arguments
        :param arg_parser:
        :return:
        """
        arg_parser.add_argument('--dropout', type=float, default=0)
        arg_parser.add_argument('--intensity_channels', type=int, default=1)
        return arg_parser

    def create_model(self, num_categories):
        """
        :param num_categories:
        :return:
        """
        dropout = self.config['dropout']
        imgsize = self.config['imgsize']
        intensity_channels = self.config['intensity_channels']
        model_fn = self.config['model_fn']
        thickness = self.config['thickness']

        # Define model
        return SketchR2CNN(CNN_MODELS[model_fn],
                           3,
                           dropout,
                           imgsize,
                           thickness,
                           num_categories,
                           intensity_channels=intensity_channels,
                           train_cnn=False,
                           device=self.device)

    def forward_sample(self, model, batch_data, index, drawing_ratio):
        points = batch_data['points3'].to(self.device)
        points_offset = batch_data['points3_offset'].to(self.device)
        points_length = batch_data['points3_length']
        category = batch_data['category'].to(self.device)

        start_time = time.time()
        logits, attention, images = model(points, points_offset, points_length)
        duration = time.time() - start_time

        _, predicts = torch.max(logits, 1)

        attention_cpu = attention.cpu().numpy()
        pred_category = predicts.cpu().numpy()[0]
        if drawing_ratio == 1:
           self.collect_stats.append((index, pred_category, batch_data['category'][0], attention_cpu))

        return logits, category, duration


class SketchFusionEval(SketchR2CNNEval):

    def add_args(self, arg_parser):
        """
        Add additional arguments
        :param arg_parser:
        :return:
        """
        super().add_args(arg_parser)
        arg_parser.add_argument('--fusion', type=str, default='late')  # ['early', 'late']
        return arg_parser

    def create_model(self, num_categories):
        """
        :param num_categories:
        :return:
        """
        cnn_fn = CNN_MODELS[self.config['model_fn']]
        dropout = self.config['dropout']
        fusion_fn = FUSION_MODELS[self.config['fusion']]
        imgsize = self.config['imgsize']
        intensity_channels = self.config['intensity_channels']
        thickness = self.config['thickness']

        if self.config['fusion'] == 'early':
            cnn_fn = None  # Default to ResNet50EarlyFusionBackbone

        return fusion_fn(cnn_fn,
                         3,
                         dropout,
                         imgsize,
                         thickness,
                         num_categories,
                         intensity_channels=intensity_channels,
                         train_cnn=False,
                         device=self.device)

    def forward_sample(self, model, batch_data, index, drawing_ratio):
        points = batch_data['points3'].to(self.device)
        points_offset = batch_data['points3_offset'].to(self.device)
        points_length = batch_data['points3_length']
        category = batch_data['category'].to(self.device)

        start_time = time.time()
        logits, images = model(points, points_offset, points_length)
        duration = time.time() - start_time

        return logits, category, duration
