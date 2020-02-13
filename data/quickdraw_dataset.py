from __future__ import division
from __future__ import print_function

from .sketch_util import SketchUtil
from torch.utils.data import Dataset
import h5py
import numpy as np
import os.path as osp
import pickle
import random


class QuickDrawDataset(Dataset):
    """
    The QuickDraw dataset
    """
    mode_indices = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, root_dir, mode, rand_stroke_order=False, rand_point_order=False):
        """
        :param root_dir:
        :param mode: ['train', 'valid', 'test']
        :param rand_stroke_order: For ablation study
        :param rand_point_order:
        """
        self.root_dir = root_dir
        self.mode = mode
        self.rand_stroke_order = rand_stroke_order
        self.rand_point_order = rand_point_order

        self.data = None  # Lazy initialization

        with open(osp.join(root_dir, 'categories.pkl'), 'rb') as fh:
            saved_pkl = pickle.load(fh)
            self.categories = saved_pkl['categories']
            self.indices = saved_pkl['indices'][self.mode_indices[mode]]

        print('[*] Created a new {} dataset: {}; use random stroke order: {}, point order: {}'.format(
            mode, root_dir, rand_stroke_order, rand_point_order))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.data is None:
            self.data = h5py.File(osp.join(self.root_dir, 'quickdraw_{}.hdf5'.format(self.mode)), 'r')

        # category_id, sketch_id
        index_tuple = self.indices[idx]
        cid = index_tuple[0]
        sid = index_tuple[1]
        sketch_path = '/sketch/{}/{}'.format(cid, sid)

        # (x, y, s)
        sid_points = np.array(self.data[sketch_path][()], dtype=np.float32)

        # Randomize stroke order for ablation study
        if self.rand_stroke_order:
            strokes = SketchUtil.to_stroke_list(sid_points)
            stroke_idxes = list(range(len(strokes)))
            random.shuffle(stroke_idxes)
            # (x, y, s)
            sid_points = np.concatenate([strokes[idx] for idx in stroke_idxes], axis=0)

        # Randomize point order in each stroke for ablation study
        if self.rand_point_order:
            strokes = SketchUtil.to_stroke_list(sid_points)
            # Break into segments, randomize, and assemble back to points3 (i.e., many sub-strokes)
            substrokes = [SketchUtil.randomize_point_order(stroke) for stroke in strokes if len(stroke) > 1]
            sid_points = np.concatenate(substrokes, axis=0)

        sample = {'points3': sid_points, 'category': cid}
        return sample

    def __del__(self):
        self.dispose()

    def dispose(self):
        if self.data is not None:
            self.data.close()

    def num_categories(self):
        return len(self.categories)

    def get_name_prefix(self):
        return 'QuickDraw-{}'.format(self.mode)
