from __future__ import division
from __future__ import print_function

from .sketch_util import SketchUtil
from torch.utils.data import Dataset
import numpy as np
import pickle
import random


class TUBerlinDataset(Dataset):

    def __init__(self, pkl_file, mode, rand_stroke_order=False, rand_point_order=False, drop_strokes=True):
        """
        :param pkl_file:
        :param mode: ['train', 'valid']
        :param rand_stroke_order:
        :param rand_point_order:
        :param drop_strokes:
        """
        self.pkl_file = pkl_file
        self.mode = mode
        self.rand_stroke_order = rand_stroke_order
        self.rand_point_order = rand_point_order
        self.drop_strokes = drop_strokes

        with open(self.pkl_file, 'rb') as fh:
            saved = pickle.load(fh)
            self.categories = saved['categories']
            self.sketches = saved['sketches']
            self.cvxhulls = saved['convex_hulls']
            self.folds = saved['folds']

        self.fold_idx = None
        self.indices = list()

    def set_fold(self, idx):
        self.fold_idx = idx
        self.indices = list()

        if self.mode == 'train':
            for i in range(len(self.folds)):
                if i != idx:
                    self.indices.extend(self.folds[i])
        else:
            self.indices = self.folds[idx]

        print(
            '[*] Created a new {} dataset with {} fold as validation data; use random stroke order: {}, point order: {}'.format(
                self.mode, idx, self.rand_stroke_order, self.rand_point_order))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cid, sid = self.indices[idx]

        # (x, y, s)
        sid_points = np.copy(self.sketches[cid][sid])

        if self.mode == 'train':
            # Apply deformation augmentation
            cvxhull = self.cvxhulls[cid][sid]
            pts_xy = sid_points[:, 0:2]
            if cvxhull is not None:
                if random.uniform(0, 1) > 0.5:
                    pts_xy = SketchUtil.random_cage_deform(np.copy(cvxhull), pts_xy, thresh=0.1)
                    pts_xy = SketchUtil.normalization(pts_xy)
                if random.uniform(0, 1) > 0.5:
                    pts_xy = SketchUtil.random_affine_transform(pts_xy, scale_factor=0.2, rot_thresh=40.0)
            pts_xy = SketchUtil.random_horizontal_flip(pts_xy)
            sid_points[:, 0:2] = pts_xy
            # Drop strokes
            if self.drop_strokes:
                sid_points = self._random_drop_strokes(sid_points)

        # Randomize stroke order for ablation study
        # *Should be done after Cvx Deformation*
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
            substrokes = [SketchUtil.randomize_point_order(stroke) for stroke in strokes]
            sid_points = np.concatenate(substrokes, axis=0)

        sample = {'points3': sid_points, 'category': cid}
        return sample

    def _random_drop_strokes(self, points3):
        # Convert point3 to stroke list
        strokes = SketchUtil.to_stroke_list(points3)
        num_strokes = len(strokes)

        if num_strokes < 2:
            return points3

        # Randomly drop strokes
        sort_idxes = SketchUtil.compute_stroke_orders([s[:, 0:2] for s in strokes])

        keep_prob = np.random.uniform(0, 1, num_strokes)
        keep_prob[:(num_strokes // 2)] = 1

        keep_idxes = np.array(sort_idxes, np.int32)[keep_prob > 0.5]
        # Still need to keep orginal stroke order
        keep_strokes = [strokes[i] for i in sorted(keep_idxes.tolist())]
        # np.random.shuffle(keep_idxes)
        # keep_strokes = [strokes[i] for i in keep_idxes.tolist()]
        return np.concatenate(keep_strokes, axis=0)

    def num_categories(self):
        return len(self.categories)

    def dispose(self):
        pass

    def get_name_prefix(self):
        return 'TUBerlin-{}-{}'.format(self.mode, self.fold_idx)
