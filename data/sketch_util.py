from pathlib import Path
from scipy.spatial import ConvexHull
import collections
import cv2
import h5py
import logging
import numpy as np
import math
import os
import random
import re
import skimage.io
import sys
import time


class SketchUtil(object):

    @staticmethod
    def bbox(points):
        """Compute the bbox of one stroke

        :param stroke_pts:
        :return: min, max
        """
        return np.amin(points, axis=0), np.amax(points, axis=0)

    @staticmethod
    def normalization(points, pad_thresh=0.26, eps=1e-4):
        bbox_min, bbox_max = SketchUtil.bbox(points)
        bbox_diag = bbox_max - bbox_min

        if np.dot(bbox_diag, bbox_diag) < eps:
            return None

        bbox_max_side = np.amax(bbox_max - bbox_min)

        # Move to the origin
        mat = SketchUtil.translate_mat(-bbox_min[0], -bbox_min[1])
        # Scaling
        mat = np.matmul(SketchUtil.scale_mat(1.0 / bbox_max_side), mat)
        mat = np.matmul(SketchUtil.scale_mat(1.0 - pad_thresh), mat)

        bbox_max_new = SketchUtil.transform(np.array([bbox_max], dtype='float32'), mat)
        mat = np.matmul(SketchUtil.translate_mat(0.5 - bbox_max_new[0][0] * 0.5, 0.5 - bbox_max_new[0][1] * 0.5), mat)

        points_new = SketchUtil.transform(points, mat)
        points_new *= 2.0
        points_new -= 1.0

        return points_new

    @staticmethod
    def rotate_mat(degree):
        m = np.identity(3, 'float32')
        theta_rad = degree * np.pi / 180.0
        sin_theta = np.sin(theta_rad)
        cos_theta = np.cos(theta_rad)

        m[0, 0] = cos_theta
        m[0, 1] = sin_theta
        m[1, 0] = -sin_theta
        m[1, 1] = cos_theta

        return m

    @staticmethod
    def scale_mat(sx, sy=None):
        if sy is None:
            sy = sx

        m = np.identity(3, 'float32')
        m[0, 0] = sx
        m[1, 1] = sy
        return m

    @staticmethod
    def translate_mat(delta_x, delta_y):
        m = np.identity(3, 'float32')
        m[0, 2] = delta_x
        m[1, 2] = delta_y
        return m

    @staticmethod
    def transform(points, mat):
        """
        :param points: each row is a vertex
        :param mat:
        :return:
        """
        temp_pts = np.ones(shape=(len(points), 3), dtype='float32')
        temp_pts[:, 0:2] = np.array(points, dtype='float32')

        transformed_pts = np.matmul(temp_pts, mat.T)
        return transformed_pts[:, 0:2]

    # =========================== Augmentation ===========================
    @staticmethod
    def convex_hull(points):
        # vertices
        # (ndarray of ints, shape (nvertices,))
        # Indices of points forming the vertices of the convex hull.
        # For 2-D convex hulls, the vertices are in counterclockwise order. For other dimensions, they are in input order.
        hull = ConvexHull(points)
        return points[hull.vertices, :]

    @staticmethod
    def convex_hull_padded(points, thresh=0.3):
        hull = SketchUtil.convex_hull(points)
        hull_center = np.average(hull, axis=0)

        hull_pad = hull + thresh * (hull - hull_center)
        hull_new = ConvexHull(hull_pad)
        return hull_pad[hull_new.vertices, :]

    @staticmethod
    def barycentric_coordinates(cage_coords, points):
        """Compute barycentric coordinates of the points in a stroke
        http://pages.cs.wisc.edu/~csverma/CS777/bary.html
        :param cage_coords: should be in counter clockwise direction, with padding guarantee
        :param points:
        :return:
        """

        n_size = len(cage_coords)
        num_pts = len(points)
        tiled_cage_coords = np.tile(cage_coords, (num_pts, 1))
        tiled_pts = np.repeat(points, n_size, axis=0)

        s = tiled_cage_coords - tiled_pts
        s_norm = np.linalg.norm(s, axis=1)

        i_list = np.arange(n_size)
        ip_list = np.mod(i_list + 1, n_size)
        tiled_offset_list = np.repeat(np.arange(num_pts) * n_size, n_size, axis=0)
        tiled_ip_list = np.tile(ip_list, num_pts) + tiled_offset_list

        Ai_list = 0.5 * (s[:, 0] * s[tiled_ip_list, 1] - s[tiled_ip_list, 0] * s[:, 1])
        Di_list = s[tiled_ip_list, 0] * s[:, 0] + s[tiled_ip_list, 1] * s[:, 1]

        rp_list = s_norm[tiled_ip_list]
        tanalpha = (s_norm * rp_list - Di_list) / (2.0 * Ai_list)

        im_list = np.mod(i_list + n_size - 1, n_size)
        tiled_im_list = np.tile(im_list, num_pts) + tiled_offset_list

        bary_coords = 2.0 * (tanalpha + tanalpha[tiled_im_list]) / s_norm
        bary_coords = np.reshape(bary_coords, (num_pts, n_size))

        bary_coords_sum = np.sum(bary_coords, axis=1, keepdims=True)
        bary_coords_sum[bary_coords_sum < 1e-8] = 1

        bary_coords /= bary_coords_sum

        return bary_coords

    @staticmethod
    def random_cage_deform(cage_coords, points, thresh=0.1):
        """
        Use bary centric coordinates to deform
        :param cage_coords:
        :param points:
        :param thresh:
        :return:
        """
        bbox_min, bbox_max = SketchUtil.bbox(points)

        bary_coords = SketchUtil.barycentric_coordinates(cage_coords, points)

        translate_mag = np.linalg.norm(bbox_max - bbox_min) * thresh
        cage_deformed = cage_coords + np.random.normal(0, 0.5, cage_coords.shape) * translate_mag

        points_deformed = np.matmul(bary_coords, cage_deformed)
        return points_deformed

    @staticmethod
    def random_affine_transform(points, scale_factor=0.2, rot_thresh=30.0):
        bbox_min, bbox_max = SketchUtil.bbox(points)
        bbox_center = (bbox_min + bbox_max) / 2.0

        # x_scale_factor = (np.random.random() - 0.5) * 2 * scale_factor + 1.0
        # y_scale_factor = (np.random.random() - 0.5) * 2 * scale_factor + 1.0
        # Only scale down
        x_scale_factor = 1.0 - np.random.random() * scale_factor
        y_scale_factor = 1.0 - np.random.random() * scale_factor
        rot_degree = (np.random.random() - 0.5) * 2 * rot_thresh

        t_0 = SketchUtil.translate_mat(-bbox_center[0], -bbox_center[1])
        s_1 = SketchUtil.scale_mat(x_scale_factor, y_scale_factor)
        r_2 = SketchUtil.rotate_mat(rot_degree)
        t_3 = SketchUtil.translate_mat(bbox_center[0], bbox_center[1])
        transform = np.matmul(t_3, np.matmul(r_2, np.matmul(s_1, t_0)))

        transformed_points = SketchUtil.transform(points, transform)
        return transformed_points

    @staticmethod
    def random_horizontal_flip(points):
        prob = random.uniform(0, 1)
        if prob > 0.5:
            points_dim0 = (points[:, 0] + 1) / 2.0
            points_copy = np.copy(points)
            points_copy[:, 0] = (1.0 - points_dim0) * 2.0 - 1
            return points_copy
        else:
            return points

    # =========================== Parse SVG ===========================
    @staticmethod
    def parse_svg_path(pathdef, sampling=False):
        """Parse the strings of a path
        https://github.com/regebro/svg.path/blob/master/src/svg/path/parser.py
        :param pathdef:
        :param sampling:
        :return:
        """
        COMMANDS = set('MmZzLlHhVvCcSsQqTtAa')
        UPPERCASE = set('MZLHVCSQTA')
        COMMAND_RE = re.compile('([MmZzLlHhVvCcSsQqTtAa])')
        FLOAT_RE = re.compile('[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?')

        def _tokenize_path(path_str):
            for x in COMMAND_RE.split(path_str):
                if x in COMMANDS:
                    yield x
                for token in FLOAT_RE.findall(x):
                    yield token

        def _sample_cubic_bezier(p0, p1, p2, p3):
            t_list = [0.2, 0.4, 0.6, 0.8, 1.0]
            pt_list = []
            for t in t_list:
                p4 = (1.0 - t)**3 * p0 + 3 * (1.0 - t)**2 * t * p1 + 3 * (1.0 - t) * t**2 * p2 + t**3 * p3
                pt_list.append(p4)
            return pt_list

        def _sample_quadratic_bezier(p0, p1, p2):
            t_list = [0.2, 0.4, 0.6, 0.8, 1.0]
            pt_list = []
            for t in t_list:
                p3 = (1.0 - t)**2 * p0 + 2 * (1.0 - t) * t * p1 + t**2 * p2
                pt_list.append(p3)
            return pt_list

        elements = list(_tokenize_path(pathdef))
        # Reverse for easy use of .pop()
        elements.reverse()

        segments = []
        start_pos = None
        command = None
        current_pos = np.array([0, 0], dtype=np.float32)

        while elements:

            if elements[-1] in COMMANDS:
                # New command.
                last_command = command  # Used by S and T
                command = elements.pop()
                absolute = command in UPPERCASE
                command = command.upper()
            else:
                # If this element starts with numbers, it is an implicit command
                # and we don't change the command. Check that it's allowed:
                if command is None:
                    raise ValueError('Unallowed implicit command in %s, position %s' %
                                     (pathdef, len(pathdef.split()) - len(elements)))
                last_command = command  # Used by S and T

            if command == 'M':
                # Moveto command.
                x = elements.pop()
                y = elements.pop()
                pos = np.array([float(x), float(y)], dtype=np.float32)
                if absolute:
                    current_pos = pos
                else:
                    current_pos += pos

                # when M is called, reset start_pos
                # This behavior of Z is defined in svg spec:
                # http://www.w3.org/TR/SVG/paths.html#PathDataClosePathCommand
                start_pos = current_pos

                # Implicit moveto commands are treated as lineto commands.
                # So we set command to lineto here, in case there are
                # further implicit commands after this moveto.
                command = 'L'

            elif command == 'L':
                x = elements.pop()
                y = elements.pop()
                pos = np.array([float(x), float(y)], dtype=np.float32)
                if not absolute:
                    pos += current_pos

                if not segments:
                    segments.append(current_pos)
                segments.append(pos)
                current_pos = pos

            elif command == 'C':
                control1 = np.array([float(elements.pop()), float(elements.pop())], dtype=np.float32)
                control2 = np.array([float(elements.pop()), float(elements.pop())], dtype=np.float32)
                end = np.array([float(elements.pop()), float(elements.pop())], dtype=np.float32)

                if not absolute:
                    control1 += current_pos
                    control2 += current_pos
                    end += current_pos

                if not segments:
                    segments.append(current_pos)

                if sampling:
                    segments.extend(_sample_cubic_bezier(current_pos, control1, control2, end))
                else:
                    segments.append(end)

                current_pos = end

            elif command == 'Q':
                control = np.array([float(elements.pop()), float(elements.pop())], dtype=np.float32)
                end = np.array([float(elements.pop()), float(elements.pop())], dtype=np.float32)

                if not absolute:
                    control += current_pos
                    end += current_pos

                if not segments:
                    segments.append(current_pos)

                if sampling:
                    segments.extend(_sample_quadratic_bezier(current_pos, control, end))
                else:
                    segments.append(end)

                current_pos = end

        return segments

    @staticmethod
    def parse_tuberlin_svg_file(svg_path_str):
        """
        Read an .svg file in the database
        :param svg_path_str:
        :return: a sketch, in the format of [numpy(2d_point, 2d_point, ..., ), numpy(2d_point, 2d_point, ..., )]
        """
        from xml.dom import minidom

        svg_file_str = Path(svg_path_str).read_text()
        svg_doc = minidom.parseString(svg_file_str)
        path_strs = [[int(p.getAttribute('id')), p.getAttribute('d')] for p in svg_doc.getElementsByTagName('path')]
        svg_doc.unlink()

        path_strokes = []
        for p in path_strs:
            p_strokes = SketchUtil.parse_svg_path(p[1], True)
            if len(p_strokes) > 1:
                path_strokes.append(p)
                path_strokes[-1][1] = np.array(p_strokes)

        sorted_strokes = sorted(path_strokes, key=lambda tup: tup[0])
        res_strokes = [tup[1] for tup in sorted_strokes]

        return res_strokes

    @staticmethod
    def stroke_length(points):
        num_pts = len(points)
        if num_pts < 2:
            return 0
        elif num_pts == 2:
            return np.linalg.norm(points[0, :] - points[1, :])
        else:
            pt_dist = points[1:, :] - points[:-1, :]
            return np.sum(np.linalg.norm(pt_dist, axis=1))

    @staticmethod
    def compute_stroke_orders(strokes, alpha=1.0, beta=2.0, scaling=100.0):
        """
        Compute an ordering by stroke length and temporal order
        Return sorting indices
        """
        num_strokes = len(strokes)
        stroke_lens = [SketchUtil.stroke_length(strokes[idx]) for idx in range(num_strokes)]
        stroke_lens = np.array(stroke_lens, dtype=np.float32) * scaling
        stroke_max_len = np.amax(stroke_lens)
        stroke_lens /= stroke_max_len

        stroke_orders = np.array(np.arange(num_strokes) + 1, dtype=np.float32)
        stroke_orders /= float(num_strokes)

        stroke_drop_prob = np.exp(alpha * stroke_orders) / np.exp(beta * stroke_lens)
        sort_idxs = np.argsort(stroke_drop_prob)
        return sort_idxs.tolist()

    @staticmethod
    def normalize_and_simplify(strokes, max_num_points, eps=1e-3):
        from rdp import rdp

        # First, normalize & pad to range [0, 1]
        stroke_lens = [len(stroke) for stroke in strokes]
        points = SketchUtil.normalization(np.concatenate(strokes))
        if points is None:
            return None
        strokes_norm = np.split(points, np.cumsum(stroke_lens)[:-1], axis=0)

        # Reduce num of points
        if np.sum(stroke_lens) <= max_num_points:
            return strokes_norm

        # Computer an ordering
        stroke_idxs = SketchUtil.compute_stroke_orders(strokes_norm)

        # Use RDP algorithm to simplify
        strokes_rdp = [rdp(stroke, epsilon=eps) for stroke in strokes_norm]

        chosen_idxs = list()
        cnt = 0
        for i in stroke_idxs:
            num_pts = len(strokes_rdp[i])
            if cnt + num_pts < max_num_points:
                chosen_idxs.append(i)
                cnt += num_pts
            else:
                break

        # ** Restore original order **
        strokes_res = [strokes_rdp[i] for i in sorted(chosen_idxs)]
        return strokes_res

    @staticmethod
    def read_image(imgpath):
        """
        :param imgpath:
        :return:
        """
        img = cv2.imread(imgpath)
        if img.ndim == 3:
            # BGR to RGB
            img = img[..., ::-1]
        return img

    @staticmethod
    def save_image(imgpath, img):
        """
        :param imgpath:
        :param img: uint8
        :return:
        """
        if img.ndim == 2:
            cv2.imwrite(imgpath, img)
        else:
            # https://www.scivision.co/numpy-image-bgr-to-rgb/
            # RGB to BGR
            cv2.imwrite(imgpath, img[..., ::-1])

    @staticmethod
    def to_stroke_list(points3):
        """
        :param points3: a sketch point array, (x, y, s)
        :return:
        """
        split_idxes = np.nonzero(points3[:, 2])[0] + 1
        strokes = np.split(points3, split_idxes[:-1], axis=0)
        return strokes

    @staticmethod
    def randomize_point_order(stroke_points3):
        """
        :param stroke_points3: a stroke point array, (x, y, s)
        :return:
        """
        num_points = len(stroke_points3)
        segments = list()
        seg_thresh = 2 * 0.1
        for i in range(num_points - 1):
            # Break into segments
            segment = stroke_points3[i:i + 2, 0:2]
            segment_dir = segment[1, :] - segment[0, :]
            seg_ratio = np.linalg.norm(segment_dir) / seg_thresh
            if seg_ratio > 1:
                steps = np.linspace(0, 1, num=int(math.ceil(seg_ratio)) + 1)
                for j in range(len(steps) - 1):
                    subsegment_start = segment[0, :] + steps[j] * segment_dir
                    subsegment_end = segment[0, :] + steps[j + 1] * segment_dir
                    subsegment = np.array([subsegment_start.tolist() + [0], subsegment_end.tolist() + [1]], dtype=np.float32)
                    segments.append(subsegment)
            else:
                subsegment = np.array([[segment[0, 0], segment[0, 1], 0], [segment[1, 0], segment[1, 1], 1]], dtype=np.float32)
                segments.append(subsegment)
        if len(segments) == 0:
            return stroke_points3
        else:
            # Randomize the order
            random.shuffle(segments)
            return np.concatenate(segments, axis=0)
