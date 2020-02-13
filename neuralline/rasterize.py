from __future__ import division
from __future__ import print_function

from .jit import rasterize_cuda
from torch.autograd import Function
import numpy as np
import torch

DEFAULT_INTENSITY = 1.0
DEFAULT_IMG_SIZE = 256
DEFAULT_THICKNESS = 1.0
DEFAULT_EPS = 1e-4
DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RasterIntensityFunc(Function):

    @staticmethod
    def forward(ctx,
                lines,
                intensities,
                img_size=DEFAULT_IMG_SIZE,
                thickness=DEFAULT_THICKNESS,
                eps=DEFAULT_EPS,
                device=DEFAULT_DEVICE):
        """
        :param ctx:
        :param lines: [batch_size, num_lines, 3]
        :param intensities: [batch_size, num_lines, channels]
        :param img_size:
        :param thickness:
        :param eps:
        :param device:
        :return:
        """

        batch_size = lines.size(0)
        intensity_channels = intensities.size(2)

        line_map = torch.zeros(batch_size, intensity_channels, img_size, img_size, dtype=torch.float32, device=device)
        line_index_map = -1 * torch.ones(batch_size, 1, img_size, img_size, dtype=torch.int32, device=device)
        line_weight_map = torch.zeros(batch_size, 1, img_size, img_size, dtype=torch.float32, device=device)
        locks = torch.zeros_like(line_index_map)

        rasterize_cuda.rasterize_forward(lines.contiguous(), intensities.contiguous(), line_map, line_index_map,
                                         line_weight_map, locks, img_size, thickness, eps)

        # Save for backward
        save_vars = [line_map, line_index_map, line_weight_map, lines, intensities]
        ctx.save_for_backward(*save_vars)
        ctx.img_size = img_size
        ctx.thickness = thickness
        ctx.eps = eps
        ctx.device = device

        return line_map

    @staticmethod
    def backward(ctx, grad_line_map):
        """
        :param ctx:
        :param grad_line_map:
        :return:
        """
        grad_intensities = torch.zeros_like(ctx.saved_tensors[4])

        rasterize_cuda.rasterize_backward(
            grad_intensities,
            grad_line_map.contiguous(),
            ctx.saved_tensors[0],  # line_map
            ctx.saved_tensors[1],  # line_index_map
            ctx.saved_tensors[2],  # line_weight_map
            ctx.saved_tensors[3].contiguous(),  # lines
            ctx.saved_tensors[4].contiguous(),  # intensities
            ctx.img_size,
            ctx.thickness,
            ctx.eps)
        return None, grad_intensities, None, None, None, None


class Raster(object):

    @staticmethod
    def to_image(lines, intensities, img_size, thickness, eps=DEFAULT_EPS, device=DEFAULT_DEVICE):
        if type(lines).__module__ == np.__name__:
            lines_gpu = torch.from_numpy(lines).to(device)
        else:
            # Torch tensor
            lines_gpu = lines.to(device)
        batch_size = lines_gpu.size(0)
        num_lines = lines_gpu.size(1)

        if isinstance(intensities, float):
            intensities_gpu = intensities * torch.ones(batch_size, num_lines, 1, dtype=torch.float32, device=device)
        else:
            if type(intensities).__module__ == np.__name__:
                intensities_gpu = torch.from_numpy(intensities).to(device)
            else:
                intensities_gpu = intensities.to(device)
            if intensities_gpu.dim() == 2:  # (batch_size, num_lines)
                intensities_gpu = torch.unsqueeze(intensities_gpu, dim=2)

        intensity_channels = intensities_gpu.size(2)

        line_map_gpu = torch.zeros(batch_size, intensity_channels, img_size, img_size, dtype=torch.float32, device=device)
        line_index_map_gpu = -1 * torch.ones(batch_size, 1, img_size, img_size, dtype=torch.int32, device=device)
        line_weight_map_gpu = torch.zeros(batch_size, 1, img_size, img_size, dtype=torch.float32, device=device)
        locks_gpu = torch.zeros_like(line_index_map_gpu)

        rasterize_cuda.rasterize_forward(lines_gpu.contiguous(), intensities_gpu, line_map_gpu, line_index_map_gpu,
                                         line_weight_map_gpu, locks_gpu, img_size, thickness, eps)
        return line_map_gpu
