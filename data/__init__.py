import torch
import numpy as np


def default_collate(batch):
    """
    :param batch: {'key1': [tensor1, ...], ...}
    :return:
    """
    if isinstance(batch, dict):
        batch_collate = dict()
        for k, v in batch.items():
            # np.stack works for a list of a single tensor
            batch_collate[k] = torch.from_numpy(np.stack(v, axis=0))
    else:
        batch_collate = torch.from_numpy(np.stack(batch, axis=0))

    return batch_collate