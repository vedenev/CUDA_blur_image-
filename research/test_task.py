import math

import torch
from torch import nn
import numpy as np
import cv2


def run(image: np.ndarray, kernel_size: int, sigma: float):
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    kernel = kernel / torch.sum(kernel)

    channels = image.shape[2]

    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)

    f = nn.Conv2d(in_channels=channels, out_channels=channels,
                  kernel_size=kernel_size, groups=channels, bias=False)

    f.weight.data = kernel
    f.weight.requires_grad = False
    f.cuda()

    return f(torch.from_numpy(np.expand_dims(np.swapaxes(image.astype(np.float32), 0, -1), axis=0)).cuda()).cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread(r'img.jpg')

    res = run(img, 15, 3)
    cv2.imshow('img', np.swapaxes(np.squeeze(res.astype(np.uint8)), 0, -1))
    cv2.waitKey()
