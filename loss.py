import torch
from torch import nn
from skimage import metrics
import torch.nn.functional as F


def mse_loss(input, target):
    return torch.mean(torch.square(input - target))


def ssim_loss(input, target):
    return 1 - metrics.structural_similarity(input, target, data_range=1, multichannel=True)


def gradient(img, ):
    # Laplacian 算子
    # [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]), [3, 3, 1, 1]]
    output_img = F.conv2d(img, weight=torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0).unsqueeze(0), padding=1)
    return output_img


def gradient_loss(input, target):
    return torch.mean((gradient(input) - gradient(target)) ** 2)

def l2_norm(input,target):
    return torch.sqrt(torch.sum((input - target) ** 2))