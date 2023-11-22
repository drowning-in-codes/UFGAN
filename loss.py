import torch
from torch import nn
from skimage import metrics
import torch.nn.functional as F
from torchvision import models


def mse_loss(input, target):
    return torch.mean(torch.square(input - target))


def vgg_loss(source_1, source_2, target):
    """
    一般来说，越靠近输入层，越容易抽取图像的细节信息；反之，则越容易抽取图像的全局信息。 为了避免合成图像过多保留内容图像的细节，
    我们选择VGG较靠近输出的层，即内容层，来输出图像的内容特征。 我们还从VGG中选择不同层的输出来匹配局部和全局的风格，这些图层也称为风格层。
    :param source_1:
    :param source_2:
    :param input:
    :param target:
    :return:
    """
    style_layers, content_layers = [0, 5, 10, 19, 28], [25]
    pretrained_net = models.vgg19(pretrained=True)
    net = nn.Sequential(
        *[
            pretrained_net.features[i]
            for i in range(max(content_layers + style_layers) + 1)
        ]
    )
    source_1_content, _ = extract_features(source_1, content_layers, style_layers, net=net)
    _, source_2_styles = extract_features(source_2, content_layers, style_layers, net=net)
    target_content,target_styles = extract_features(target, content_layers, style_layers, net=net)
    target_styles = [gram(Y) for Y in target_styles]
    total_loss = compute_loss(target, source_1_content, source_2_styles, target_content, target_styles)
    return total_loss

def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

def content_loss(Y_hat, Y):
    """
    与线性回归中的损失函数类似，内容损失通过平方误差函数衡量合成图像与内容图像在内容特征上的差异。
    """
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。
    return torch.square(Y_hat - Y.detach()).mean()

def tv_loss(Y_hat):
    """
    我们学到的合成图像里面有大量高频噪点，即有特别亮或者特别暗的颗粒像素。 一种常见的去噪方法是全变分去噪
    """
    return 0.5 * (
            torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean()
            + torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean()
    )


def extract_features(X, content_layers, style_layers, net):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    content_weight, style_weight, tv_weight = 1, 1e3, 10
    # 分别计算内容损失、风格损失和全变分损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return l


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


def l2_norm(input, target):
    return torch.sqrt(torch.sum((input - target) ** 2))
