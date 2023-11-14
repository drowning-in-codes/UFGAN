from pathlib import Path
import torch
import numpy as np
import argparse
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import h5py
import tqdm

from logger import getLogger


class imgDataset(Dataset):
    global args
    global mylogger

    def __init__(self, is_train=True, transform=True, path="./Train_ir"):
        super(imgDataset, self).__init__()
        self.checkpoint_path = None
        self.is_train = is_train
        self.transform = transform
        if self.transform:
            self.trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        if self.is_train:
            self.patch_img(path)
            with h5py.File(self.checkpoint_path, 'r') as hf:
                self.img = np.array(hf.get('data'))
                self.label = np.array(hf.get('label'))
        else:
            self.img_path = path
            self.total_img = list(Path(path).glob("*.bmp"))
            self.total_img.extend(Path(path).glob("*.jpg"))
            self.total_img.extend(Path(path).glob("*.png"))
            self.total_img.extend(Path(path).glob("*.tif"))
            self.total_img.sort(key=lambda x: int(x.stem))
            # with h5py.File(str(self.checkpoint_path), "w") as hf:
            #     hf.create_dataset('data', data=sub_img)
            #     hf.create_dataset('label', data=sub_label)

    def __len__(self):
        return len(self.total_img)

    def __getitem__(self, idx):
        if self.is_train:
            img = self.img[idx]
            label = self.label[idx]
            if self.transform:
                img = self.trans(img)
                label = self.trans(label)
            return img, label
        else:
            img = cv2.imread(str(Path(self.img_path).joinpath(self.total_img[idx].name)), cv2.IMREAD_GRAYSCALE)

            padding = 6
            img = np.pad(img, ((padding // 2, padding - padding // 2), (padding // 2, padding - padding // 2)),
                         'constant', constant_values=(127, 127))
            [h, w] = img.shape
            img = img.reshape([h, w, 1])
            label = cv2.imread(str(Path(self.img_path).joinpath(self.total_img[idx].name)), cv2.IMREAD_GRAYSCALE)
            if self.transform:
                img = self.trans(img)
                label = self.trans(label)

        return img, label

    def patch_img(self, img_path):
        mylogger.info(f"训练|开始切分训练集")
        if not Path(f"{img_path}_patch").exists():
            Path(f"{img_path}_patch").mkdir(exist_ok=False)
            mylogger.info(f"创建{img_path}_patch成功")

        total_img = list(Path(img_path).glob("*.bmp"))
        total_img.extend(list(Path(img_path).glob("*.tif")))

        # total_vi_img = list(Pathmain(vi_path).glob("*.bmp"))
        # total_vi_img.extend(list(Path(vi_path).glob("*.tif")))

        # assert len(total_ir_img) == len(total_vi_img), "红外图像和可见光图像数量不一致"
        total_img.sort(key=lambda x: int(x.stem))
        self.total_img = total_img
        # total_vi_img.sort(key=lambda x: int(x.stem))
        self._patch(total_img, img_path)
        # self._patch(total_vi_img,vi_path)

    def _patch(self, total_img, path):
        sub_img = []
        sub_label = []
        padding = (args.patch_size - args.label_size) // 2
        for index in range(len(total_img)):
            [h, w] = cv2.imread(str(total_img[index]), cv2.IMREAD_GRAYSCALE).shape
            for x in range(0, h - args.patch_size, args.stride_size):
                for y in range(0, w - args.patch_size, args.stride_size):
                    patch_img = cv2.imread(str(total_img[index]), cv2.IMREAD_GRAYSCALE)  # 读取整张图像
                    label = patch_img[x + padding:x + padding + args.label_size,
                            y + padding:y + padding + args.label_size]
                    patch = patch_img[x:x + args.patch_size, y:y + args.patch_size]
                    patch = patch.reshape([args.patch_size, args.patch_size, 1])
                    sub_img.append(patch)
                    sub_label.append(label)
        sub_img = np.asarray(sub_img)
        sub_label = np.asarray(sub_label)

        (Path(args.checkpoint_dir) / path).mkdir(exist_ok=True)
        self.checkpoint_path = Path(args.checkpoint_dir) / path / "train.h5"
        with h5py.File(str(self.checkpoint_path), "w") as hf:
            hf.create_dataset('data', data=sub_img)
            hf.create_dataset('label', data=sub_label)


class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.conv_bn_relu_1 = self.conv_bn_relu(2, 64, 3, 1, 0)
        self.downsampling = self.downsample()
        self.conv_bn_relu_2 = self.conv_bn_relu(64, 128, 3, 1, 0)
        self.conv_bn_relu_3 = self.conv_bn_relu(128, 256, 3, 1, 0)
        self.conv_bn_relu_4 = self.conv_bn_relu(256, 512, 3, 1, 0)
        self.conv_bn_relu_5 = self.conv_bn_relu(512, 1024, 3, 1, 0)
        self.upsampling_1 = self.upsample(1024)
        self.conv_bn_relu_6 = self.conv_bn_relu(1024, 512, 3, 1, 0)
        self.upsampling_2 = self.upsample(512)
        self.conv_bn_relu_7 = self.conv_bn_relu(512, 256, 3, 1, 0)
        self.upsampling_3 = self.upsample(256)
        self.conv_bn_relu_8 = self.conv_bn_relu(256, 128, 3, 1, 0)
        self.upsampling_4 = self.upsample(128)
        self.conv_bn_relu_9 = self.conv_bn_relu(128, 64, 3, 1, 0)

        self.conv = nn.Conv2d(64, 1, 1, 1, 0)

    def conv_bn_relu(self, in_channels, out_channels, kernel_size, stride, padding):
        cbr = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        return cbr

    def downsample(self):
        down = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=1)
        )
        return down

    def upsample(self, channels):
        up = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 2, stride=2, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.LeakyReLU()
        )
        return up

    def forward(self, x):
        out = self.conv_bn_relu_1(x)
        concat_1 = out
        out = self.downsampling(out)
        out = self.conv_bn_relu_2(out)
        concat_2 = out
        out = self.downsampling(out)
        out = self.conv_bn_relu_3(out)
        out = nn.Dropout(.2)(out)
        concat_3 = out
        out = self.downsampling(out)
        out = self.conv_bn_relu_4(out)
        concat_4 = out
        out = self.downsampling(out)
        out = nn.Dropout(.2)(out)

        out = self.conv_bn_relu_5(out)
        out = self.upsampling_1(out)
        diffH = concat_4.shape[2] - out.shape[2]
        diffW = concat_4.shape[3] - out.shape[3]
        out = F.pad(out, [diffH // 2, diffH - diffH // 2, diffW // 2, diffW - diffW // 2], "constant",
                    127)
        out = torch.cat([out, concat_4], dim=1)
        out = self.conv_bn_relu_6(out)
        out = self.upsampling_2(out)
        diffH = concat_3.shape[2] - out.shape[2]
        diffW = concat_3.shape[3] - out.shape[3]
        out = F.pad(out, [diffH // 2, diffH - diffH // 2, diffW // 2, diffW - diffW // 2], "constant",
                    127)
        out = torch.cat([out, concat_3], dim=1)
        out = self.conv_bn_relu_7(out)
        out = self.upsampling_3(out)
        diffH = concat_2.shape[2] - out.shape[2]
        diffW = concat_2.shape[3] - out.shape[3]
        out = F.pad(out, [diffH // 2, diffH - diffH // 2, diffW // 2, diffW - diffW // 2], "constant",
                    127)
        out = torch.cat([out, concat_2], dim=1)

        out = self.conv_bn_relu_8(out)
        out = self.upsampling_4(out)
        diffH = concat_1.shape[2] - out.shape[2]
        diffW = concat_1.shape[3] - out.shape[3]
        out = F.pad(out, [diffH // 2, diffH - diffH // 2, diffW // 2, diffW - diffW // 2], "constant",
                    127)
        out = torch.cat([out, concat_1], dim=1)

        out = self.conv_bn_relu_9(out)
        out = self.conv(out)
        print(f"U-net output shape:{out.shape}")
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_bn_lr_1 = self.conv_bn_lr(1, 32, 0, 2)
        self.conv_bn_lr_2 = self.conv_bn_lr(32, 64, 0, 2)
        self.conv_bn_lr_3 = self.conv_bn_lr(64, 128, 0, 2)
        self.conv_bn_lr_4 = self.conv_bn_lr(128, 256, 0, 2)

        self.liner = nn.LazyLinear(1)

    def conv_bn_lr(self, in_channels, out_channels, padding, stride):
        cbl = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        return cbl

    def forward(self, x):
        out = self.conv_bn_lr_1(x)
        out = self.conv_bn_lr_2(out)
        out = self.conv_bn_lr_3(out)
        out = self.conv_bn_lr_4(out)
        out = out.view(out.size(0), -1)
        out = self.liner(out)
        return out


def gradient(input):
    d = F.conv2d(input, torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float())
    return d


def train(G, D, ir_dataloader, vi_dataloader):
    if args.is_train:
        G_optimizer = torch.optim.Adam(G.parameters())
        D_optimizer = torch.optim.Adam(D.parameters())

        for epoch in range(args.epochs):
            for batch, ((ir_img, ir_label), (vi_img, vi_label)) in enumerate(zip(ir_dataloader, vi_dataloader)):
                ir_img = ir_img.to(device)
                ir_label = ir_label.to(device)
                vi_img = vi_img.to(device)
                vi_label = vi_label.to(device)
                mylogger.info(f"训练|开始训练第{epoch + 1}个epoch")
                input_img = torch.cat([ir_img, vi_img], dim=1)
                G.eval()
                D.train()
                G_out = G(input_img)
                D_out = D(G_out)
                pos = D(vi_label)
                D_loss = torch.mean(torch.square(D_out - torch.rand([args.batch_size, 1],device=device) * 0.3)) + torch.mean(
                    torch.square(pos - torch.rand([args.batch_size, 1],device=device) * 0.5 + 0.7))
                D_loss.backward()
                D_optimizer.step()
                D_optimizer.zero_grad()
                if (batch + 1) % args.generator_interval == 0:
                    G.train()
                    D.eval()
                    G_out = G(input_img)
                    D_out = D(G_out)
                    G_content_loss = torch.mean(
                        torch.square(G_out - ir_label) + 5 * torch.square(gradient(G_out) - gradient(vi_label)))
                    G_adversarial_loss = torch.mean(torch.square(D_out - torch.rand([args.batch_size, 1],device=device) * 0.5 + 0.7))
                    G_loss = G_adversarial_loss + 100 * G_content_loss
                    G_loss.backward()
                    G_optimizer.step()
                    G_optimizer.zero_grad()
                else:
                    continue
            if (epoch + 1) % args.log_interval == 0:
                mylogger.info(f"训练|第{epoch + 1}个epoch|G_loss:{G_loss:>5f}|D_loss:{D_loss:>5f}")
                torch.save(G.state_dict(), f"{args.checkpoint_dir}/G_{epoch + 1}.pth")
                torch.save(D.state_dict(), f"{args.checkpoint_dir}/D_{epoch + 1}.pth")
    else:
        D.eval()
        G.eval()
        with torch.inference_mode():
            for epoch in args.epochs:
                for batch, (ir_img, ir_label, (vi_img, vi_label),) in enumerate(zip(ir_dataloader, vi_dataloader)):
                    mylogger.info(f"测试|开始训练第{epoch + 1}个epoch")
                    input_img = torch.cat([ir_img, vi_img], dim=1)
                    G_out = G(input_img)
                    D_out = D(G_out)
                    pos = D(vi_label)
                    D_loss = torch.mean(torch.square(D_out - torch.rand([args.batch_size, 1]) * 0.3)) + torch.mean(
                        torch.square(pos - torch.rand([args.batch_size, 1]) * 0.5 + 0.7))
                    G_content_loss = torch.mean(
                        torch.square(G_out - ir_label) + 5 * torch.square(gradient(G_out) - gradient(vi_label)))
                    G_adversarial_loss = torch.mean(torch.square(D_out - torch.rand([args.batch_size, 1]) * 0.5 + 0.7))
                    G_loss = G_adversarial_loss + 100 * G_content_loss
                if (epoch + 1) % args.log_interval == 0:
                    mylogger.info(f"测试|第{epoch + 1}个epoch|G_loss:{G_loss:>5f}|D_loss:{D_loss:>5f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FusionGAN for pytorch.')
    parser.add_argument("--is_train", "-t", type=bool, default=True)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--patch_size", "-p", type=int, default=160)
    parser.add_argument("--label_size", "-l", type=int, default=144)
    parser.add_argument("--stride_size", "-s", type=int, default=60)
    parser.add_argument("--epochs", "-e", type=int, default=30)
    parser.add_argument("--checkpoint_dir", "-c", type=str, default="./checkpoint")
    parser.add_argument("--log_dir", "-ld", type=str, default="./log")
    parser.add_argument("--log_interval", "-li", type=int, default=5)
    parser.add_argument("--generator_interval", "-gi", type=int, default=2, help="interval between update G")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mylogger = getLogger("FusionGAN", log_dir=args.log_dir)
    ir_dataset = imgDataset(path="./Train_ir")
    vi_dataset = imgDataset(path="./Train_vi")
    ir_dataloader = DataLoader(ir_dataset, batch_size=args.batch_size, shuffle=True)
    vi_dataloader = DataLoader(vi_dataset, batch_size=args.batch_size, shuffle=True)
    G = FusionModel().to(device)
    D = Discriminator().to(device)
    train(G, D, ir_dataloader, vi_dataloader)
