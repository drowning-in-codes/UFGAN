import math
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
from tqdm import tqdm
from model import U_GAN, Discriminator
from logger import getLogger
from torch.utils.tensorboard import SummaryWriter
from loss import gradient_loss, l2_norm, mse_loss, ssim_loss


class imgDataset(Dataset):
    global args
    global mylogger

    def __init__(self, is_train=True, transform=True, path="./Train_ir"):
        super(imgDataset, self).__init__()
        self.checkpoint_path = None
        self.is_train = is_train
        self.transform = transform
        (Path(args.checkpoint_dir) / path).mkdir(exist_ok=True)
        if self.transform:
            self.trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        if self.is_train:
            self.checkpoint_path = str(Path(args.checkpoint_dir) / path / "train.h5")
            if not args.override_data and Path(self.checkpoint_path).exists():
                mylogger.info(f"训练|已经存在训练集的h5文件,直接读取")
            else:
                self.patch_img(path)

            with h5py.File(self.checkpoint_path, 'r') as hf:
                self.img = np.array(hf.get('data'))
                self.label = np.array(hf.get('label'))
        else:
            self.img_path = path
            self.checkpoint_path = Path(args.checkpoint_dir) / path / "test.h5"
            if not args.override_data and Path(self.checkpoint_path).exists():
                mylogger.info(f"测试|已经存在测试集的h5文件,直接读取")
            else:
                mylogger.info(f"测试|制作测试集")
                total_img = list(Path(path).glob("*.bmp"))
                total_img.extend(Path(path).glob("*.jpg"))
                total_img.extend(Path(path).glob("*.png"))
                total_img.extend(Path(path).glob("*.tif"))
                total_img.sort(key=lambda x: int(x.stem))
                self.patch_img(total_img)
                patch_gen = self._patch(total_img)
                pbar = tqdm(patch_gen)
                for idx, patch_data in enumerate(pbar):
                    pbar.set_description(f"测试|第{idx + 1}张图片做切分")
                    sub_img, sub_label = patch_data

                    with h5py.File(str(self.checkpoint_path), "a") as hf:
                        hf.create_dataset('data', data=sub_img)
                        hf.create_dataset('label', data=sub_label)

            with h5py.File(self.checkpoint_path, 'r') as hf:
                self.img = np.array(hf.get('data'))
                self.label = np.array(hf.get('label'))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.label[idx]
        if self.transform:
            img = self.trans(img)
            label = self.trans(label)
        return img, label

    def patch_img(self, img_path):
        mylogger.info(f"训练|开始切分训练集")

        total_img = list(Path(img_path).glob("*.bmp"))
        total_img.extend(list(Path(img_path).glob("*.tif")))
        total_img.extend(list(Path(img_path).glob("*.jpg")))
        total_img.extend(list(Path(img_path).glob("*.png")))

        total_img.sort(key=lambda x: int(x.stem))

        patch_gen = self._patch(total_img)
        pbar = tqdm(patch_gen)
        for idx, patch_data in enumerate(pbar):
            pbar.set_description(f"训练|第{idx + 1}张图片做切分")
            sub_img, sub_label = patch_data

            with h5py.File(str(self.checkpoint_path), "a") as hf:
                hf.create_dataset('data', data=sub_img)
                hf.create_dataset('label', data=sub_label)

    def _patch(self, total_img):
        sub_img = []
        sub_label = []
        if args.is_train:
            padding = (args.patch_size - args.label_size) // 2
            for index in range(len(total_img)):
                [h, w] = cv2.imread(str(total_img[index]), cv2.IMREAD_GRAYSCALE).shape
                for x in range(0, h - args.patch_size, args.stride_size):
                    for y in range(0, w - args.patch_size, args.stride_size):
                        patch_img = cv2.imread(str(total_img[index]), cv2.IMREAD_GRAYSCALE)  # 读取整张图像
                        label = patch_img[x + padding:x + padding + args.label_size,
                                y + padding:y + padding + args.label_size]
                        label = label.reshape([args.label_size, args.label_size, 1])
                        patch = patch_img[x:x + args.patch_size, y:y + args.patch_size]
                        patch = patch.reshape([args.patch_size, args.patch_size, 1])
                        sub_img.append(patch)
                        sub_label.append(label)
            sub_img = np.asarray(sub_img)
            sub_label = np.asarray(sub_label)
            yield sub_img, sub_label
        else:
            for index in range(len(total_img)):
                label = cv2.imread(str(total_img[index]), cv2.IMREAD_GRAYSCALE)
                padding = args.patch_size - args.label_size
                # 将源图像做填充
                img = np.pad(label, ((padding // 2, padding - padding // 2), (padding // 2, padding - padding // 2)),
                             "constant", constant_values=(127, 127))
                label = np.reshape(img, [img.shape[0], img.shape[1], 1])
                label = np.reshape(label, [label.shape[0], label.shape[1], 1])

                sub_img.append(img)
                sub_label.append(label)
            sub_img = np.array(sub_img)
            sub_label = np.array(sub_label)
            yield sub_img, sub_label

        #
        # with h5py.File(str(self.checkpoint_path), "w") as hf:
        #     hf.create_dataset('data', data=sub_img)
        #     hf.create_dataset('label', data=sub_label)


def gradient(x):
    # [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]] sobel算子
    # [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]] laplace算子
    d = F.conv2d(x, torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device, ).unsqueeze(0).unsqueeze(
        0), padding=1)  # sobel算子
    return d


def train(G, D, ir_dataloader, vi_dataloader):
    global writer
    epoch_G_loss = []
    epoch_D_loss = []
    if args.is_train:
        G_optimizer = torch.optim.Adam(G.parameters())
        D_optimizer = torch.optim.Adam(D.parameters())
        for epoch in tqdm(range(args.epochs)):
            mylogger.info(f"训练|开始训练第{epoch + 1}个epoch,一次epoch包含{len(ir_dataloader)}个batch")
            for batch, ((ir_img, ir_label), (vi_img, vi_label)) in enumerate(zip(ir_dataloader, vi_dataloader)):
                ir_img = ir_img.to(device)
                ir_label = ir_label.to(device)
                vi_img = vi_img.to(device)
                vi_label = vi_label.to(device)
                input_img = torch.cat([ir_img, vi_img], dim=1)
                G_out = G(input_img)
                D_out = D(G_out)
                pos = D(vi_label)
                batch_size = D_out.shape[0]
                # 辨别器损失
                # D_loss = torch.mean(
                #     torch.square(D_out - torch.rand([batch_size, 1], device=device) * 0.3)) + torch.mean(
                #     torch.square(pos - torch.rand([batch_size, 1], device=device) * 0.5 + 0.7))
                b = torch.rand([batch_size, 1], device=device) * 0.5 + 0.7
                a = torch.rand([batch_size, 1], device=device) * 0.3
                D_loss = mse_loss(D_out, a) + mse_loss(pos, b)
                D_loss.backward()
                epoch_D_loss.append(D_loss.item())
                D_optimizer.step()
                D_optimizer.zero_grad()
                if (batch + 1) % args.generator_interval == 0:
                    G_out = G(input_img)
                    D_out = D(G_out)
                    G_content_loss = l2_norm(G_out, ir_label) / G_out.numel() + 5 * l2_norm(gradient(G_out), gradient(
                        vi_label)) / G_out.numel()
                    # G_content_loss = torch.mean(
                    #     torch.square(G_out - ir_label)) + 5 * torch.mean(
                    #     torch.square(gradient(G_out) - gradient(vi_label)))
                    # G_adversarial_loss = torch.mean(
                    #     torch.square(D_out - torch.rand([batch_size, 1], device=device) * 0.5 + 0.7))
                    c = torch.rand([batch_size, 1], device=device) * 0.5 + 0.7
                    G_adversarial_loss = mse_loss(D_out, c)
                    # 生成器损失
                    G_loss = G_adversarial_loss + 100 * G_content_loss
                    epoch_G_loss.append(G_loss.item())
                    G_loss.backward()
                    G_optimizer.step()
                    G_optimizer.zero_grad()
                else:
                    continue
            if (epoch + 1) % args.log_interval == 0:
                mean_G_loss = np.mean(epoch_G_loss)
                mean_D_loss = np.mean(epoch_D_loss)
                mylogger.info(f"训练|第{epoch + 1}个epoch|G_loss:{mean_G_loss:>5f}|D_loss:{mean_D_loss:>5f}")
                writer.add_scalar("train/G_loss", mean_G_loss, epoch + 1)
                writer.add_scalar("train/D_loss", mean_D_loss, epoch + 1)

                torch.save(G.state_dict(), f"{args.checkpoint_dir}/G_{epoch + 1}.pth")
                torch.save(D.state_dict(), f"{args.checkpoint_dir}/D_{epoch + 1}.pth")
    else:
        D.eval()
        G.eval()
        epoch_G_loss = []
        epoch_D_loss = []
        with torch.inference_mode():
            for epoch in tqdm(args.epochs):
                mylogger.info(f"测试|开始训练第{epoch + 1}个epoch")
                for batch, (ir_img, ir_label, (vi_img, vi_label),) in enumerate(zip(ir_dataloader, vi_dataloader)):
                    input_img = torch.cat([ir_img, vi_img], dim=1)
                    G_out = G(input_img)
                    D_out = D(G_out)
                    pos = D(vi_label)
                    batch_size = D_out.shape[0]
                    b = torch.rand([batch_size, 1], device=device) * 0.5 + 0.7
                    a = torch.rand([batch_size, 1], device=device) * 0.3
                    D_loss = mse_loss(D_out, a) + mse_loss(pos, b)
                    # D_loss = torch.mean(torch.square(D_out - torch.rand([args.batch_size, 1]) * 0.3)) + torch.mean(
                    #     torch.square(pos - torch.rand([args.batch_size, 1]) * 0.5 + 0.7))
                    # G_content_loss = torch.mean(
                    #     torch.square(G_out - ir_label) + 5 * torch.square(gradient(G_out) - gradient(vi_label)))
                    # G_adversarial_loss = torch.mean(torch.square(D_out - torch.rand([args.batch_size, 1]) * 0.5 + 0.7))
                    # G_loss = G_adversarial_loss + 100 * G_content_loss

                    # 生成器损失
                    c = torch.rand([batch_size, 1], device=device) * 0.5 + 0.7
                    G_adversarial_loss = mse_loss(D_out, c)
                    G_content_loss = l2_norm(G_out, ir_label) / G_out.numel() + 5 * l2_norm(gradient(G_out), gradient(
                        vi_label)) / G_out.numel()

                    G_ssim_loss = ssim_loss(G_out, ir_label) + 5 * ssim_loss(G_out, vi_label)

                    G_loss = G_adversarial_loss + 100 * G_content_loss

                    epoch_G_loss.append(G_loss.item())
                    epoch_D_loss.append(D_loss.item())
                if (epoch + 1) % args.log_interval == 0:
                    mean_G_loss = np.mean(epoch_G_loss)
                    mean_D_loss = np.mean(epoch_D_loss)
                    mylogger.info(f"测试|第{epoch + 1}个epoch|G_loss:{mean_G_loss:>5f}|D_loss:{mean_D_loss:>5f}")
                    writer.add_scalar("test/G_loss", mean_G_loss, epoch + 1)
                    writer.add_scalar("test/D_loss", mean_D_loss, epoch + 1)
                    writer.add_image("test/output_img", G_out.unsqueeze(0).detach().cpu(), epoch + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FusionGAN for pytorch.')
    parser.add_argument("--is_train", "-t", type=bool, default=True)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--patch_size", "-p", type=int, default=160)
    parser.add_argument("--label_size", "-l", type=int, default=152)
    parser.add_argument("--stride_size", "-s", type=int, default=60)
    parser.add_argument("--epochs", "-e", type=int, default=30)
    parser.add_argument("--checkpoint_dir", "-c", type=str, default="./checkpoint")
    parser.add_argument("--log_dir", "-ld", type=str, default="./log.txt")
    parser.add_argument("--vis_log", "-vl", type=str, default="./log")
    parser.add_argument("--log_interval", "-li", type=int, default=5)
    parser.add_argument("--override_data", "-od", type=bool, default=False, help="whether to override dataset")
    parser.add_argument("--generator_interval", "-gi", type=int, default=2, help="interval between update G")
    args = parser.parse_args()
    # 设置运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tensorboard可视化
    Path(args.vis_log).mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=args.vis_log)

    mylogger = getLogger("FusionGAN", log_dir=args.log_dir)
    ir_dataset = imgDataset(path="./Train_ir")
    vi_dataset = imgDataset(path="./Train_vi")
    ir_dataloader = DataLoader(ir_dataset, batch_size=args.batch_size, shuffle=True)
    vi_dataloader = DataLoader(vi_dataset, batch_size=args.batch_size, shuffle=True)
    assert len(ir_dataloader) == len(vi_dataloader), "红外图像和可见光图像数量不一致"
    G = U_GAN().to(device)
    D = Discriminator().to(device)
    train(G, D, ir_dataloader, vi_dataloader)
