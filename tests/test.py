# import glob
# from pathlib import Path
# import numpy as np
# import cv2
# img = cv2.imread("../Train_ir/1.bmp", cv2.IMREAD_GRAYSCALE)
# padding = 6
# cv2.imshow("img",img)
# print(img.shape)
# cv2.waitKey(0)
# img = np.pad(img, ((padding // 2, padding - padding // 2), (padding // 2, padding - padding // 2)),'constant', constant_values=(127, 127))
#
# cv2.imshow("img",img)
# cv2.waitKey(0)
import cv2
import torch
from torch import nn
# img = cv2.imread("../Train_ir/1.bmp")
# [h,w,_] = img.shape
# print(h,w)
# img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).float()
# out = nn.Conv2d(3, 1, kernel_size=160, stride=60, padding=0, bias=False)
# img = out(img)
# print(img.shape)
import numpy as np
import h5py
from PIL import Image
import imageio
import matplotlib.pylab as plt
import numpy as np
import cv2
print(("训练" if True else "测试") + f"|第张图片做切分")
# # 新建numpy数组，注意np.zero()创建的数据类型为float64

# plt.imshow(img,cmap="gray",vmin=0,vmax=255)
# plt.show()
# c = Image.fromarray(np.array(a))
# c.show()
# cv2.imshow("img",img[1000])
# cv2.imshow("label",label[1000])
# cv2.waitKey(0)
# img = torch.ones(152,152,1,dtype=torch.uint8)*200
# img = img.numpy()
# plt.imshow(img,cmap="gray",vmin=0,vmax=255)
# plt.show()
# label[0] = label[0].reshape(1, 152, 152)
from model import U_GAN
from model import Discriminator
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# D = Discriminator().to(device)
# trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
#
# print(trans(label[0]).shape)
#
import torch.nn.functional as F
from torchsummary import summary

import torch.nn as nn
# from pathlib import Path
# path = "../Train_ir"
#
# ir_img = list(Path(path).glob("*.bmp"))
# ir_img.extend(Path(path).glob("*.jpg"))
# ir_img.extend(Path(path).glob("*.png"))
# ir_img.extend(Path(path).glob("*.tif"))
# ir_img.sort(key=lambda x: int(x.stem))
# print(ir_img)
#
# vi_img = list(Path(path).glob("*.bmp"))
# vi_img.extend(Path(path).glob("*.jpg"))
# vi_img.extend(Path(path).glob("*.png"))
# vi_img.extend(Path(path).glob("*.tif"))
# vi_img.sort(key=lambda x: int(x.stem))
# print(vi_img)
#
#
#
# img = cv2.imread("../Train_ir/1.bmp",cv2.IMREAD_GRAYSCALE)
# img = torch.from_numpy(img).unsqueeze(0).cpu().float()
# img = torch.cat([img,img],dim=1)
# print(img.shape)
# model = U_GAN()
# summary(model,input_size=(2,1152,768),device="cpu")
#
