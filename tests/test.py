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
from torchsummary import summary
# with h5py.File("../checkpoint/Train_ir/train.h5", 'r') as hf:
#     img = np.array(hf.get('data'))
#     label = np.array(hf.get('label'))
# print(len(img))

from model import U_GAN


img = cv2.imread("../Train_ir/1.bmp",cv2.IMREAD_GRAYSCALE)
img = torch.from_numpy(img).unsqueeze(0).cpu().float()
img = torch.cat([img,img],dim=1)
print(img.shape)
model = U_GAN()
summary(model,input_size=(2,1152,768),device="cpu")