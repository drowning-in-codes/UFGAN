import torch
from torch import nn
import torchvision
from pathlib import Path
from model import U_GAN
import cv2

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = U_GAN().to(device)
    Path("./Test_result").mkdir(exist_ok=True)
    model.load_state_dict(torch.load("./checkpoint/G_30.pth"))
    ir_img = list(Path("./Test_ir").glob("*.bmp"))
    ir_img.extend(list(Path("./Test_ir").glob("*.tif")))
    ir_img.extend(list(Path("./Test_ir").glob("*.jpg")))
    ir_img.sort(key=lambda x: int(x.stem))

    vi_img = list(Path("./Test_vi").glob("*.bmp"))
    vi_img.extend(list(Path("./Test_vi").glob("*.tif")))
    vi_img.extend(list(Path("./Test_vi").glob("*.jpg")))
    vi_img.sort(key=lambda x: int(x.stem))
    for i in range(len(vi_img)):
        ir_ig = cv2.imread(str(ir_img[i]), cv2.IMREAD_GRAYSCALE)
        vi_ig = cv2.imread(str(vi_img[i]), cv2.IMREAD_GRAYSCALE)
        input_img = torch.cat([torch.from_numpy(ir_ig).unsqueeze(0).unsqueeze(0).to(device).float(),
                               torch.from_numpy(vi_ig).unsqueeze(0).unsqueeze(0).to(device).float()], dim=1)
        output_img = model(input_img.to(device))
        output_img = output_img.squeeze(0).squeeze(0).detach().cpu().numpy()
        output_img = output_img.astype("uint8")
        cv2.imwrite(f"./Test_result/{ir_img[i].name}", output_img)
        print(f"第{i+1}张图片已经完成")