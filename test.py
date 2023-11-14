import torch
from torch import nn
import torchvision
from pathlib import Path
from main import FusionModel
import cv2

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel().to(device)
    Path("./Test_result").mkdir(exist_ok=True)
    model.load_state_dict(torch.load("./checkpoint/epoch_0.pth"))
    total_img = list(Path("./Test_ir").glob("*.bmp"))
    total_img.extend(list(Path("./Test_ir").glob("*.tif")))
    total_img.extend(list(Path("./Test_ir").glob("*.jpg")))
    for i in range(1, len(total_img) + 1):
        ir_img = cv2.imread(f"./Test_ir/{i}.bmp", cv2.IMREAD_GRAYSCALE)
        vi_img = cv2.imread(f"./Test_vi/{i}.bmp", cv2.IMREAD_GRAYSCALE)
        input_img = torch.cat([torch.from_numpy(ir_img).unsqueeze(0).unsqueeze(0).to(device).float(),
                               torch.from_numpy(vi_img).unsqueeze(0).unsqueeze(0).to(device).float()], dim=1)
        output_img = model(input_img)
        output_img = output_img.squeeze(0).squeeze(0).detach().numpy()
        output_img = output_img.astype("uint8")
        cv2.imwrite(f"./Test_result/{i}.bmp", output_img)
