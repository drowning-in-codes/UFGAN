import torch
from torch import nn
import torchvision
from pathlib import Path
from model import U_GAN, FusionModel, DDcGAN
import cv2
import numpy as np
import argparse
from utils import str2bool
def test_loop(rescale_way, model, ir_img, vi_img, padding=8, do_patch=False,limit=50):
    norm_neg = False
    resize_flag = False
    val = rescale_way.split("_")[1]
    if val == -1:
        norm_neg = True
    scale = rescale_way.split("_")[-1]
    if scale == "resize":
        resize_flag = True

    if do_patch:
        save_path = f"./Test_result/{model.__class__.__name__}/train_on_patch/{rescale_way}"
    else:
        save_path = f"./Test_result/{model.__class__.__name__}/{rescale_way}"

    print(f"测试|{'_'.join(save_path.split('/')[2:])}")
    limit = min(limit,len(ir_img))
    print(f"一共测试{limit}张照片")
    Path(save_path).mkdir(exist_ok=True, parents=True)
    for i in range(limit):
        ir_ig = cv2.imread(str(ir_img[i]), cv2.IMREAD_GRAYSCALE)
        vi_ig = cv2.imread(str(vi_img[i]), cv2.IMREAD_GRAYSCALE)
        if norm_neg:
            ir_ig = (ir_ig - 127.5) / 127.5
            vi_ig = (vi_ig - 127.5) / 127.5
        else:
            ir_ig = ir_ig / 255
            vi_ig = vi_ig / 255
        if resize_flag:
            # do resize
            ir_ig = cv2.resize(ir_ig, (ir_ig.shape[1] + padding, ir_ig.shape[0] + padding))
            vi_ig = cv2.resize(vi_ig, (vi_ig.shape[1] + padding, vi_ig.shape[0] + padding))
        else:
            ir_ig = np.pad(ir_ig, ((padding // 2, padding - padding // 2), (padding // 2, padding - padding // 2)),
                           "constant", constant_values=0)
            vi_ig = np.pad(vi_ig, ((padding // 2, padding - padding // 2), (padding // 2, padding - padding // 2)),
                           "constant", constant_values=0)
        input_img = torch.cat([torch.from_numpy(ir_ig).unsqueeze(0).unsqueeze(0).to(device).float(),
                               torch.from_numpy(vi_ig).unsqueeze(0).unsqueeze(0).to(device).float()], dim=1)
        output_img = model(input_img.to(device))
        output_img = output_img.squeeze(0).detach().cpu().numpy()
        # output_img = (output_img + 1)/2
        output_img = output_img.transpose(1, 2, 0)
        output_img = output_img * 127.5 + 127.5
        output_img = output_img.astype("uint8")

        cv2.imwrite(f"{save_path}/{ir_img[i].name}", output_img)
        print(f"第{i + 1}张图片已经完成")
    print(f"结束测试|{'_'.join(save_path.split('/')[2:])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate fused image for test')
    parser.add_argument('--do_patch',"-dp", type=str2bool, default=True,help='use model trained on patch or not')
    parser.add_argument('--epoch_size',"-es", type=int, default=30, help='epoch size of trained model')
    parser.add_argument("--model_name","-m",type=str,default="U_GAN",help="model name")
    parser.add_argument("--checkpoint_path","-c",type=str,default="./checkpoint",help="location of checkpoint")

    args = parser.parse_args()

    do_patch = args.do_patch
    epoch_size = args.epoch_size

    ir_path = "./Test_ir"
    vi_path = "./Test_vi"

    rescale_ways = [f"norm_{num}_1_{way}" for num in [-1,0] for way in ["resize","padding"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name == "U_GAN":
        model = U_GAN().to(device)
        padding = 8
    elif args.model_name == "FusionModel":
        model = FusionModel().to(device)
        padding = 0
    elif args.model_name == "DDcGAN":
        model = DDcGAN().to(device)
        padding = 0
    else:
        model = FusionModel(True).to(device)
        padding = 20

    if do_patch:
        checkpoint_path = f"{args.checkpoint_path}/{model.__class__.__name__}/train_on_patch/G_{epoch_size}.pth"
    else:
        checkpoint_path = f"{args.checkpoint_path}/{model.__class__.__name__}/G_{epoch_size}.pth"
    assert Path(checkpoint_path).exists(), f"没有{checkpoint_path}模型的训练结果"
    assert Path(ir_path).exists(), f"没有{ir_path}文件夹"
    assert Path(vi_path).exists(), f"没有{vi_path}文件夹"

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    ir_img = list(Path(ir_path).glob("*.bmp"))
    ir_img.extend(list(Path(ir_path).glob("*.tif")))
    ir_img.extend(list(Path(ir_path).glob("*.jpg")))
    ir_img.extend(list(Path(ir_path).glob("*.png")))

    ir_img.sort(key=lambda x: int(x.stem))

    vi_img = list(Path(vi_path).glob("*.bmp"))
    vi_img.extend(list(Path(vi_path).glob("*.tif")))
    vi_img.extend(list(Path(vi_path).glob("*.jpg")))
    vi_img.extend(list(Path(vi_path).glob("*.png")))

    vi_img.sort(key=lambda x: int(x.stem))
    [test_loop(rescale_way, model, ir_img, vi_img, padding, do_patch) for rescale_way in rescale_ways]
