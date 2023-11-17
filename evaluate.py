import numpy as np
from matplotlib.pylab import plt
from pathlib import Path
import cv2
import glob
import os
import numpy as np
import skimage.metrics as metrics
import skimage.measure as measure


def img_generator(path):
    total_img = list(Path(path).glob("*.bmp"))
    total_img.extend(list(Path(path).glob("*.jpg")))
    total_img.extend(list(Path(path).glob("*.tif")))
    total_img.extend(list(Path(path).glob("*.png")))
    total_img.sort(key=lambda x: int(x.stem))
    for file in total_img:
        if file.is_file():
            img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            img = img.reshape(img.shape[0], img.shape[1], 1)
            yield img


def qualitative_analysis(source_1_path, source_2_path, fused_path,limit=None):
    """
    :param source_1_path: near focus
    """
    img = glob.glob(fused_path + "/*.jpg")
    img.extend(glob.glob(fused_path + "/*.bmp"))
    img.extend(glob.glob(fused_path + "/*.tif"))
    img.extend(glob.glob(fused_path + "/*.png"))

    img.sort(key=lambda x: Path(x).stem)
    total_img = len(img)
    model_path, rescale_way = fused_path.split("/")[-2:]
    print(f"定性分析:|{model_path}|{rescale_way}一共{total_img}张融合图片")
    if limit is None:
        rows = total_img
    else:
        rows = min(limit, total_img)
    source_1 = img_generator(source_1_path)
    source_2 = img_generator(source_2_path)
    fused = img_generator(fused_path)
    fig, axes = plt.subplots(rows, 3, figsize=(5, 5))
    for i in range(rows):
        s1 = next(source_1)
        s2 = next(source_2)
        f = next(fused)
        if i == 0:
            axes[i, 0].title.set_text(f"infrared img")
            axes[i, 1].title.set_text(f"visible img")
            axes[i, 2].title.set_text(f"fused")
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
        axes[i, 2].axis("off")
        axes[i, 0].imshow(s1,cmap="gray",vmin=0,vmax=255)
        axes[i, 1].imshow(s2,cmap="gray",vmin=0,vmax=255)
        axes[i, 2].imshow(f,cmap="gray",vmin=0,vmax=255)
    # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.35)
    savepath = f"figs/{model_path}/{rescale_way}"
    Path(savepath).mkdir(exist_ok=True, parents=True)
    plt.savefig(f'{savepath}/qualitative_analysis.jpg')
    plt.show()


def img_ssim(img1, img2):
    channels = img1.shape[2]
    ssims = []
    for channel in range(channels):
        ssim = metrics.structural_similarity(img1[:, :, channel], img2[:, :, channel], data_range=255, )
        ssims.append(ssim)
    return np.mean(ssims)


def quantitative_analysis(source_1_path, source_2_path, fused_path,limit=None):
    """
    SSIM,PSNR,MI,MSE,EN
    """
    img = glob.glob(fused_path + "/*.jpg")
    img.extend(glob.glob(fused_path + "/*.bmp"))
    img.extend(glob.glob(fused_path + "/*.tif"))
    img.extend(glob.glob(fused_path + "/*.png"))

    img.sort(key=lambda x: Path(x).stem)
    total_img = len(img)
    model_path,rescale_way = fused_path.split("/")[-2:]
    print(f"定量分析|{model_path}|{rescale_way}:一共{total_img}张融合图片")
    source_1 = img_generator(source_1_path)
    source_2 = img_generator(source_2_path)
    fused = img_generator(fused_path)
    if limit is None:
        rows = total_img
    else:
        rows = min(limit,total_img)
    all_ssim = []
    all_psnr = []
    all_mi = []
    all_mse = []
    all_en = []
    all_sf = []
    all_sd = []
    all_ag = []
    all_metrics = {'SSIM': all_ssim, 'PSNR': all_psnr, 'NMI': all_mi, 'MSE': all_mse, 'EN': all_en, 'SF': all_sf,
                   'SD': all_sd, 'AG': all_ag}
    fig, axes = plt.subplots(1, len(all_metrics), figsize=(10, 10), tight_layout=True)
    for i in range(rows):
        img1 = next(source_1)
        img2 = next(source_2)
        f = next(fused)
        # ssim
        ssim_1 = img_ssim(img1, f)
        ssim_2 = img_ssim(img2, f)
        ssim = (ssim_1 + ssim_2) / 2
        all_ssim.append(ssim)
        # psnr
        psnr_1 = metrics.peak_signal_noise_ratio(img1, f, data_range=255)
        psnr_2 = metrics.peak_signal_noise_ratio(img2, f, data_range=255)
        psnr = (psnr_1 + psnr_2) / 2
        all_psnr.append(psnr)
        # mi
        mi_1 = metrics.normalized_mutual_information(img1, f)
        mi_2 = metrics.normalized_mutual_information(img2, f)
        mi = (mi_1 + mi_2) / 2
        all_mi.append(mi)
        # mse
        mse_1 = metrics.mean_squared_error(img1, f)
        mse_2 = metrics.mean_squared_error(img2, f)
        mse = (mse_1 + mse_2) / 2
        all_mse.append(mse)
        # en
        en = measure.shannon_entropy(f)
        # sf
        sf = img_sf(f)
        # sd
        sd = img_sd(f)
        # ag
        ag = img_ag(f)

        all_en.append(en)
        all_sf.append(sf)
        all_sd.append(sd)
        all_ag.append(ag)
    all_en.append(np.mean(all_en))
    all_sf.append(np.mean(all_sf))
    all_sd.append(np.mean(all_sd))
    all_ssim.append(np.mean(all_ssim))
    all_psnr.append(np.mean(all_psnr))
    all_mi.append(np.mean(all_mi))
    all_mse.append(np.mean(all_mse))
    all_ag.append(np.mean(all_ag))
    markers = ["8", "o", "s", "^", "D", "4", "1", "2"]
    # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.35)
    for index, [key, value] in enumerate(all_metrics.items()):
        axes[index].set_xticks([])
        axes[index].bar([str(i) for i in range(1,len(value))] + ["mean"], value)
        axes[index].set_title(key)
        # axes[index].legend()
    plt.suptitle("quantitative analysis")
    savepath = f"figs/{model_path}/{rescale_way}"
    Path(savepath).mkdir(exist_ok=True,parents=True)
    plt.savefig(f'{savepath}/quantitative_analysis.jpg')
    plt.show()


def img_ag(img):
    """
    average gradient
    """
    h, w, channels, = img.shape
    img = img / 255
    img = img + 0.00001
    ags = []
    for i in range(channels):
        [grady, gradx] = np.gradient(img[:, :, i])
        s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)
        ag = np.sum(s) / (h * w)
        ags.append(ag)
    return np.mean(ags)


def img_sf(img):
    """
    spatial frequency
    """
    h, w, channels, = img.shape
    img = img / 255
    img = img + 0.00001
    sfs = []
    for i in range(channels):
        rf = np.sqrt(np.mean(np.square(img[:, 1:] - img[:, 0:-1])))
        cf = np.sqrt(np.mean(np.square(img[1:, :] - img[0:-1, :])))
        sf = np.sqrt(np.square(rf)) + np.square(cf)
        sfs.append(sf)
    return np.mean(sfs)

    # image_array = np.array(image)
    # RF = np.diff(image_array, axis=0)
    # RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    # CF = np.diff(image_array, axis=1)
    # CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    # SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    # return SF


def img_sd(img):
    """
    standard deviation
    """
    img = img.astype(np.float32)
    img = img / 255
    sd = np.std(img)
    return sd


if __name__ == '__main__':
    model_name = "U_GAN"
    assert Path(f"./Test_result/{model_name}").exists() ,f"没有{model_name}模型的测试结果"
    total_rescale_way = list([i for i in os.listdir(f"./Test_result/{model_name}") if os.path.isdir(f"./Test_result/{model_name}/{i}") and not i.startswith(".")])
    # rescale_way = "norm_-1_1_resize"
    fused_paths = [f"./Test_result/{model_name}/{rescale_way}" for rescale_way in total_rescale_way]
    ir_path ="./Test_ir"
    vi_path = "./Test_vi"
    assert Path(ir_path).exists(), f"没有{ir_path}文件夹"
    assert Path(vi_path).exists(), f"没有{vi_path}文件夹"
    [qualitative_analysis(ir_path, vi_path, fused_path,limit=5) for fused_path in fused_paths]
    [quantitative_analysis(ir_path, vi_path, fused_path) for fused_path in fused_paths]