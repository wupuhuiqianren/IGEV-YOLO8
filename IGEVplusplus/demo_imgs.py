import sys
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from IGEVplusplus.core.igev_stereo import IGEVStereo
from IGEVplusplus.core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2
from IGEVplusplus.core.utils.frame_utils import readPFM


DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)

            # 打印输出张量的形状和基本信息
            print(f"Shape of raw disparity output: {disp.shape}")
            print(f"Sample disparity values (before squeeze): {disp.cpu().numpy()[:1, :5, :5]}")  # 仅打印前 5x5 的值

            disp = disp.squeeze()  # 移除 batch 维度
            print(f"Shape of disparity after squeeze: {disp.shape}")
            print(f"Sample disparity values (after squeeze): {disp[:5, :5]}")

            file_stem = imfile1.split('/')[-2]
            filename = os.path.join(output_directory, f'{file_stem}.png')
            disp = disp.cpu().numpy().squeeze()
            plt.imsave(filename, disp.squeeze(), cmap='jet')
            
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disp.squeeze())

            # disp = np.round(disp * 256).astype(np.uint16)
            # cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(disp.squeeze(), alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="C:/Users/Tianle Zhu/PycharmProjects/openStereo/OpenStereo/IGEVplusplus/pretrained_models/igev_plusplus/sceneflow.pth")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="C:/Users/Tianle Zhu/PycharmProjects/openStereo/OpenStereo/IGEVplusplus/demo-imgs/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="C:/Users/Tianle Zhu/PycharmProjects/openStereo/OpenStereo/IGEVplusplus/demo-imgs/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="C:/Users/Tianle Zhu/PycharmProjects/openStereo/OpenStereo/IGEVplusplus/demo_output")
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    parser.add_argument('--s_disp_range', type=int, default=48, help="max disp of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_range', type=int, default=96, help="max disp of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_range', type=int, default=192, help="max disp of large disparity-range geometry encoding volume")
    parser.add_argument('--s_disp_interval', type=int, default=1, help="disp interval of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_interval', type=int, default=2, help="disp interval of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_interval', type=int, default=4, help="disp interval of large disparity-range geometry encoding volume")
    
    args = parser.parse_args()

    demo(args)
