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
import onnxruntime as ort

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def visualize_disparity(disparity_map, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(disparity_map, cmap='jet')
    plt.colorbar(label="Disparity")
    plt.title(title)
    plt.axis('off')
    plt.show()


def demo(args):
    # 加载模型
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    model = model.module
    model.to(DEVICE)
    model.eval()

    # 导出 ONNX 模型
    onnx_model_path = args.output_onnx
    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images.")

        # 获取输入尺寸
        sample_image1 = load_image(left_images[0])
        sample_image2 = load_image(right_images[0])
        padder = InputPadder(sample_image1.shape, divis_by=32)
        sample_image1, sample_image2 = padder.pad(sample_image1, sample_image2)

        # 导出为 ONNX
        dummy_input = (sample_image1, sample_image2)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            input_names=["left", "right"],
            output_names=["pred_disp"],
            dynamic_axes={
                "left": {0: "batch_size", 2: "height", 3: "width"},
                "right": {0: "batch_size", 2: "height", 3: "width"},
                "pred_disp": {0: "batch_size", 1: "height", 2: "width"}
            },
            opset_version=16,
        )
        print(f"ONNX model exported to {onnx_model_path}")

    # PyTorch 和 ONNX 推理对比
    for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        # PyTorch 推理
        with torch.no_grad():
            disp_torch = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp_torch = padder.unpad(disp_torch).squeeze().cpu().numpy()

        # ONNX 推理
        ort_session = ort.InferenceSession(onnx_model_path)
        input_l_np = image1.cpu().numpy()
        input_r_np = image2.cpu().numpy()
        onnx_inputs = {"left": input_l_np, "right": input_r_np}
        onnx_outputs = ort_session.run(None, onnx_inputs)
        disp_onnx = onnx_outputs[0].squeeze()

        # 可视化对比
        visualize_disparity(disp_torch, title="PyTorch Disparity Map")
        visualize_disparity(disp_onnx, title="ONNX Disparity Map")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default="IGEVplusplus/pretrained_models/igev_plusplus/sceneflow.pth")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
                        default="IGEVplusplus/demo-imgs/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
                        default="IGEVplusplus/demo-imgs/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output",
                        default="IGEVplusplus/demo_output")
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float16', choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')
    parser.add_argument('--output_onnx', help="path to save the ONNX model", default="igevmodel.onnx")

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    parser.add_argument('--s_disp_range', type=int, default=48,
                        help="max disp of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_range', type=int, default=96,
                        help="max disp of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_range', type=int, default=192,
                        help="max disp of large disparity-range geometry encoding volume")
    parser.add_argument('--s_disp_interval', type=int, default=1,
                        help="disp interval of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_interval', type=int, default=2,
                        help="disp interval of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_interval', type=int, default=4,
                        help="disp interval of large disparity-range geometry encoding volume")

    args = parser.parse_args()
    demo(args)
