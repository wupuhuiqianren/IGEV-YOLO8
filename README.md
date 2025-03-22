# IGEV-YOLO8
Combine IGEV++ with YOLO8 segmentation, used for autonomous driving applications.  
将 IGEV++ 与 YOLO8 分割模型相结合，用于自动驾驶场景。

---

# 项目说明 / Project Description
这个项目基于 [YOLOv8](https://github.com/ultralytics/ultralytics) 和 [IGEV++](https://github.com/gangweiX/IGEV-plusplus)。  
This project is based on [YOLOv8](https://github.com/ultralytics/ultralytics) and [IGEV++](https://github.com/gangweiX/IGEV-plusplus).

此项目融合了两种深度学习模型作为一个系统，实现 YOLO 进行目标检测和分割，IGEV++ 提取被检测目标与摄像头之间的距离信息，最后一起显示在摄像头输出画面上。    
It integrates two deep learning models into one system: YOLO performs object detection and segmentation, while IGEV++ extracts the distance information between detected objects and the camera, and then displays the results on the camera output.

简单的效果展示，项目已部署在 Jetson AGX Orin 上：    
A simple demonstration of the project deployed on Jetson AGX Orin: 

https://github.com/user-attachments/assets/43714278-6011-4893-bc5b-b06206666e1c

---

# 项目内容 / Project Content
1. Windows 平台部署    
   Windows platform deployment  
2. Jetpack AGX Orin 平台部署    
   Jetpack AGX Orin platform deployment

---

# Windows 部署 / Windows Deployment

## 运行环境 / Environment Setup
参考 IGEVplusplus 文件夹内的 readme.md。    
Please refer to the readme.md inside the IGEVplusplus folder.

推荐使用 CUDA 11.4 及以上、TensorRT 8.5.1 及以上，PyTorch 版本请根据 CUDA 版本选择。    
It is recommended to use CUDA 11.4 or higher, TensorRT 8.5.1 or higher, and the appropriate PyTorch version matching your CUDA version.

首先通过 conda 创建虚拟环境：  
First, create a virtual environment using conda:
```bash
conda create -n IGEV_YOLO python=3.8
```
  
进入虚拟环境  
```bash
conda activate IGEV_YOLO
```

安装 CUDA、PyTorch 和 TensorRT（参考官网选择对应版本）。  
Install CUDA, PyTorch, and TensorRT according to the official recommendations.  
我使用的是 CUDA 11.6、PyTorch 1.13.1、TensorRT 12.6。  
For example, I used CUDA 11.6, PyTorch 1.13.1, and TensorRT 12.6.  

注意：TensorRT 大版本间部分 API 名称不同，我提供了对应的 Python 文件，请根据实际情况选用。  
Note: Some API names differ between major TensorRT versions. I have provided corresponding Python files; please choose the appropriate one based on your setup.  

安装所需依赖库：
Install the required dependencies:  
```bash
pip install tqdm
pip install scipy
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install timm==0.5.4
pip install onnx onnxruntime onnxsim
pip install tensorrt
conda install pycuda
```
## 模型准备 / Model Preparation

你需要准备相应的 YOLO 模型和 IGEV 模型（通常为 .pth 格式的 PyTorch 模型），无论是自行训练还是下载预训练模型（可从项目说明中的仓库下载），但请确保模型可以正常推理。  
You need to prepare the YOLO and IGEV models (usually in .pth format for PyTorch), either by training them yourself or by downloading pre-trained models (available from the project repositories), ensuring they can perform inference correctly.

本部署基于下载好的预训练模型。  
This deployment is based on pre-trained models that have been downloaded.

相关模型也可以从我的 Google Drive 下载：  
Related models can also be downloaded from my Google Drive:
[YOLO模型（ONNX格式）](https://drive.google.com/drive/folders/1jTuoAWUdAMZGFGJIEzNLa7_GrNLIS4ds?usp=sharing)  
[IGEV模型（ONNX格式）](https://drive.google.com/drive/folders/18nu_z_qmnXnhEStgzXqOK7igYER9oS2O?usp=sharing)  

## 模型转换 / Model Conversion

使用 IGEVplusplus/transform_IGEV++.py 将 IGEV++ 的 PyTorch 模型转换为 ONNX 模型；  
Use IGEVplusplus/transform_IGEV++.py to convert the IGEV++ PyTorch model to an ONNX model;

使用 IGEVplusplus/transform_RTIGEV.py 将 RT 版本的 IGEV++ 的 PyTorch 模型转换为 ONNX 模型。  
and use IGEVplusplus/transform_RTIGEV.py to convert the real-time version of the IGEV++ PyTorch model to ONNX.

你需要根据实际情况配置模型路径，找到下面代码片段：  
Configure the model path accordingly. Locate the following snippet in the code:
```python
parser.add_argument('--restore_ckpt', help="put your dir of .pth model here",
default="IGEVplusplus/pretrained_models/igev_plusplus/sceneflow.pth")
```

下图展示了 pth 模型和 ONNX 模型的推理结果：  
The images below show the inference results of the pth model and the ONNX model:

![pth推理结果](https://github.com/user-attachments/assets/1efd537a-4904-42e8-8a7b-94a0fe17c788)
![onnx推理结果](https://github.com/user-attachments/assets/ac33f540-9916-43ac-89cb-d7b342c954eb)  
可以看到转换的精度损失很小，说明转换成功。  
It can be seen that the precision loss is minimal, indicating a successful conversion.

转换完成后得到 ONNX 模型，可以使用 onnxsim 工具简化模型。  
After conversion, you can simplify the ONNX model using onnxsim.

安装相关依赖版本（示例版本如下）：  
Install the required package versions (example versions):
| Package                  | Version | Source   | Channel |
|--------------------------|---------|----------|---------|
| onnx                     | 1.17.0  | pypi     | pypi    |
| onnxoptimizer             | 0.3.13  | pypi     | pypi    |
| onnxruntime-gpu           | 1.19.2  | pypi     | pypi    |
| onnxruntime-tools         | 1.7.0   | pypi     | pypi    |
| onnxsim                   | 0.4.36  | pypi     | pypi    |

使用以下命令简化 ONNX 模型：  
Simplify the ONNX model using:
```bash
onnxsim your_original_model_name.onnx your_simplified_model_name.onnx
```


如果需要使用 TensorRT 加速，请确保安装 TensorRT 8.5.1 及以上版本，否则转换时会遇到不支持的算子 grid_sample。  
If you need TensorRT acceleration, make sure to install TensorRT 8.5.1 or above; otherwise, you may encounter unsupported operator grid_sample during conversion.

确保 CUDA 与 PyTorch 版本匹配，如果没有安装 pycuda，请安装相应版本以避免转换失败。  
Ensure that the CUDA and PyTorch versions are compatible. If pycuda is not installed, install the corresponding version to avoid conversion failures.

首先，根据你解压 TensorRT 的位置添加系统环境变量：  
First, add the following system environment variables according to your TensorRT installation path:
```bash
/your path/TensorRT-XXX/bin
/your path/TensorRT-XXX/lib
/your path/TensorRT-XXX/include
```
运行命令行语句验证  
Verify the installation by running:
```bash
trtexec
```
显示版本号和“passed”则表示安装成功。  
If you see the version information and "passed", the installation is successful.

进入模型所在文件夹，并运行以下命令（请将 “your_name_of_onnx_model” 替换为你的 ONNX 模型名称，“your_name_of_engine” 替换为你想要的 TensorRT 模型名称）：  
Navigate to the folder containing your model and run:
```bash
trtexec --onnx=your_name_of_onnx_model.onnx  --saveEngine=your_name_of_engine.engine --fp16
```

转换完成后，可以使用 igev_yolo_segment.py 在 Windows 上运行推理。  
After conversion, you can run inference on Windows using igev_yolo_segment.py.

注意：你需要一个双目摄像头设备，并调整参数。  
Note: A stereo camera is required, and you should adjust the parameters accordingly.

例如，修改摄像头和模型路径参数：  
For example, modify the camera and model path parameters:
```python
# 摄像头编号，根据实际调整：0/1/2/...
# Camera index, adjust as needed (0/1/2/...):
cap = cv2.VideoCapture(1)

# 设置视频帧大小，根据实际调整
# Set the video frame size:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 摄像头参数，根据实际调整
# Camera parameters, adjust as needed:
focal_length_mm = 3.0
baseline = 60.0
sensor_width_mm = 8.47
image_width = 640
focal_length_pixels = (focal_length_mm * image_width) / sensor_width_mm

# 模型路径修改
# Update your model paths:
'''
这里放模型路径
Put your model paths here
'''
model_path = "your_YOLO.engine"      # PUT YOLO MODEL
model_path_depth = "your_IGEV.engine"  # PUT IGEV MODEL
```

完成后运行该 Python 程序即可。  
After that, run the Python program to perform inference.

# Jetpack 部署 / Jetson (Jetpack) Deployment
## 环境配置 / Environment Setup
我使用的是图为版 Jetson AGX Orin，出厂系统为 Jetpack 5.0.1，预装 TensorRT 8.4 和 CUDA 11.4。  
I used a modified Jetson AGX Orin with the factory system Jetpack 5.0.1, which comes pre-installed with TensorRT 8.4 and CUDA 11.4.

然而 IGEV++ 模型使用 grid_sample 算子，该算子在 TensorRT 8.5 及以上版本才能支持，因此需要升级 TensorRT 版本。  
However, the IGEV++ model uses the grid_sample operator, which is supported only in TensorRT 8.5 and above, so an upgrade is necessary.

当然，你也可以升级 CUDA 和 PyTorch 版本以兼容更高版本的 TensorRT，但最低要求是 TensorRT 8.5。  
Alternatively, you can upgrade CUDA and PyTorch to be compatible with higher versions of TensorRT, but the minimum requirement is TensorRT 8.5.

如果你使用 NVIDIA 原厂的 Jetson 设备，出厂系统版本号以 5 开头及以上的设备可以同时安装两套 CUDA 和 TensorRT 版本，参考 官方指导视频。  
If you are using an official NVIDIA Jetson device (system version starting with 5 or higher), you can install two sets of CUDA and TensorRT versions concurrently; refer to the official tutorial video.

如果你使用的是国内改装版，可能存在硬件锁，无法使用官方文档提供的方式升级系统版本，但依然有解决方法。  
If you are using a domestically modified version, there might be a hardware lock that prevents upgrading the system version using official methods, but there are workarounds.

从 此网址 下载以下三个 deb 文件：  
Download the following three deb files from this website:

    libcudnn8-dev_8.6.0.166-1+cuda11.4_arm64.deb

    libcudnn8-samples_8.6.0.166-1+cuda11.4_arm64.deb

    libcudnn8_8.6.0.166-1+cuda11.4_arm64.deb

卸载原有的 TensorRT，并安装上述版本。  
Uninstall the original TensorRT and install the above versions.

虽然 Jetpack 5.1.x 系统原生支持 TensorRT 8.6，但实践表明 Jetpack 5.0.x 也能正常使用 CUDA 11.4 与 TensorRT 8.6 的组合。  
Although TensorRT 8.6 is natively supported on Jetpack 5.1.x, in practice, the combination of CUDA 11.4 and TensorRT 8.6 works on Jetpack 5.0.x as well.

```bash
sudo dpkg -i libcudnn8_8.6.0.166-1+cuda11.4_arm64.deb
sudo dpkg -i libcudnn8-dev_8.6.0.166-1+cuda11.4_arm64.deb
sudo dpkg -i libcudnn8-samples_8.6.0.166-1+cuda11.4_arm64.deb
```

谨慎更改国内版Jetson的cuda版本，容易导致未知原因的黑屏或无法开机问题，以上方案仅需要更换tensorrt版本，因此推荐使用。    
Be cautious when modifying the CUDA version on modified domestic Jetson devices, as it may cause black screens or boot failures. The above solution only requires upgrading TensorRT, which is recommended.  
如果你从官网  https://developer.nvidia.com/nvidia-tensorrt-8x-download  查找版本，    
会发现ARM SBSA分类下没有cuda 11.4 + tensorrt 8.6 的选择，只能找到至少是cuda 11.8 + tensorrt 8.5的组合,因为我们需要的tensorrt版本最低为8.5。  
笔者尝试后设备无法正常开机，只能刷机。而且国内版无法通过官网方法刷入其他版本的Jetpack，只要不是厂商硬盘刷机，都无法正常开机，怀疑是有硬件锁的原因。    
If you check the NVIDIA TensorRT 8.x download page, you’ll notice that under the ARM SBSA category there is no option for CUDA 11.4 + TensorRT 8.6; the available option is at least CUDA 11.8 + TensorRT 8.5.  
fter my attempts, the device could not boot normally and could only be re-flashed. Moreover, the domestic version cannot be flashed with other versions of Jetpack through the official method—as long as it isn't flashed using the manufacturer's drive, it won't boot normally. I suspect this is due to a hardware lock.  

你需要在Jetpack上先完成miniforge的安装，此类教程网上非常多，此处不再赘述。  
创建虚拟环境后，与windows部署一样，安装相应的库，流程也相同。    
You need to install miniforge on Jetpack first (many tutorials are available online), then create a virtual environment and install the dependencies as in the Windows deployment.

最大的区别在于 Jetpack 使用 tensorrt 时，需要手动加载插件，相应代码在对应的 igev_seg_Jetpack.py 中已经加入。   
The main difference on Jetpack is that when using TensorRT, you need to manually load the plugins. This is already included in igev_seg_Jetpack.py
```python
import ctypes

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# 手动加载插件库
ctypes.CDLL("/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)

# 注册所有插件
trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
```
请按照流程在Jetpack上生成tensorrt模型后，修改 igev_seg_Jetpack.py 中相应的参数，如模型路径，摄像头参数，完成后运行这个python文件即可。    
After generating the TensorRT engine on Jetpack, modify the corresponding parameters (e.g., model paths, camera settings) in igev_seg_Jetpack.py and then run the Python script.  

igev_seg_Jetpack_ros2.py 在 igev_seg_Jetpack.py 的基础上，增加了使用ros2发送处理结果的功能，可后续用于复合项目的总线通信。   
igev_seg_Jetpack_ros2.py builds upon igev_seg_Jetpack.py by adding ROS2 functionality for inter-module communication in more complex projects.  
