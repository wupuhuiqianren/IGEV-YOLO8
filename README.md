# IGEV-YOLO8
Conbine IGEV++ with YOLO8 segment，used for auto driving occasions

# 项目说明
这个项目基于 [YOLOv8](https://github.com/ultralytics/ultralytics) 和 [IGEV++](https://github.com/gangweiX/IGEV-plusplus)。  
此项目融合了两种深度学习模型作为一个系统，实现YOLO进行目标检测和分割，IGEV++提取被检测目标与摄像头之间的距离信息，最后一起显示在摄像头输出画面上的功能。

简单的效果展示，项目已部署在Jetson AGX Orin上

https://github.com/user-attachments/assets/43714278-6011-4893-bc5b-b06206666e1c

# 项目内容
1.Windows 平台部署   
2.Jetpack AGX Orin 平台部署  

# Windows 部署

## 运行环境
参考IGEVplusplus文件夹内的readme.md  
推荐使用cuda11.4及以上,tensorrt8.5.1及以上，pytorch版本根据cuda对应  

首先通过conda创建虚拟环境  
```bash
conda create -n IGEV_YOLO python=3.8
```
进入虚拟环境  
```bash
conda activate IGEV_YOLO
```
安装cuda和pytorch和tensorrt，参考官网选择对应版本。  
我使用的是cuda 11.6,pytorch 1.13.1,tensorrt 12.6  

注意：tensorrt大版本间部分api名称不同，我提供了对应的python文件，请根据实际情况选用。

安装所需要的依赖库  
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
## 模型准备
你需要先准备相应的YOLO模型和IGEV模型（通常是.pth格式的pytorch模型），无论是自己训练或者下载预训练模型（可以从【项目说明】中的仓库下载），但请确保你的模型可以正常完成推理。  
本部署基于下载好的预训练模型。  
相关模型也可以从我的google drive下载  
[YOLO模型（ONNX格式）](https://drive.google.com/drive/folders/1jTuoAWUdAMZGFGJIEzNLa7_GrNLIS4ds?usp=sharing)  
[IGEV模型（ONNX格式）](https://drive.google.com/drive/folders/18nu_z_qmnXnhEStgzXqOK7igYER9oS2O?usp=sharing)  

## 模型转换
IGEVplusplus/transform_IGEV++.py 用于转换IGEV++的pytorch模型为ONNX模型  
IGEVplusplus/transform_RTIGEV.py 用于转换rt版本的IGEV++的pytorch模型为ONNX模型  
你需要根据实际情况配置你的模型路径。找到代码中下面的片段：  
```python
parser.add_argument('--restore_ckpt', help="put your dir of .pth model here",
default="IGEVplusplus/pretrained_models/igev_plusplus/sceneflow.pth")
```

![pth推理结果](https://github.com/user-attachments/assets/1efd537a-4904-42e8-8a7b-94a0fe17c788)
![onnx推理结果](https://github.com/user-attachments/assets/ac33f540-9916-43ac-89cb-d7b342c954eb)  
可以看到转换的精度损失是很小的，说明转换成功。

转换完成得到ONNX模型后，可以使用onnxsim工具简化模型，你需要先安装相关的库，这是我使用的版本。
| Package                  | Version | Source   | Channel |
|--------------------------|---------|----------|---------|
| onnx                     | 1.17.0  | pypi     | pypi    |
| onnxoptimizer             | 0.3.13  | pypi     | pypi    |
| onnxruntime-gpu           | 1.19.2  | pypi     | pypi    |
| onnxruntime-tools         | 1.7.0   | pypi     | pypi    |
| onnxsim                   | 0.4.36  | pypi     | pypi    |

使用以下命令行语句简化onnx模型
```bash
onnxsim your_original_model_name.onnx your_simplified_model_name.onnx
```


如果你需要使用Tensorrt加速，务必安装8.5.1版本及以上，否则转换模型时会遇到不支持的算子grid_sample。确保你的cuda,pytorch版本匹配，如果没有安装pycuda，务必安装相应的版本，否则也可能造成转换失败。  
首先根据你解压Tensorrt的位置，添加系统环境变量：
```bash
/your path/TensorRT-XXX/bin
/your path/TensorRT-XXX/lib
/your path/TensorRT-XXX/include
```
运行命令行语句验证
```bash
trtexec
```
出现版本号等信息，显示"passed"即为安装成功。

命令行cd语句进入模型所在文件夹，运行命令行语句。请更换“your_name_of_onnx_model”为自己的onnx模型名称，更换“your_name_of_engine”为你想要的tensorrt模型的名称
```bash
trtexec --onnx=your_name_of_onnx_model.onnx  --saveEngine=your_name_of_engine.engine --fp16
```

转换完成后，你可以使用 igev_yolo_segment.py 在windows上运行推理。  
注意，你首先需要一个双目摄像头设备，并调整参数。  
```python
    # 摄像头编号，根据实际调整：0/1/2/...
    cap = cv2.VideoCapture(1)

    # 设置视频帧大小，根据实际调整
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 摄像头参数，根据实际调整
    focal_length_mm = 3.0
    baseline = 60.0
    sensor_width_mm = 8.47
    image_width = 640
    focal_length_pixels = (focal_length_mm * image_width) / sensor_width_mm
``` 
改为你的模型路径
```python
    '''
    这里放模型路径
    '''
    model_path = "your_YOLO.engine" #PUT YOLO MODEL
    model_path_depth = "your_IGEV.engine" #PUT IGEV MODEL
```

完成后运行这个python程序。

# Jetpack部署
## 环境配置
我使用的是图为版Jetson AGX Orin, 出厂系统Jetpack 5.0.1，出厂自带tensorrt 8.4版本,cuda 11.4版本。  
然而igev++模型使用了grid_sample算子，这一算子在tensorrt 8.5版本及以上才能够支持，因此需要升级tensorrt版本。
当然你也可以升级cuda和pytorch版本，来兼容更高的tensorrt版本。最低要求是tensorrt 8.5。

如果你使用英伟达原厂的Jetson设备，那么出厂系统编号为5开头及以上的设备，可以同时安装两套cuda和tensorrt版本，  
参考[官方指导视频](https://www.youtube.com/watch?v=_JgNA82325I&t)  
 
如果你使用的是和我一样的国内改装版，则大概率存在硬件锁，无法使用官方文档提供的方式升级系统版本，但是依然有办法解决，下面会进行说明。
从[这个网址](https://repo.download.nvidia.cn/jetson/#Jetpack%205.1)搜索并下载以下三个deb:  

libcudnn8-dev_8.6.0.166-1+cuda11.4_arm64.deb  
libcudnn8-samples_8.6.0.166-1+cuda11.4_arm64.deb  
libcudnn8_8.6.0.166-1+cuda11.4_arm64.deb  

卸载原来的tensorrt，安装此版本。虽然Jetpack 5.1.x系统才对tensorrt 8.6支持，但是经过实践发现Jetpack 5.0.x版本可以正常使用这一套cuda11.4和tensorrt 8.6的组合。

```bash
sudo dpkg -i libcudnn8_8.6.0.166-1+cuda11.4_arm64.deb
sudo dpkg -i libcudnn8-dev_8.6.0.166-1+cuda11.4_arm64.deb
sudo dpkg -i libcudnn8-samples_8.6.0.166-1+cuda11.4_arm64.deb
```

谨慎更改国内版Jetson的cuda版本，容易导致未知原因的黑屏或无法开机问题，以上方案仅需要更换tensorrt版本，因此推荐使用。 
如果你从官网  https://developer.nvidia.com/nvidia-tensorrt-8x-download  查找版本，  
会发现ARM SBSA分类下没有cuda 11.4 + tensorrt 8.6 的选择，只能找到至少是cuda 11.8 + tensorrt 8.5的组合,因为我们需要的tensorrt版本最低为8.5。
笔者尝试后设备无法正常开机，只能刷机。而且国内版无法通过官网方法刷入其他版本的Jetpack，只要不是厂商硬盘刷机，都无法正常开机，怀疑是有硬件锁的原因。

你需要在Jetpack上先完成miniforge的安装，此类教程网上非常多，此处不再赘述。
创建虚拟环境后，与windows部署一样，安装相应的库，流程也相同。

最大的区别在于 Jetpack 使用 tensorrt 时，需要手动加载插件，相应代码在对应的 igev_seg_Jetpack.py 中已经加入。
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
igev_seg_Jetpack_ros2.py 在 igev_seg_Jetpack.py 的基础上，增加了使用ros2发送处理结果的功能，可后续用于复合项目的总线通信。
