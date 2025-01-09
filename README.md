# IGEV-YOLO8
Conbine IGEV++ with YOLO8 segment，used for auto driving occasions

# 项目说明
这个项目基于 [YOLOv8](https://github.com/ultralytics/ultralytics) 和 [IGEV++](https://github.com/gangweiX/IGEV-plusplus)。

# 项目内容
1.Windows 平台部署
2.Jetpack AGX Orin 平台部署

# 运行环境
参考IGEVplusplus文件夹内的readme.md
推荐使用cuda11.4及以上,tensorrt8.5.1及以上，pytorch版本根据cuda对应
```bash
pip install tqdm
pip install scipy
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install timm==0.5.4
```
# 模型准备
你需要先准备相应的YOLO模型和IGEV模型（通常是.pth格式的pytorch模型），无论是自己训练或者下载预训练模型（可以从【项目说明】中的仓库下载），但请确保你的模型可以正常完成推理。本部署基于下载好的预训练模型。
相关模型也可以从我的google drive下载
[YOLO模型（ONNX格式）](https://drive.google.com/drive/folders/1jTuoAWUdAMZGFGJIEzNLa7_GrNLIS4ds?usp=sharing)

[IGEV模型（ONNX格式）](https://drive.google.com/drive/folders/18nu_z_qmnXnhEStgzXqOK7igYER9oS2O?usp=sharing)

# 模型转换
IGEVplusplus/transform_IGEV++.py 用于转换IGEV++的pytorch模型为ONNX模型
IGEVplusplus/transform_RTIGEV.py 用于转换rt版本的IGEV++的pytorch模型为ONNX模型

你需要根据实际情况配置你的模型路径。
```python
parser.add_argument('--restore_ckpt', help="put your dir of .pth model here",
default="IGEVplusplus/pretrained_models/igev_plusplus/sceneflow.pth")
```
转换完成后，可以使用onnxsim工具简化模型，你需要先安装相关的库，这是我使用的版本。
| Package                  | Version | Source   | Channel |
|--------------------------|---------|----------|---------|
| onnx                     | 1.17.0  | pypi     | pypi    |
| onnxoptimizer             | 0.3.13  | pypi     | pypi    |
| onnxruntime-gpu           | 1.19.2  | pypi     | pypi    |
| onnxruntime-tools         | 1.7.0   | pypi     | pypi    |
| onnxsim                   | 0.4.36  | pypi     | pypi    |


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
注意，你首先需要一个双目摄像头设备，并调整参数，以及模型路径
```python
    # 摄像头参数
    focal_length_mm = 3.0
    baseline = 60.0
    sensor_width_mm = 8.47
    image_width = 640
    focal_length_pixels = (focal_length_mm * image_width) / sensor_width_mm
```
```python
    '''
    这里放模型路径
    '''
    model_path = "your.engine" #YOLO MODEL
    model_path_depth = "your.engine" #IGEV MODEL
```
完成后运行这个python程序。

