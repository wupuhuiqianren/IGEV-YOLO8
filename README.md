# IGEV-YOLO8
Conbine IGEV++ with YOLO8 segment，used for auto driving occasions

# 项目说明
这个项目基于 [YOLOv8](https://github.com/ultralytics/ultralytics) 和 [IGEV++](https://github.com/gangweiX/IGEV-plusplus)。

# 项目内容
1.Windows 平台部署
2.Jetpack AGX Orin 平台部署

# 运行环境
参考IGEVplusplus文件夹内的readme.md

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





