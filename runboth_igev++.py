import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from ultralytics import YOLO
import time

# 定义视差图的可视化函数
def visualize_disparity(disp, title="Disparity Map"):
    """
    将视差图显示为伪彩色图像
    :param disp: 输入视差图 (2D numpy array)
    :param title: 显示窗口的标题
    """
    disp_normalized = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_colored = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_VIRIDIS)
    cv2.imshow(title, disp_colored)

# 载入 TensorRT 引擎
engine_file = "IGEVplusplus/igev_fp16.engine"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(engine_file, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

# 创建执行上下文
context = engine.create_execution_context()

# 获取输入和输出的绑定信息
num_bindings = engine.num_io_tensors
inputs = []
outputs = []
bindings = []
for binding_index in range(num_bindings):
    binding_name = engine.get_tensor_name(binding_index)
    dtype = engine.get_tensor_dtype(binding_name)
    shape = engine.get_tensor_shape(binding_name)
    size = trt.volume(shape) * np.dtype(np.float32).itemsize
    host_mem = np.empty(shape, dtype=np.float32)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    if engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
        inputs.append({"host": host_mem, "device": device_mem})
    else:
        outputs.append({"host": host_mem, "device": device_mem})
    bindings.append(int(device_mem))

# 预处理函数
def preprocess_image(image):
    image_input = np.transpose(image, (2, 0, 1))[None, :, :, :].astype(np.float32)
    return np.ascontiguousarray(image_input)

# 计算深度函数
def calculate_depth(disparity, focal_length_pixels, baseline):
    disparity[disparity == 0] = 0.1
    depth = (focal_length_pixels * baseline) / disparity
    return depth * 2.1

def calculate_selected_depth(depth_map, x1, y1, x2, y2):
    region_width = x2 - x1
    region_height = y2 - y1
    if region_width < 5 or region_height < 5:
        return -1
    region = depth_map[y1:y2, x1:x2]
    center_x = region_width // 2
    center_y = region_height // 2
    half_window = 2
    start_y = max(center_y - half_window, 0)
    end_y = min(center_y + half_window + 1, region.shape[0])
    start_x = max(center_x - half_window, 0)
    end_x = min(center_x + half_window + 1, region.shape[1])
    center_region = region[start_y:end_y, start_x:end_x]
    depths = center_region.flatten()
    median_depth = np.median(depths)
    median_depth = median_depth/1000
    std_depth = np.std(depths)
    valid_depths = depths[(depths >= median_depth - 1.5 * std_depth) & (depths <= median_depth + 1.5 * std_depth)]
    valid_depths = valid_depths/1000
    return np.mean(valid_depths) if valid_depths.size > 0 else median_depth

# 加载 YOLOv8 模型
model = YOLO("YOLO/yolov8x.engine", task="detect")

# 打开摄像头设备
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("无法打开摄像头，请检查设备索引")
    exit()

# 设置视频帧大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 摄像头参数
focal_length_mm = 3.0
baseline = 60.0
sensor_width_mm = 8.47
image_width = 640
focal_length_pixels = (focal_length_mm * image_width) / sensor_width_mm

prev_disp_pred = None

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 获取左右图像
        left_image = frame[:, :640]
        right_image = frame[:, 640:]

        # 预处理图像
        left_image_input = preprocess_image(left_image)
        right_image_input = preprocess_image(right_image)

        # 推理处理
        cuda.memcpy_htod(inputs[0]["device"], left_image_input)
        cuda.memcpy_htod(inputs[1]["device"], right_image_input)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(outputs[0]["host"], outputs[0]["device"])

        # 获取视差预测
        disp_pred = outputs[0]["host"].reshape(480, 640)  # 根据实际模型输出调整形状

        disp_pred = cv2.medianBlur(disp_pred.astype(np.uint8), 5)
        disp_pred = cv2.GaussianBlur(disp_pred, (5, 5), 0)

        if prev_disp_pred is not None:
            disp_pred = 0.5 * prev_disp_pred + 0.5 * disp_pred
        prev_disp_pred = disp_pred

        # 可视化视差图
        visualize_disparity(disp_pred, title="TensorRT Disparity Map")

        # 计算深度图
        depth_map = calculate_depth(disp_pred, focal_length_pixels, baseline)

        # YOLOv8 目标检测
        results = model.predict(left_image, show_conf=False)
        boxes = results[0].boxes

        # 在左图上绘制目标框和信息
        left_image_display = left_image.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            selected_depth = calculate_selected_depth(depth_map, x1, y1, x2, y2)
            # 绘制目标框
            cv2.rectangle(left_image_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 显示目标名称
            class_id = int(box.cls[0])
            class_name = model.names[class_id]  # 获取目标名称

            # 显示距离信息
            text_position = (x1, y1 - 10)
            label = f"{class_name} Dis: {selected_depth:.2f} m" if selected_depth is not None else f"{class_name} Dis: N/A"
            cv2.putText(left_image_display, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 显示带标注的左视图
        cv2.imshow("Left View with Detections", left_image_display)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
