import time

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 6: 'train', 7: 'truck',
           9: 'traffic light'}

class Colors:

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class YOLOv8Seg:
    """YOLOv8 segmentation model."""

    def __init__(self, engine_file):

        # Load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context
        self.context = self.engine.create_execution_context()
        self.num_bindings = self.engine.num_bindings
        # Get model width and height (YOLOv8-seg only has one input)
        self.model_height, self.model_width = self.engine.get_binding_shape("images")[-2:]

        # Allocate memory for inputs and outputs
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding_index in range(self.num_bindings):
            binding_name = self.engine.get_binding_name(binding_index)
            dtype = self.engine.get_binding_dtype(binding_name)
            shape = self.engine.get_binding_shape(binding_name)
            size = trt.volume(shape) * np.dtype(np.float32).itemsize
            host_mem = np.empty(shape, dtype=np.float32)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            if self.engine.binding_is_input(binding_name):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})
            self.bindings.append(int(device_mem))

        # Numpy dtype: support both FP32 and FP16 TensorRT model
        self.ndtype = np.float16 if self.engine.get_binding_dtype(binding_name) == trt.float16 else np.float32

        # Load COCO class names
        self.classes = classes

        # Create color palette
        self.color_palette = Colors()

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):

        # 预处理
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # 确保输入数组正确赋值到 host 内存
        np.copyto(self.inputs[0]['host'], im)

        # 将 host 数据复制到设备
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])

        # 执行推理
        if not self.context.execute_v2(bindings=self.bindings):
            raise RuntimeError("[ERROR] Inference execution failed.")

        # 从设备复制输出到 host
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        cuda.memcpy_dtoh(self.outputs[1]['host'], self.outputs[1]['device'])

        # 解析输出
        preds = [
            self.outputs[0]['host'].reshape(self.engine.get_binding_shape(self.engine.get_binding_name(1))),
            self.outputs[1]['host'].reshape(self.engine.get_binding_shape(self.engine.get_binding_name(2)))
        ]

        # 后处理
        boxes, segments, masks = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm
        )

        return boxes, segments, masks

    # YOLO相关函数，预处理，后处理
    def preprocess(self, img):
        '''
        YOLO预处理
        :param img:
        :return:
        '''
        # 原始图像形状
        shape = img.shape[:2]

        # 模型输入大小
        new_shape = (self.model_height, self.model_width)

        # 计算缩放比例和填充
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        # 调整大小
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 添加边框
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=(114, 114, 114))

        # 转换为 CHW 格式，并归一化
        img_chw = np.ascontiguousarray(np.einsum('HWC->CHW', img_padded)[::-1], dtype=self.ndtype) / 255.0
        img_batch = img_chw[None]  # 增加批次维度
        return img_batch, ratio, (pad_w, pad_h)

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):

        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum('bcn->bnc', x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]
        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # 过滤掉不在预定义 classes 字典中的类别
        valid_indices = [i for i, cls in enumerate(x[:, 5]) if cls in self.classes.keys()]
        x = x[valid_indices]

        # Decode and return
        if len(x) > 0:

            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)

            # Masks -> Segments(contours)
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []

    @staticmethod
    def masks2segments(masks):

        segments = []
        for x in masks.astype('uint8'):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype('float32'))
        return segments

    @staticmethod
    def crop_mask(masks, boxes):

        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):

        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum('HWN -> NHW', masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):

        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]),
                           interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    def draw_and_visualize(self, im, bboxes, segments, vis=True, save=False):

        # Draw rectangles and polygons
        im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            # draw contour and fill mask
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 1)  # white borderline
            cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))

            # draw bbox rectangle
            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          self.color_palette(int(cls_), bgr=True), 1, cv2.LINE_AA)
            cv2.putText(im, f'{self.classes[cls_]}', (int(box[0]), int(box[1] - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_palette(int(cls_), bgr=True), 1, cv2.LINE_AA)

        # Mix image
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
        return im

class DepthEstimationModel:
    """深度估计模型类"""

    def __init__(self, engine_file):
        # 初始化TensorRT引擎
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # 创建执行上下文
        self.context = self.engine.create_execution_context()

        # 获取输入和输出的绑定信息
        self.num_bindings = self.engine.num_io_tensors
        self.inputs = []
        self.outputs = []
        self.bindings = []
        for binding_index in range(self.num_bindings):
            binding_name = self.engine.get_binding_name(binding_index)
            dtype = self.engine.get_binding_dtype(binding_name)
            shape = self.engine.get_binding_shape(binding_name)
            size = trt.volume(shape) * np.dtype(np.float32).itemsize
            host_mem = np.empty(shape, dtype=np.float32)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            if self.engine.get_binding_mode(binding_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})
            self.bindings.append(int(device_mem))

    def estimate_depth(self, left_image, right_image):
        """
        进行深度估计推理，返回视差图（深度图）。
        :param left_image: 左图像
        :param right_image: 右图像
        :return: 视差图（深度图）
        """
        # 预处理图像
        left_image_input = preprocess_image(left_image)
        right_image_input = preprocess_image(right_image)

        # 将输入数据复制到设备
        cuda.memcpy_htod(self.inputs[0]["device"], left_image_input)
        cuda.memcpy_htod(self.inputs[1]["device"], right_image_input)

        # 执行推理
        if not self.context.execute_v2(bindings=self.bindings):
            raise RuntimeError("[ERROR] Inference execution failed.")

        # 从设备复制输出到主机
        cuda.memcpy_dtoh(self.outputs[0]["host"], self.outputs[0]["device"])

        # 获取视差预测并返回
        disp_pred = self.outputs[0]["host"].reshape(480, 640)  # 根据实际模型输出调整形状
        disp_pred = cv2.resize(disp_pred, (1280, 720))
        return disp_pred

#深度模型预处理，后处理函数
# 预处理函数
def preprocess_image(image):
    image = cv2.resize(image, (640, 480))
    image_input = np.transpose(image, (2, 0, 1))[None, :, :, :].astype(np.float32)
    return np.ascontiguousarray(image_input)

# 计算深度函数
def calculate_depth(disparity, focal_length_pixels, baseline):
    # 确保视差数组为浮点类型
    disparity = disparity.astype(np.float32)
    # 替换无效值
    disparity[np.isnan(disparity) | (disparity <= 0)] = 0.01
    depth = (focal_length_pixels * baseline) / disparity
    return depth * 2.8

def visualize_disparity(disp, title="Disparity Map"):
    """
    将视差图显示为伪彩色图像
    :param disp: 输入视差图 (2D numpy array)
    :param title: 显示窗口的标题
    """
    disp_normalized = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_colored = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
    cv2.imshow(title, disp_colored)

def process_segments_depth(depth_map, segment):
    try:
        if len(segment.shape) != 2 or segment.shape[1] != 2:
            if len(segment.shape) == 1 and segment.shape[0] == 2:
                segment = segment.reshape(1, 2)
            else:
                print(f"Invalid segment. Shape: {segment.shape}")
                return None
        segment = segment.astype(np.int32)
        mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [segment], 1)

        # 找到 mask 范围的坐标
        coords = np.column_stack(np.where(mask == 1))
        if coords.size == 0:
            return None

        # 计算中心范围（50%）
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        x_center_min = x_min + (x_max - x_min) * 0.3
        x_center_max = x_min + (x_max - x_min) * 0.7
        y_center_min = y_min + (y_max - y_min) * 0.3
        y_center_max = y_min + (y_max - y_min) * 0.7

        # 筛选中心范围内的坐标
        center_coords = coords[
            (coords[:, 0] >= x_center_min) & (coords[:, 0] <= x_center_max) &
            (coords[:, 1] >= y_center_min) & (coords[:, 1] <= y_center_max)
        ]

        if center_coords.size == 0:
            return None

        # 获取中心范围的深度值
        segment_depths = depth_map[center_coords[:, 0], center_coords[:, 1]]

        # 计算平均值
        aveg_depth = np.average(segment_depths)

        # 筛选偏差小于20%的值
        valid_depths = [depth for depth in segment_depths if abs(depth - aveg_depth) <= 0.20 * aveg_depth]

        if valid_depths:
            # 返回有效值的平均值
            return np.average(valid_depths)
        else:
            # 如果没有有效值，返回中值作为备用
            print("no valid depth")
            return np.median(segment_depths)
    except Exception as e:
        print(f"Error processing segment: {e}")
        return None



if __name__ == '__main__':
    '''
    这里放模型路径
    '''
    model_path = "your_YOLO.engine" #PUT YOLO MODEL
    model_path_depth = "your_IGEV.engine" #PUT IGEV MODEL
    # 实例化模型
    model = YOLOv8Seg(model_path)
    model_depth = DepthEstimationModel(model_path_depth)
    conf = 0.35
    iou = 0.45

    # 摄像头图像分割
    cap = cv2.VideoCapture(1)
    # 设置视频帧大小
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 摄像头参数
    focal_length_mm = 3.0 #焦距
    baseline = 60.0 #基线
    sensor_width_mm = 8.47 #感光芯片宽度
    image_width = 640 #图像宽度
    #focal_length_pixels = (focal_length_mm * image_width) / sensor_width_mm #计算像素焦距
    focal_length_pixels = 450 #像素焦距，如果直接可知

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # 获取左右图像
            left_image = frame[:, :1280]
            right_image = frame[:, 1280:]

            #深度模型推理
            disp_pred = model_depth.estimate_depth(left_image, right_image)

            disp_pred = cv2.medianBlur(disp_pred.astype(np.uint8), 5)
            disp_pred = cv2.GaussianBlur(disp_pred, (5, 5), 0)

            '''
            可视化视差图，可选
            '''
            #visualize_disparity(disp_pred, title="TensorRT Disparity Map")

            depth_map = calculate_depth(disp_pred, focal_length_pixels, baseline)

            # YOLO推理
            boxes, segments, _ = model(left_image, conf_threshold=conf, iou_threshold=iou)

            # 计算每一个 segment 的深度值
            segment_depths = []
            for segment in segments:
                depth = process_segments_depth(depth_map, segment)
                if depth is not None:
                    segment_depths.append(depth)

            # 画图
            if len(boxes) > 0:
                output_image = model.draw_and_visualize(left_image, boxes, segments, vis=False, save=False)
                for i, (box, depth) in enumerate(zip(boxes, segment_depths)):
                    x1, y1 = int(box[0]), int(box[1])
                    depth /= 1000
                    depth_text = f"{depth:.1f} m"
                    cv2.putText(output_image, depth_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                output_image = left_image

            # 显示图像
            cv2.imshow('seg', output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

