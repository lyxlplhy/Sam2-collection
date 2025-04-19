import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time
import gc
from ultralytics import YOLO
import torch
import argparse

def yolo_model(checkpoint):
    yolo = YOLO(checkpoint)

    print(yolo)
    return yolo

class Encoder:
    def __init__(self, engine_path, input_shape=(1, 3, 1024, 1024)):
        self.input_shape = input_shape
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_index = self.engine.get_binding_index(self.engine.get_binding_name(0))
        self.context.set_binding_shape(self.input_index, self.input_shape)

        self.input_data = np.empty(self.input_shape, dtype=np.float32)
        self.input_device = cuda.mem_alloc(self.input_data.nbytes)
        self.bindings = [None] * self.engine.num_bindings
        self.bindings[self.input_index] = int(self.input_device)

        # 输出
        self.output_data = []
        self.output_devices = []
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                output_shape = self.context.get_binding_shape(i)
                output_dtype = trt.nptype(self.engine.get_binding_dtype(i))
                host_buf = np.empty(output_shape, dtype=output_dtype)
                device_buf = cuda.mem_alloc(host_buf.nbytes)
                self.bindings[i] = int(device_buf)
                self.output_data.append(host_buf)
                self.output_devices.append(device_buf)

        self.stream = cuda.Stream()
    def process(self,img):
        img = cv2.resize(img, (1024, 1024))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        img = (img - mean) / std
        img = np.ascontiguousarray(img, dtype=np.float32)
        return img

    def infer(self, img):
        cuda.memcpy_htod(self.input_device, img)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for i in range(len(self.output_data)):
            cuda.memcpy_dtoh_async(self.output_data[i], self.output_devices[i], self.stream)
        self.stream.synchronize()
        return self.output_data


class Decoder:
    def __init__(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_shapes = [
            (1, 256, 64, 64),
            (1, 32, 256, 256),
            (1, 64, 128, 128),
            (1, 2, 2),
            (1, 2),
            (1, 1, 256, 256),
            (1,),
            (2,)
        ]

        self.input_devices = []
        self.input_indices = []
        self.bindings = [None] * self.engine.num_bindings

        # 初始化输入绑定
        for i, shape in enumerate(self.input_shapes):
            idx = self.engine.get_binding_index(self.engine.get_binding_name(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            self.context.set_binding_shape(idx, shape)
            device_input = cuda.mem_alloc(np.empty(shape, dtype=dtype).nbytes)
            self.bindings[idx] = int(device_input)
            self.input_devices.append(device_input)
            self.input_indices.append(idx)

        # 初始化输出绑定
        self.output_data = []
        self.output_devices = []
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                shape = self.context.get_binding_shape(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                host_out = np.empty(shape, dtype=dtype)
                device_out = cuda.mem_alloc(host_out.nbytes)
                self.bindings[i] = int(device_out)
                self.output_data.append(host_out)
                self.output_devices.append(device_out)

        self.stream = cuda.Stream()

    def infer(self, inputs):
        # 每次推理只拷贝输入数据
        for i, np_input in enumerate(inputs):
            dtype = trt.nptype(self.engine.get_binding_dtype(self.input_indices[i]))
            np_input = np_input.astype(dtype)
            cuda.memcpy_htod(self.input_devices[i], np_input)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for i in range(len(self.output_data)):
            cuda.memcpy_dtoh_async(self.output_data[i], self.output_devices[i], self.stream)

        self.stream.synchronize()



        return self.output_data


def postprocess_to_mask(img, output, save_path, ori_height, ori_width):
    output = output.squeeze()
    resize_output = cv2.resize(output, (ori_width, ori_height))
    sigmoid_output = 1.0 / (1.0 + np.exp(-resize_output))
    thresholded = np.uint8(np.clip(sigmoid_output * 255, 0, 255))

    # 二值图像用于提取轮廓
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 确保原图是RGB格式，避免灰度图或BGR混用
    if len(img.shape) == 2 or img.shape[2] == 1:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    # 画轮廓，颜色为红色，粗细为2
    cv2.drawContours(img_color, contours, -1, (0, 0, 255), 2)

    # 保存带有轮廓的原图
    cv2.imwrite(save_path, img_color)

    return img_color


# =============== 主流程 ===============
encoder = Encoder(r"D:\sam2\segment-anything-2-main\tools\tensorrt_model\conver_tiny_encoder.engine")
decoder = Decoder("D:/sam2/segment-anything-2-main/tools/tensorrt_model/conver_tiny_decoder.engine")

input_folder = r"E:\LYX_date\堆叠图\1"
output_folder = r"E:\LYX_date\deve_obb\images\output_masks"
os.makedirs(output_folder, exist_ok=True)
yolo_conver = yolo_model("E:\era5/best.pt")


for img_name in os.listdir(input_folder):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)
    #####################YOLO#####################

    xx1=829
    yy1=1013
    xx2=1365
    yy2=1540
    #####################YOLO#####################

    ori_img=img
    if img is None:
        continue
    h, w = img.shape[:2]
    img=encoder.process(img)
    time1 = time.time()
    encoder_outputs = encoder.infer(img)
    time2 = time.time()
    input_data1 = encoder_outputs[0]
    input_data2 = encoder_outputs[2]
    input_data3 = encoder_outputs[1]
    input_data4 = np.array([[[xx1 * 1024 / w, yy1 * 1024 / h, xx2 * 1024 / w, yy2 * 1024 / h]]], dtype=np.float32)
    input_data5 = np.array([[2, 3]], dtype=np.float32)
    input_data6 = np.zeros((1, 1, 256, 256), dtype=np.float32)
    input_data7 = np.array([0], dtype=np.float32)
    input_data8 = np.array([w, h], dtype=np.int32)
    time3 = time.time()
    decoder_outputs = decoder.infer([
        input_data1, input_data2, input_data3,
        input_data4, input_data5, input_data6,
        input_data7, input_data8
    ])

    time4 = time.time()
    print("time", time2 - time1 + time4 - time3)
    print("encoder",time2-time1)
    save_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_mask.jpg")
    postprocess_to_mask(ori_img,decoder_outputs[1], save_path, h, w)


