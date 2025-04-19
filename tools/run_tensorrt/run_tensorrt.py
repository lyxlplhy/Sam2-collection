import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time



def run_inference(img):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    ENGINE_PATH = r"D:/sam2/segment-anything-2-main/tools/tensorrt_model/conver_tiny_encoder3.engine"  # 替换为你的模型路径
    with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    input_shape = (1, 3, 1024, 1024)
    input_index = engine.get_binding_index(engine.get_binding_name(0))
    context.set_binding_shape(input_index, input_shape)

    img = cv2.resize(img, (1024, 1024))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    input_data = (img - mean) / std
    input_data = np.ascontiguousarray(input_data, dtype=np.float32)

    # 分配输入内存
    input_device = cuda.mem_alloc(input_data.nbytes)
    cuda.memcpy_htod(input_device, input_data)
    # 绑定输入  和输出
    time1 = time.time()
    bindings = [None] * engine.num_bindings
    bindings[input_index] = int(input_device)
    output_data = []
    output_devices = []
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            output_shape = context.get_binding_shape(i)
            output_dtype = trt.nptype(engine.get_binding_dtype(i))
            host_buf = np.empty(output_shape, dtype=output_dtype)
            device_buf = cuda.mem_alloc(host_buf.nbytes)
            bindings[i] = int(device_buf)
            output_data.append(host_buf)
            output_devices.append(device_buf)
    stream = cuda.Stream()

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    time2 = time.time()
    print("encoder", time2 - time1)


    for i in range(len(output_data)):
        cuda.memcpy_dtoh_async(output_data[i], output_devices[i], stream)
    stream.synchronize()

    return output_data, output_devices


def run_decoder_inference(input_data1, input_data2, input_data3, input_data4, input_data5, input_data6, input_data7,
                          input_data8, w, h):
    engine_file = r"D:\sam2\segment-anything-2-main\tools\tensorrt_model\conver_tiny_decoder.engine"  # 替换为你的模型路径

    # ========== 初始化 TRT 引擎 ==========
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # ========== 输入数据准备 ==========
    input_shapes = [
        (1, 256, 64, 64),  # 输入 1
        (1, 32, 256, 256),  # 输入 2
        (1, 64, 128, 128),  # 输入 3
        (1, 2, 2),  # 输入 4
        (1, 2),  # 输入 5
        (1, 1, 256, 256),  # 输入 6
        (1,),  # 输入 7
        (2,),
    ]

    bindings = [None] * engine.num_bindings
    input_devices = []

    # 将输入数据按顺序放入一个列表中
    input_data_list = [input_data1, input_data2, input_data3, input_data4, input_data5, input_data6, input_data7,
                       input_data8]

    # 为每个输入分配内存并复制数据
    for i, shape in enumerate(input_shapes):
        idx = engine.get_binding_index(engine.get_binding_name(i))
        context.set_binding_shape(idx, shape)
        dtype = trt.nptype(engine.get_binding_dtype(idx))

        # 获取对应的输入数据
        np_input = input_data_list[i].astype(dtype)

        # 将数据传输到 GPU
        device_input = cuda.mem_alloc(np_input.nbytes)
        cuda.memcpy_htod(device_input, np_input)
        bindings[idx] = int(device_input)
        input_devices.append(device_input)
    # ========== 输出数据准备 ==========
    output_data = []
    output_devices = []
    time1 = time.time()
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            output_shape = context.get_binding_shape(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            host_output = np.empty(output_shape, dtype=dtype)
            device_output = cuda.mem_alloc(host_output.nbytes)
            bindings[i] = int(device_output)
            output_data.append(host_output)
            output_devices.append(device_output)

    # ========== 执行推理 ==========

    stream = cuda.Stream()

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    time2 = time.time()
    print("decoder", time2 - time1)
    # ========== 拷贝输出数据 ==========
    for i in range(len(output_data)):
        cuda.memcpy_dtoh_async(output_data[i], output_devices[i], stream)

    stream.synchronize()

    # ========== 返回输出结果 ==========
    return output_data  # 返回所有输出张量


def postprocess_to_mask(output, save_path=r"D:\sam2\segment-anything-2-main\tools/27.jpg", ori_height=1024,
                        ori_width=1024):
    output = output.squeeze()  # 移除多余的维度
    resize_output = cv2.resize(output, (ori_width, ori_height))  # 调整尺寸
    sigmoid_output = 1.0 / (1.0 + np.exp(-resize_output))  # 应用 Sigmoid 函数
    thresholded = np.uint8(np.clip(sigmoid_output * 255, 0, 255))  # 阈值化处理
    thresholded = thresholded.astype(np.uint8)  # 确保数据类型为 uint8
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    contour_image = np.zeros_like(thresholded)  # 创建全黑图像
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)  # 绘制轮廓
    cv2.imwrite(save_path, contour_image)  # 保存图像
    return contour_image


img_path = "E:\LYX_date\deve_obb\images\images/27.jpg"
img = cv2.imread(img_path)
height, width, channels = img.shape

outputs, output_device = run_inference(img)


for i, out in enumerate(outputs):
    print(f"输出 {i}: shape={out.shape}, dtype={out.dtype}")
    # print(out)
input_data1 = outputs[0]
input_data2 = outputs[2]
input_data3 = outputs[1]
input_data4 = np.array([[[1493 * 1024 / width, 605 * 1024 / height, 2332 * 1024 / width, 1223 * 1024 / height]]],
                       dtype=np.float32)
input_data5 = np.array([[2, 3]], dtype=np.float32)
input_data6 = np.zeros((1, 1, 256, 256), dtype=np.float32)
input_data7 = np.array([0], dtype=np.float32)
input_data8 = np.array([width, height], dtype=np.int32)
# 调用推理函数


outputs = run_decoder_inference(input_data1, input_data2, input_data3, input_data4, input_data5, input_data6,
                                input_data7, input_data8, width, height)

for i, out in enumerate(outputs):
    print(f"\n输出 {i}: shape={out.shape}, dtype={out.dtype}")
    print(out)

mask_preddict = postprocess_to_mask(outputs[1], ori_height=height, ori_width=width)
cv2.imwrite("1.png", mask_preddict)