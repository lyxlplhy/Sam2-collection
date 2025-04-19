#sam2 encoder int8 校准+量化+导出
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2
import os

class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir, input_shape=(1, 3, 1024, 1024), cache_file="calib.cache"):
        super(MyCalibrator, self).__init__()
        self.cache_file = cache_file
        self.input_shape = input_shape
        self.image_paths = [os.path.join(calib_dir, f) for f in os.listdir(calib_dir)]
        self.index = 0
        self.device_input = cuda.mem_alloc(trt.volume(input_shape) * np.float32().nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.index >= len(self.image_paths):
            return None
        path = self.image_paths[self.index]
        img = cv2.imread(path)
        img = cv2.resize(img, (1024, 1024))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        img = (img - mean) / std
        img = np.ascontiguousarray(img, dtype=np.float32)
        cuda.memcpy_htod(self.device_input, img)
        self.index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open("E:\era5/conver_tiny_encoder.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
config.set_flag(trt.BuilderFlag.INT8)

calibrator = MyCalibrator(r"D:\sam2\int8校准数据集/")
config.int8_calibrator = calibrator

engine = builder.build_engine(network, config)

with open(r"D:\sam2\segment-anything-2-main\tools\tensorrt_model/model_int8.engine", "wb") as f:
    f.write(engine.serialize())
