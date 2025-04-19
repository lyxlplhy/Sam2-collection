#encoder f16导出，未校准
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("E:\era5/conver_tiny_encoder.onnx", "rb") as f:
    parser.parse(f.read())

for i in range(network.num_inputs):
    t = network.get_input(i)
    print(f"Input {i}: name={t.name}, shape={t.shape}, is shape tensor: {t.is_shape_tensor}")

for i in range(network.num_outputs):
    t = network.get_output(i)
    print(f"Input {i}: name={t.name}, shape={t.shape}, is shape tensor: {t.is_shape_tensor}")


# 创建配置
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
# config.set_flag(trt.BuilderFlag.FP16)
# 设置 profile（动态输入 shape）
# 构建引擎
engine = builder.build_serialized_network(network, config)


# 保存引擎
with open(r"D:\sam2\segment-anything-2-main\tools\tensorrt_model/conver_tiny_encoder2.engine", "wb") as f:
    f.write(engine)
