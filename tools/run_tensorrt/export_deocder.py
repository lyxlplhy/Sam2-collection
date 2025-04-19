#decoder onnx导出为tensorrt，支持动态输出输入
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("E:\era5/conver_tiny_decoder_del.onnx", "rb") as f:
    parser.parse(f.read())

print("intiwwwwwww")
# for i in range(network.num_inputs):
#     t = network.get_input(i)
#     print(f"Input {i}: name={t.name}, shape={t.shape}, is shape tensor: {t.is_shape_tensor}")
#
# for i in range(network.num_outputs):
#     t = network.get_output(i)
#     print(f"Input {i}: name={t.name}, shape={t.shape}, is shape tensor: {t.is_shape_tensor}")


# 创建配置
config = builder.create_builder_config()
config.max_workspace_size = 1 << 32  # 4GB

# config.set_flag(trt.BuilderFlag.FP16)
# 设置 profile（动态输入 shape）
profile = builder.create_optimization_profile()
profile.set_shape("point_coords", (1, 1, 2), (1,2, 2), (1, 5, 2))
profile.set_shape("point_labels", (1, 1), (1,2), (1, 5))
profile.set_shape("mask_input", (1, 1,256,256),  (1, 1,256,256),  (1, 1,256,256))
profile.set_shape("has_mask_input", (1,),  (1,),  (1, ))
profile.set_shape_input('orig_im_size', min=(1,2), opt=(1,2), max=(1,2))



print("intit")
config.add_optimization_profile(profile)
# 构建引擎
engine = builder.build_serialized_network(network, config)

print(engine)
# 保存引擎
with open(r"D:\sam2\segment-anything-2-main\tools\tensorrt_model/conver_tiny_decoder.engine", "wb") as f:
    f.write(engine)
