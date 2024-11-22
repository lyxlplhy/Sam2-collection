
# 功能
  * sam2针对点和框作为提示信息的微调
  * onnx导出
  * sam2+手动给提示框分割，可生成mask标签
  * sam2+yolov8自动分割，可生成mask标签

# 环境
 * sam2: pip install -e .
 * onnx: pip install onnx
 * ultralytics: pip install ultralytics

# sam2微调
 * 点作为提示信息微调[tools/train_point.py](./tools/train_point.py)
 * 点+框作为提示信息微调[tools/train_box.py](./tools/train_box.py)

# onnx导出
 *[export_sam2onnx.py](./tools/export_sam2onnx.py)

# sam2+手动给提示框分割，可生成mask标签
 * [tools/inference2.py](./tools/inference2.py)

# sam2+yolov8自动分割，可生成mask标签
 * [tools/inference.py](./tools/inference.py)
