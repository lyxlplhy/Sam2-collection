import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 模型配置和路径
checkpoint = "D:\\sam2\\segment-anything-2-main\\checkpoints\\sam2_hiera_tiny.pt"
model_cfg = "D:\\sam2\\segment-anything-2-main\\sam2_configs\\sam2_hiera_t.yaml"

# 假设你已经有了一个可以加载并初始化模型的函数
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# 视频路径
your_video = r"E:\LYX_date\SAM_data\SAM_data\sam2_lianxu\data\data3\images"  # 注意路径格式

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(your_video)

    # 添加新的点提示或框提示
    frame_idx = 0  # 第 0 帧
    obj_id = 1     # 对象 ID
    points = [[2028, 1331]]  # 像素坐标
    box=[1644,837,2405,1585]
    labels = [1]  # 标签，1 表示前景，0 表示背景

    # 添加提示并获得当前帧的分割结果
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
        clear_old_points=True,
        normalize_coords=True,
        box=box
    )
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        # 加载原始帧
        img_path = your_video + f"\\{frame_idx}.jpg"
        print(f"Image path: {img_path}")
        original_frame = cv2.imread(img_path)

        # 检查图像是否加载成功
        if original_frame is None:
            print(f"Error: Unable to load image at {img_path}")
            exit(1)  # 退出程序，避免后续错误

        # 获取mask：假设你只有一个mask
        mask = masks[0]  # 如果有多个mask，可能需要遍历多个mask

        # 将mask从PyTorch Tensor转换为NumPy数组
        mask = mask.cpu().numpy()  # 从GPU转换到CPU，再转为NumPy数组

        # 确保mask是二值图像
        mask = (mask > 0.5).astype(np.uint8)  # 如果mask是浮动值，将其转为二值图像

        # 检查mask的形状与图像一致
        print(f"Mask shape: {mask.shape}")
        print(f"Original frame shape: {original_frame.shape}")

        # 去掉第一个维度，使mask成为二维
        mask = mask[0]  # 现在 mask 的形状是 (2048, 2448)

        # 确保mask和图像尺寸相同，如果不同则需要调整大小
        if mask.shape != original_frame.shape[:2]:
            print(f"Resizing mask to match image dimensions...")
            mask = cv2.resize(mask, (original_frame.shape[1], original_frame.shape[0]))

        # 将mask应用到原图，使用合适的颜色（比如红色）显示mask
        masked_frame = original_frame.copy()
        masked_frame[mask == 1] = [0, 0, 255]  # 将mask区域设置为红色

        # 显示结果
        cv2.imshow('Masked Frame', masked_frame)
        cv2.waitKey(0)  # 等待按键
        cv2.destroyAllWindows()
        output_filename = f"E:\LYX_date\SAM_data\SAM_data\sam2_lianxu\data\data3\out1/masked_frame_{frame_idx}.jpg"  # 这里将 frame_idx 加到文件名中
        cv2.imwrite(output_filename, masked_frame)  # 保存图片
