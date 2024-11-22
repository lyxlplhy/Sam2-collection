import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import cv2
import os
from get_xy  import startRoi
from ultralytics import YOLO


def yolo_model(checkpoint,device):
    yolo = YOLO(checkpoint)
    yolo.to(device)
    return yolo

def sam2_model(config,checkpoint,device):
    sam2 = build_sam2(config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2)
    return predictor

if __name__ == '__main__':
    #sam2
    checkpoint = r"D:\sam2\segment-anything-2-main\checkpoints\sam2_hiera_tiny.pt"  # 权重
    model_cfg = "D:\sam2\segment-anything-2-main\sam2_configs\sam2_hiera_t.yaml"  # 配置文件
    device = torch.device("cuda:0")
    predictor = sam2_model(model_cfg, checkpoint, device)

    # yolo
    yolo_conver = yolo_model(r"D:\sam2\ultralytics-main\ultralytics-main\runs\detect\train4\weights\best.pt",torch.device("cpu"))
    file_path = r"C:\Users\Admin\Documents\WeChat Files\wxid_sw6pddplsk6z22\FileStorage\File\2024-11\Data\Data"  # 读取的文件
    file_save = r"C:\Users\Admin\Documents\WeChat Files\wxid_sw6pddplsk6z22\FileStorage\File\2024-11\Data\out"  # 保存地址

    for file_name in os.listdir(file_path):
        image = os.path.join(file_path, file_name)
        print(image)
        img = cv2.imread(image)
        xyxy,point=startRoi(image)
        input_point = np.array([[[point[0],point[1]]]])
        predictor.set_image(img)
        input_label = np.array([[1]])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,#输入点
            box=xyxy,  # 输入框
            point_labels=input_label,
            multimask_output=True,
        )
        m = np.argmax(scores)
        mask = masks[m].astype(np.uint8)

        mask_filename = os.path.splitext(file_name)[0] + "_mask.png"  # 生成mask文件名
        mask_filepath = os.path.join(file_save, mask_filename)
        cv2.imwrite(mask_filepath, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), thickness=2)
        #img = cv2.rectangle(img, (xyxy[0], xyxy[1], xyxy[2], xyxy[3]), (0, 0, 255), 2)
        cv2.namedWindow("img", 0)
        cv2.resizeWindow("img", 1280, 720)  # 设置长宽
        cv2.imshow("img", img)
        cv2.waitKey(0)