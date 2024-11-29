import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import cv2
import os
from get_xy  import startRoi
from ultralytics import YOLO
import argparse


def yolo_model(checkpoint,device):
    yolo = YOLO(checkpoint)
    yolo.to(device)
    return yolo

def sam2_model(config,checkpoint,device):
    sam2 = build_sam2(config, checkpoint, device=torch.device(device))
    predictor = SAM2ImagePredictor(sam2)
    return predictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="D:/sam2/segment-anything-2-main/sam2_configs/sam2_hiera_t.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="D:\sam2\segment-anything-2-main\checkpoints\sam2_hiera_tiny.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--chatpoint_weitiao",
        default=r"D:\sam2\segment-anything-2-main\tools\checkpoint_sam2/model_燕窝框.torch",
        type=str,
        help="微调的权重",
    )
    parser.add_argument(
        "--sam_device",
        default="cuda:0",
        type=str,
        help="sam2运行的device",
    )
    parser.add_argument(
        "--input_dir",
        default=r"E:/LYX_date/yanwo_cover/1_yanwo_cover_data/",
        help="输入照片路径",
    )
    parser.add_argument(
        "--output_dir",
        default=r"E:\LYX_date\yanwo_cover\2024_11_27",
        help="输出结果文件路径",
    )
    args = parser.parse_args()
    #sam2
    predictor = sam2_model(args.sam2_cfg,args.sam2_checkpoint, args.sam_device)

    for file_name in os.listdir(args.input_dir):
        image = os.path.join(args.input_dir, file_name)
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
        mask_filepath = os.path.join(args.output_dir, mask_filename)
        cv2.imwrite(mask_filepath, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), thickness=2)
        #img = cv2.rectangle(img, (xyxy[0], xyxy[1], xyxy[2], xyxy[3]), (0, 0, 255), 2)
        cv2.namedWindow("img", 0)
        cv2.resizeWindow("img", 1280, 720)  # 设置长宽
        cv2.imshow("img", img)
        cv2.waitKey(0)
if __name__ == "__main__":
    main()