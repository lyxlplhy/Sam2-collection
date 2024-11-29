import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import cv2
import os
import time
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def yolo_model(checkpoint,device):
    yolo = YOLO(checkpoint)
    yolo.to(device)
    print(yolo)
    return yolo

def sam2_model(config,checkpoint,device,checkpoint_weitiao):
    sam2 = build_sam2(config, checkpoint, device=torch.device(device))
    predictor = SAM2ImagePredictor(sam2)
    if checkpoint_weitiao!=None:
        predictor.model.load_state_dict(torch.load(checkpoint_weitiao))
    return predictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_point_inference",
        type=bool,
        default=None,
        help="使用点进行推理（框和点至少要选一个，可以同时选）",
    )
    parser.add_argument(
        "--sam2_box_inference",
        type=bool,
        default=True,
        help="使用框进行推理（框和点至少要选一个，可以同时选）",
    )
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
        "--yolo_checkpoint",
        type=str,
        default=r"D:\sam2\ultralytics-main\ultralytics-main\runs\detect\train27\weights\best.pt",
        help="yolo目标检测的权重文件",
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
    parser.add_argument(
        "--is_save_mask",
        type=bool,
        default=False,
        help="是否需要保存mask结果",
    )
    parser.add_argument(
        "--mask_area",
        type=bool,
        default=False,
        help="是否需要计算mask的面积",
    )
    parser.add_argument(
        "--save_mask_file",
        default="E:\LYX_date\mapImage\mapImage_mask",
        help="保存mask的路径",
    )
    args = parser.parse_args()
    predictor = sam2_model(args.sam2_cfg, args.sam2_checkpoint, args.sam_device,args.chatpoint_weitiao)
    # yolo
    yolo_conver = yolo_model(args.yolo_checkpoint,torch.device("cpu"))
    for file_name in os.listdir(args.input_dir):
        image = os.path.join(args.input_dir, file_name)
        img = cv2.imread(image)

        yolo_reslut = yolo_conver.predict(image)
        for box in yolo_reslut[0].boxes:
            #print(box)
            x1, y1, x2, y2 = box.xywh[0]
            xx1,yy1,xx2,yy2=box.xyxy[0]
            input_point = np.array([[[int(x1), int(y1)]]])  # yolo
            input_box=np.array([[int(xx1), int(yy1)],[int(xx2),int(yy2)]])  # yolo
            input_label = np.array([[1]])
            start_time = time.time()
            predictor.set_image(img)
            if not args.sam2_point_inference:
                input_point=None
            if not args.sam2_box_inference:
                input_box=None
            masks, scores, logits = predictor.predict(
                point_coords=input_point,#输入点
                box=input_box,  # 输入框
                point_labels=input_label,
                multimask_output=False,
            )
            end_time=time.time()
            execution_time = end_time - start_time
            print(execution_time)
            m=np.argmax(scores)
            mask = masks[m].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 绘制最大轮廓
            cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=4)
            img = cv2.rectangle(img, (int(xx1), int(yy1)), (int(xx2), int(yy2)), (0, 255, 0), 2)
            #################################################
            if contours and args.mask_area:
                areas = [cv2.contourArea(cnt) for cnt in contours]
                max_area_index = np.argmax(areas)  # 找到最大面积的索引
                max_area = areas[max_area_index]  # 获取最大面积
                max_contour = contours[max_area_index]  # 获取对应的轮廓

                # 绘制最大轮廓
                cv2.drawContours(img, [max_contour], -1, (0, 0, 255), thickness=4)

                # 在边界框上方绘制最大面积
                cv2.putText(img, f'Area: {int(max_area)}', (int(xx1), int(yy1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)  # 增大字体
            #######################################################################################
            #是否保存掩膜
            if args.is_save_mask:
                Image_save=cv2.imread(image)
                save_Instance=args.save_mask_file+"/Instance/"+file_name[:-3]+"png"
                save_Image = args.save_mask_file + "/Images/" + file_name
                cv2.imwrite(save_Image, Image_save)
                #show_mask(mask, plt.gca(), random_color=True)
                mask_tensor = torch.as_tensor(masks[m]).unsqueeze(2)
                mask_tensor1 = mask_tensor.expand(masks[m].shape[0], masks[m].shape[1], 3)
                mask_np = mask_tensor1.numpy()
                plt.imsave(save_Instance, mask_np)
        save = os.path.join(args.output_dir, file_name)
        cv2.imwrite(save, img)
if __name__ == "__main__":
    main()