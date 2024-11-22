import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import cv2
import os
import time
from get_xy import startRoi
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
    sam2 = build_sam2(config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2)
    if checkpoint_weitiao!=None:
        predictor.model.load_state_dict(torch.load(checkpoint_weitiao))
    return predictor

#sam2

if __name__ == '__main__':
    is_save_mask=False#是否保存mask
    save_mask_file=r"E:\LYX_date\mapImage\mapImage_mask"
    mask_area=False#是否计算绘制mask的面积
    #sam2
    checkpoint = r"D:\sam2\segment-anything-2-main\checkpoints\sam2_hiera_tiny.pt"  # 权重
    model_cfg = r"D:/sam2/segment-anything-2-main/sam2_configs/sam2_hiera_t.yaml"  # 配置文件
    chatpoint_weitiao=None
    device = torch.device("cuda:0")
    predictor = sam2_model(model_cfg, checkpoint, device,chatpoint_weitiao)
    # yolo
    yolo_conver = yolo_model(r"E:\LYX_date\mapImage/best.pt",torch.device("cpu"))

    file_path = r"E:\LYX_date\mapImage\mapImage"  # 读取的文件
    file_save = r"E:\LYX_date\mapImage\mapImage4"  # 保存地址
    for file_name in os.listdir(file_path):
        image = os.path.join(file_path, file_name)
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
            masks, scores, logits = predictor.predict(
                #point_coords=input_point,#输入点
                box=input_box,  # 输入框
                point_labels=input_label,
                multimask_output=True,
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
            if contours and mask_area:
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
            if is_save_mask:
                Image_save=cv2.imread(image)
                save_Instance=save_mask_file+"/Instance/"+file_name[:-3]+"png"
                save_Image = save_mask_file + "/Images/" + file_name
                cv2.imwrite(save_Image, Image_save)
                #show_mask(mask, plt.gca(), random_color=True)
                mask_tensor = torch.as_tensor(masks[m]).unsqueeze(2)
                mask_tensor1 = mask_tensor.expand(masks[m].shape[0], masks[m].shape[1], 3)
                mask_np = mask_tensor1.numpy()
                plt.imsave(save_Instance, mask_np)
        save = os.path.join(file_save, file_name)
        cv2.imwrite(save, img)