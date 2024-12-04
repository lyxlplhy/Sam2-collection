# Train/Fine-Tune SAM 2 on the LabPics 1 dataset

# This script use a single image batch, if you want to train with multi image per batch check this script:
# https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/blob/main/TRAIN_multi_image_batch.py

# Toturial: https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3
# Main repo: https://github.com/facebookresearch/segment-anything-2
# Labpics Dataset can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1
# Pretrained models for sam2 Can be downloaded from: https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints

import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import argparse


def read_batch(data):  # read random image and its annotaion from  the dataset (LabPics)

    #  select image

    ent = data[np.random.randint(len(data))]  # choose random entry
    Img = cv2.imread(ent["image"])[..., ::-1]  # read image
    ann_map = cv2.imread(ent["annotation"])  # read annotation

    # resize image

    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # scalling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                         interpolation=cv2.INTER_NEAREST)

    # merge vessels and materials annotations

    mat_map = ann_map[:, :, 0]  # material annotation map
    ves_map = ann_map[:, :, 2]  # vessel  annotaion map
    mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)  # merge maps

    # Get binary masks and points

    inds = np.unique(mat_map)[1:]  # load all indices
    boxs = []
    points = []
    masks = []
    for ind in inds:
        mask = (mat_map == ind).astype(np.uint8)  # make binary mask corresponding to index ind
        masks.append(mask)
        coords = np.argwhere(mask > 0)  # get all coordinates in mask
        yx = np.array(coords[np.random.randint(len(coords))])  # choose random point/coordinate
        # center_y = int(np.mean(coords[:, 0]))  # y 坐标的平均值
        # center_x = int(np.mean(coords[:, 1]))  # x 坐标的平均值
        # yx = [center_y, center_x]

        # mask_image = np.zeros_like(Img)  # 创建与原图像相同大小的空白图像
        # cv2.circle(mask_image, (yx[1], yx[0]), radius=5, color=(0, 0, 255), thickness=-1)  # 绘制红色圆点
        # cv2.imwrite(f'mask_{ind}.png', mask * 255)  # 将二进制掩码保存为图像
        # combined_image = cv2.addWeighted(Img, 0.5, mask_image, 0.5, 0)  # 合成带点的图像
        # cv2.imwrite(f'mask_with_point_{ind}.png', combined_image)  # 保存带点的图像
        points.append([[yx[1], yx[0]]])
        #############

        leftmost = coords[coords[:, 1].argmin()]  # 最左边的坐标
        rightmost = coords[coords[:, 1].argmax()]  # 最右边的坐标
        topmost = coords[coords[:, 0].argmin()]  # 最上边的坐标
        bottommost = coords[coords[:, 0].argmax()]  # 最下边的坐标
        left_point = [topmost[0], leftmost[1]]
        right_point = [bottommost[0], rightmost[1]]

        box_yxyx = [left_point, right_point]
        boxs.append(box_yxyx)

    return Img, np.array(masks), np.array(boxs), np.array(points), np.ones([len(masks), 1])
# Read data
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="E:\LYX_date\yanwo_cover\simple5\Train/",
        help="存放数据的路径",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="D:\sam2\segment-anything-2-main\checkpoints\sam2_hiera_tiny.pt",
        help="sam2权重路径",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="sam2_hiera_t.yaml",
        help="sam2配置文件路径",
    )
    args = parser.parse_args()

    data_dir = args.data_dir  # Path to dataset (LabPics 1)
    data = []  # list of files in dataset
    for ff, name in enumerate(os.listdir(data_dir + "Image/")):  # go over all folder annotation
        data.append({"image": data_dir + "Image/" + name,
                     "annotation": data_dir + "Instance/" + name[:-4] + ".png"})

    # Load model

    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device="cuda:0")  # load model
    predictor = SAM2ImagePredictor(sam2_model)

    # Set training parameters

    predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
    predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder
    '''
    #The main part of the net is the image encoder, if you have good GPU you can enable training of this part by using:
    predictor.model.image_encoder.train(True)
    #Note that for this case, you will also need to scan the SAM2 code for “no_grad” commands and remove them (“ no_grad” blocks the gradient collection, which saves memory but prevents training).
    '''
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler()  # mixed precision

    # Training loop

    for itr in range(100000):
        with torch.cuda.amp.autocast():  # cast to mix precision
            image, mask, input_box, input_point, input_label = read_batch(data)  # load data batch
            if mask.shape[0] == 0: continue  # ignore empty batches
            predictor.set_image(image)  # apply SAM image encoder to the image

            # prompt encoding

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(point_coords=input_point,
                                                                                    point_labels=input_label,
                                                                                    box=input_box,
                                                                                    mask_logits=None,
                                                                                    normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),
                                                                                     boxes=unnorm_box, masks=None, )

            # mask decoder
            batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings, multimask_output=True, repeat_image=batched_mode,
                high_res_features=high_res_features, )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[
                -1])  # Upscale the masks to the original image resolution

            # Segmentaion Loss caclulation

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])  # Turn logit map to probability map
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log(
                (1 - prd_mask) + 0.00001)).mean()  # cross entropy loss

            # Score loss calculation (intersection over union) IOU

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05  # mix losses

            # apply back propogation

            predictor.model.zero_grad()  # empty gradient
            scaler.scale(loss).backward()  # Backpropogate
            scaler.step(optimizer)
            scaler.update()  # Mix precision

            if itr % 50 == 0: torch.save(predictor.model.state_dict(),
                                         "tools/checkpoint_sam2/model_燕窝框.torch");print("save model")

            # Display results
            if itr == 0: mean_iou = 0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print("step)", itr, "Accuracy(IOU)=", mean_iou, "  loss=", loss)

if __name__ == "__main__":
    main()