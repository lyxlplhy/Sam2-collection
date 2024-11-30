import numpy as np
import torch
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import argparse


# Read data

def read_batch(data, batch_size=4):  # Allow batch size parameter
    images = []
    masks = []
    points = []
    labels = []
    for _ in range(batch_size):
        ent = data[np.random.randint(len(data))]  # Randomly select a data entry
        Img = cv2.imread(ent["image"])[..., ::-1]  # Read the image (convert BGR to RGB)
        ann_map = cv2.imread(ent["annotation"])  # Read the annotation

        # Resize image and annotation map
        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scale factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                             interpolation=cv2.INTER_NEAREST)

        # Merge material and vessel annotations
        mat_map = ann_map[:, :, 0]  # Material annotation map
        ves_map = ann_map[:, :, 2]  # Vessel annotation map
        mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)  # Merge maps

        # Get binary masks and points
        inds = np.unique(mat_map)[1:]  # Get all unique indices (excluding 0)
        batch_points = []
        batch_masks = []
        for ind in inds:
            mask = (mat_map == ind).astype(np.uint8)  # Create binary mask for this index
            batch_masks.append(mask)
            coords = np.argwhere(mask > 0)  # Get all coordinates where mask > 0
            yx = np.array(coords[np.random.randint(len(coords))])  # Randomly select a coordinate point
            batch_points.append([[yx[1], yx[0]]])  # Store selected point

        # Return a batch of images, masks, points, and labels
        images.append(Img)
        masks.append(np.array(batch_masks))
        points.append(np.array(batch_points))
        labels.append(np.ones([len(batch_masks), 1]))

    return np.array(images), np.array(masks), np.array(points), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="E:/LYX_date/yanwo_cover/simple5/Train/",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="D:/sam2/segment-anything-2-main/checkpoints/sam2_hiera_tiny.pt",
        help="Path to the SAM2 checkpoint",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="sam2_hiera_t.yaml",
        help="Path to the SAM2 configuration file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,  # Default batch size is 4
        help="Number of images per batch",
    )
    args = parser.parse_args()

    data_dir = args.data_dir  # Path to the dataset (LabPics 1)
    batch_size = args.batch_size  # Get batch size from command-line arguments

    # Load dataset
    data = []
    for ff, name in enumerate(os.listdir(data_dir + "Image/")):
        data.append({
            "image": data_dir + "Image/" + name,
            "annotation": data_dir + "Instance/" + name[:-4] + ".png"
        })

    # Build and load the SAM2 model
    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device="cuda:0")
    predictor = SAM2ImagePredictor(sam2_model)

    # Set the model for training
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    # Setup optimizer and scaler for mixed precision training
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision

    # Training loop
    for itr in range(100000):
        with torch.cuda.amp.autocast():  # Mixed precision
            images, masks, input_points, input_labels = read_batch(data, batch_size=batch_size)  # Load data batch
            if masks.shape[0] == 0: continue  # Skip empty batches

            # Process each image in the batch
            for i in range(batch_size):
                image = images[i]
                mask = masks[i]
                input_point = input_points[i]
                input_label = input_labels[i]

                predictor.set_image(image)  # Set the image to the model

                # Prepare prompts for the model (coordinates and labels)
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    input_point, input_label, box=None, mask_logits=None, normalize_coords=True
                )

                # Get sparse and dense embeddings
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), boxes=None, masks=None
                )

                # Perform mask decoding
                batched_mode = unnorm_coords.shape[0] > 1  # Multi-object prediction
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in
                                     predictor._features["high_res_feats"]]
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
                prd_masks = predictor._transforms.postprocess_masks(
                    low_res_masks, predictor._orig_hw[-1]
                )  # Upscale the masks to the original image resolution

                # Calculate segmentation loss (cross-entropy)
                gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                prd_mask = torch.sigmoid(prd_masks[:, 0])  # Convert logits to probability map
                seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log(
                    (1 - prd_mask) + 0.00001
                )).mean()

                # Calculate score loss (Intersection over Union, IoU)
                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                # Total loss
                loss = seg_loss + score_loss * 0.05  # Mix segmentation loss and score loss

                # Backpropagation
                predictor.model.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()  # Update mixed precision

            # Save the model every 50 iterations
            if itr % 50 == 0:
                torch.save(predictor.model.state_dict(), "model_checkpoint.torch")
                print("Saved model checkpoint.")

            # Display the mean IOU and loss every iteration
            if itr == 0:
                mean_iou = 0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print(f"Step {itr}, Accuracy (IOU) = {mean_iou:.4f}, Loss = {loss.item():.4f}")


if __name__ == "__main__":
    main()