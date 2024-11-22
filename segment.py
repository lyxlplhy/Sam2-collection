import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys
sys.warnoptions.append("ignore")
device = torch.device("cuda:0")
np.random.seed(3)

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

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


image = Image.open(r"D:\sam2\segment-anything-2-main\material_stack\box2/32_88691.343750.jpg")
# image = Image.open("D:\sam2\segment-anything-2-main\\notebooks\images\\truck.jpg")
image = np.array(image.convert("RGB"))
sam2_checkpoint = "D:\sam2\segment-anything-2-main\checkpoints\sam2_hiera_base_plus.pt"
model_cfg = "D:\sam2\segment-anything-2-main\sam2_configs\sam2_hiera_b+.yaml"

plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis('off')
plt.show()

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(image)
# input_point = np.array([[978, 976]])
# input_label = np.array([1])
# input_point = np.array([[[252, 265]],[[220, 287]]])
# input_label = np.array([[1],[1]])
# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=False,
# )

input_box = np.array([[1096, 224, 1895, 934]])

input_label = np.array([[1]])

masks, scores, logits = predictor.predict(
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
)
print(masks.shape)
# show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    #show_mask(mask, plt.gca(), random_color=True)
    mask_tensor = torch.as_tensor(mask).unsqueeze(2)
    mask_tensor1 = mask_tensor.expand(mask.shape[0], mask.shape[1], 3)
    mask_np = mask_tensor1.numpy()
    plt.imsave("D:\sam2\segment-anything-2-main\material_stack\\box2\\31_88611.265625.png", mask_np)

# show_points(input_point, input_label, plt.gca())
for box in input_box:
    show_box(box, plt.gca())
# show_box(input_box, plt.gca())
plt.axis('off')
plt.show()