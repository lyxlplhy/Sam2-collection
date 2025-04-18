from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def sam2_model(config,checkpoint,device,checkpoint_weitiao):
    sam2 = build_sam2(config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2)
    if checkpoint_weitiao!=None:
        predictor.model.load_state_dict(torch.load(checkpoint_weitiao))
    return predictor.model

from typing import Optional, Tuple, Any
import torch
from torch import nn
import torch.nn.functional as F
import argparse
from sam2.modeling.sam2_base import SAM2Base
import onnx
from onnxconverter_common import float16
class SAM2ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.no_mem_embed = sam_model.no_mem_embed

    def forward(self, x: torch.Tensor) -> tuple[Any, Any, Any]:
        backbone_out = self.image_encoder(x)
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        feature_maps = backbone_out["backbone_fpn"][-self.model.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.model.num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]

        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        feats = [feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
                 for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]

        return feats[0], feats[1], feats[2]


class SAM2ImageDecoder(nn.Module):
    def __init__(
            self,
            sam_model: SAM2Base,
            multimask_output: bool
    ) -> None:
        super().__init__()
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.model = sam_model
        self.multimask_output = multimask_output

    @torch.no_grad()
    def forward(
            self,
            image_embed: torch.Tensor,
            high_res_feats_0: torch.Tensor,
            high_res_feats_1: torch.Tensor,
            point_coords: torch.Tensor,
            point_labels: torch.Tensor,
            mask_input: torch.Tensor,
            has_mask_input: torch.Tensor,
            img_size: torch.Tensor
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        self.sparse_embedding = sparse_embedding
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        high_res_feats = [high_res_feats_0, high_res_feats_1]
        image_embed = image_embed

        masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        if self.multimask_output:
            masks = masks[:, 1:, :, :]
            iou_predictions = iou_predictions[:, 1:]
        else:
            masks, iou_predictions = self.mask_decoder._dynamic_multimask_via_stability(masks, iou_predictions)

        masks = torch.clamp(masks, -32.0, 32.0)
        print(masks.shape, iou_predictions.shape)

        masks = F.interpolate(masks, (img_size[0], img_size[1]), mode="bilinear", align_corners=False)

        return masks, iou_predictions

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:

        point_coords = point_coords + 0.5

        padding_point = torch.zeros((point_coords.shape[0], 1, 2), device=point_coords.device)
        padding_label = -torch.ones((point_labels.shape[0], 1), device=point_labels.device)
        point_coords = torch.cat([point_coords, padding_point], dim=1)
        point_labels = torch.cat([point_labels, padding_label], dim=1)

        point_coords[:, :, 0] = point_coords[:, :, 0] / self.model.image_size
        point_coords[:, :, 1] = point_coords[:, :, 1] / self.model.image_size

        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (
                point_labels == -1
        )

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[i].weight * (point_labels == i)

        return point_embedding

    def _embed_masks(self, input_mask: torch.Tensor, has_mask_input: torch.Tensor) -> torch.Tensor:
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (
                1 - has_mask_input
        ) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size",type=int,default=1024,help="输入模型时候照片的大小",)
    parser.add_argument("--multimask_output",type=bool,default=True,help="是否输出多个mask",)
    parser.add_argument("--sam2_checkpoint",default=r"D:\sam2\segment-anything-2-main\checkpoints\sam2_hiera_tiny.pt",type=str,help="sam2权重",)
    parser.add_argument("--model_cfg",default=r"D:\sam2\segment-anything-2-main\sam2_configs\sam2_hiera_t.yaml",type=str,help="sam2配置文件",)
    parser.add_argument("--weitiao_checkpoint",default="E:\era5\model_透明小盖子框.torch",help="sam2微调权重",)
    parser.add_argument("--f16", default=True, help="是否使用f16精度", )
    args = parser.parse_args()
    input_size = args.input_size  # Bad output if anything else (for now)
    sam2_model = build_sam2(args.model_cfg,args.sam2_checkpoint, device="cpu")
    sam2_model.load_state_dict(torch.load(args.weitiao_checkpoint))

    img = torch.randn(1, 3, input_size, input_size).cpu()

    sam2_encoder = SAM2ImageEncoder(sam2_model).cpu()
    high_res_feats_0, high_res_feats_1, image_embed = sam2_encoder(img)
    print(high_res_feats_0.shape)
    print(high_res_feats_1.shape)
    print(image_embed.shape)

    torch.onnx.export(sam2_encoder,
                      img,
                      f"onnx/conver_tiny_encoder.onnx",
                      export_params=True,
                      opset_version=17,
                      do_constant_folding=True,
                      input_names=['image'],
                      output_names=['high_res_feats_0', 'high_res_feats_1', 'image_embed']
                      )
    sam2_decoder = SAM2ImageDecoder(sam2_model, multimask_output=args.multimask_output).cpu()
    embed_dim = sam2_model.sam_prompt_encoder.embed_dim
    embed_size = (
    sam2_model.image_size // sam2_model.backbone_stride, sam2_model.image_size // sam2_model.backbone_stride)
    mask_input_size = [4 * x for x in embed_size]
    print(embed_dim, embed_size, mask_input_size)

    point_coords = torch.randint(low=0, high=input_size, size=(1, 5, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=1, size=(1, 5), dtype=torch.float)
    mask_input = torch.randn(1, 1, *mask_input_size, dtype=torch.float)
    has_mask_input = torch.tensor([1], dtype=torch.float)
    orig_im_size = torch.tensor([input_size, input_size], dtype=torch.int32)

    masks, scores = sam2_decoder(image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels,
                                 mask_input, has_mask_input, orig_im_size)

    torch.onnx.export(sam2_decoder,
                      (image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels, mask_input,
                       has_mask_input, orig_im_size),
                      "onnx/conver_tiny_decoder.onnx",
                      export_params=True,
                      opset_version=16,
                      do_constant_folding=True,
                      input_names=['image_embed', 'high_res_feats_0', 'high_res_feats_1', 'point_coords',
                                   'point_labels', 'mask_input', 'has_mask_input', 'orig_im_size'],
                      output_names=['masks', 'iou_predictions'],
                      dynamic_axes={"point_coords": {0: "num_labels", 1: "num_points"},
                                    "point_labels": {0: "num_labels", 1: "num_points"},
                                    "mask_input": {0: "num_labels"},
                                    "has_mask_input": {0: "num_labels"}
                                    }
                      )
    if args.f16:
        encoder_model_fp32 = onnx.load("onnx/conver_tiny_encoder.onnx")
        decoder_model_fp32 = onnx.load("onnx/conver_tiny_decoder.onnx")

        encoder_model_fp16 = float16.convert_float_to_float16(encoder_model_fp32, keep_io_types=True)
        decoder_model_fp16 = float16.convert_float_to_float16(decoder_model_fp32, keep_io_types=True)

        onnx.save(encoder_model_fp16, "D:\sam2\segment-anything-2-main/tools\onnx_float16/conver_tiny_encoder_f16.onnx")
        onnx.save(decoder_model_fp16, "D:\sam2\segment-anything-2-main/tools\onnx_float16/conver_tiny_decoder_f16.onnx")
if __name__ == "__main__":
    main()

