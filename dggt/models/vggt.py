# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from dggt.heads.camera_head import CameraHead
from dggt.heads.dpt_head import DPTHead, GaussianHead
from dggt.heads.track_head import TrackHead
from dggt.heads.gs_head import GaussianDecoder
from dggt.models.sky import SkyGaussian
from dggt.models.fusion import PointNetGSFusion
from dggt.dpm.aggregator import Aggregator
from dggt.dpm.decoder import Decoder as VDPMDecoder
#from dggt.splatformer.feature_predictor import FeaturePredictor


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, semantic_num = 10, use_vdpm_backbone=False, cfg=None):
        super().__init__()
        
        self.use_vdpm_backbone = use_vdpm_backbone
        
        # 使用 v-dpm 的 aggregator 和 decoder
        # aggregator输入为images，shape：B, S, C_in(3), H, W
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.decoder = VDPMDecoder(
            cfg,
            dim_in=2*embed_dim,
            embed_dim=embed_dim,
            depth=cfg.model.decoder_depth if hasattr(cfg.model, 'decoder_depth') else 4
        )
        
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1",intermediate_layer_idx=[0,1,2,3])# ,down_ratio=2)
        #self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")# ,down_ratio=2)
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="sigmoid",intermediate_layer_idx=[0,1,2,3])

        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        
        #GS attributes
        # modify:换成2*embed_dim,intermediate_layer_idx=[0,1,2,3]
        self.gs_head = GaussianHead(dim_in= 3 * embed_dim, output_dim=3 + 1 + 3 + 4 + 1 , activation="sigmoid")# ,down_ratio=2)#RGB
        self.instance_head = DPTHead(dim_in= embed_dim, output_dim = 1 + 1, activation="linear",intermediate_layer_idx=[0,1,2,3]) # ,down_ratio=2)#RGB
        self.semantic_head = DPTHead(dim_in= embed_dim, output_dim = semantic_num + 1, activation="linear",intermediate_layer_idx=[0,1,2,3])# ,down_ratio=2)#RGB
        # Color, opacity, scale, rotation
        self.sky_model = SkyGaussian()
        #self.fusion_model = PointNetGSFusion()
        #self.splatformer = FeaturePredictor()
        #self.point_offset_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log_1")

    def get_time_varient_pts(self, images):
        
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, image_tokens_list, patch_start_idx = self.aggregator(images)

        with torch.cuda.amp.autocast(enabled=False):
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
        return aggregated_tokens_list, image_tokens_list, patch_start_idx, pts3d, pts3d_conf
    
    def get_time_invarient_pts(self, images, aggregated_tokens_list, patch_start_idx, timestamp):
        
        # 构造cond_view_idxs
        # 含义：对每一个时间戳做解码时，应该用哪个位置token作为condition
        # shape：[B,S],每帧对应一个位置token
        # 想要decode哪一帧，就把cond_view_idxs每帧需要的condition都设成那一帧index即可，即cond_view_idxs常常是整个矩阵是同一个数
        if self.use_vdpm_backbone and self.decoder is not None:
            B, S = images.shape[0], images.shape[1]
            # 取中间的时间帧
            cond_view_idxs=torch.ones(B, S, device=images.device, dtype=torch.int64)*timestamp
            decoded_tokens_list = self.decoder(images, aggregated_tokens_list, patch_start_idx, cond_view_idxs)
            # print("decoded_tokens.shape:", len(decoded_tokens_list), decoded_tokens_list[0].shape) #长度为4的list，每个元素[1,4,931,2048]

        with torch.cuda.amp.autocast(enabled=False):
            pts3d, pts3d_conf = self.point_head(
                decoded_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
        return decoded_tokens_list, pts3d, pts3d_conf
    
    def forward(
        self,
        images: torch.Tensor,
        aggregated_tokens_list: list = None,
        image_tokens_list: list = None,
        decoded_tokens_list: list = None,
        dino_tokes_list: list = None,
        patch_start_idx: int = None,
        query_points: torch.Tensor = None,
        cond_view_idxs: torch.Tensor = None,
        timestamp: int = None
    ):
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        
        # image_tokens_list #[24,1,4,931,2048] [aa_block_size,B,S,P,2C],和原先DGGT只差了P不一样

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.gs_head is not None:
                gs_map, gs_conf = self.gs_head(image_tokens_list, images, patch_start_idx)
                predictions["gs_map"] = gs_map
                predictions["gs_conf"] = gs_conf


        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        return predictions



