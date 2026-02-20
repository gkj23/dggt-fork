import os
import argparse
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from IPython import embed
import lpips

from dggt.models.vggt import VGGT
from dggt.utils.load_fn import load_and_preprocess_images
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import palette_10, concat_list, get_split_gs, gs_dict,get_gs_items,downsample_3dgs
from gsplat.rendering import rasterization
from datasets.dataset import WaymoOpenDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import numpy as np


def compute_lifespan_loss(gamma):
    return torch.mean(torch.abs(1 / (gamma + 1e-6)))

def alpha_t(t, t0, alpha, gamma0 = 1, gamma1 = 0.1):
    # TBD：gamma1的系数怎么设定？
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) /  ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma*(t0-t)**2)
    alpha_ = alpha * conf
    return alpha_.float()


def get_dynamic_mask(time_invarient_pts_list, time_varient_pts, threshold=0.05, temperature=0.002):
    """
    根据 time_invarient_pts 与 time_varient_pts 在同一像素上的点云差异得到每帧的 dynamic_mask 与 dynamic_conf。
    同一像素对应的两点差距大于 threshold 时，该像素记为 dynamic（True）；conf 由 dist 平滑得到，越远越大。

    Args:
        time_invarient_pts_list: list of [B, S, H, W, 3] 时不变点图（可多份）
        time_varient_pts: [B, S, H, W, 3] 时变点图
        threshold: 判定为动态的 L2 距离阈值（世界坐标单位）
        temperature: sigmoid 软化的温度，越小在 threshold 附近越陡

    Returns:
        dynamic_mask: [B, S, H, W] bool，True 表示该像素为动态
        dynamic_conf: [B, S, H, W] float，基于 dist 的置信度 [0,1]，越大越可能是动态
    """
    device = time_varient_pts.device
    shape = time_varient_pts.shape[:-1]  # [B, S, H, W]
    dynamic_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    dynamic_conf = torch.zeros(shape, dtype=time_varient_pts.dtype, device=device)

    reference_pts = time_invarient_pts_list[0]
    for i, time_invarient_pts in enumerate(time_invarient_pts_list):
        if i == 0:
            continue
        diff = time_invarient_pts - reference_pts  # [B, S, H, W, 3]
        dist = torch.norm(diff, dim=-1, p=2)  # [B, S, H, W]
        valid = (
            torch.isfinite(time_invarient_pts).all(dim=-1)
            & torch.isfinite(reference_pts).all(dim=-1)
        )
        # 二值 mask：距离 > threshold 且有效
        dynamic_mask = dynamic_mask | ((dist > threshold) & valid)
        # conf：用 sigmoid((dist - threshold) / temperature)，dist 越大 conf 越大；无效处为 0
        conf = torch.sigmoid((dist - threshold) / temperature).to(reference_pts.dtype)
        conf = torch.where(valid, conf, torch.zeros_like(conf))
        dynamic_conf = torch.maximum(dynamic_conf, conf)
    return dynamic_mask, dynamic_conf


def parse_args():
    parser = argparse.ArgumentParser()
    # --image_dir <path>: Path to the Waymo dataset directory containing processed training data (required).
    # --log_dir <path>: Directory where training logs, checkpoints, and visualizations will be saved (required).
    # --ckpt_path <path>: Path to the pretrained model checkpoint to initialize weights (required).
    # --sequence_length <length>: Number of input frames for each training iteration, defaulting to 4 (optional).
    parser.add_argument('--image_dir', type=str, default="/DATA_EDS2/AIGC/2312/xuhr2312/gkj/dggt-fork/data/waymo/processed_test/training")
    parser.add_argument('--ckpt_path', type=str, default='/DATA_EDS2/AIGC/2312/xuhr2312/gkj/dggt-fork/pretrain/vdpm_model.pt')
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--sequence_length', type=int, default=4)#8,4
    parser.add_argument('--chunk_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--save_image', type=int, default=10)
    parser.add_argument('--save_ckpt', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--use_splatformer', type=bool, default=False)
    parser.add_argument('--downsample_3dgs', type=bool, default=False)
    parser.add_argument('--use_vdpm_backbone', action='store_true', help='Use v-dpm backbone instead of original aggregator')
    parser.add_argument('--decoder_depth', type=int, default=4, help='Depth of decoder when using v-dpm backbone')
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--dataset_length', type=int, default=50)
    return parser.parse_args()

def export_checkpoint_keys(checkpoint,output_file):
    
    try:
        # 2. 确定参数字典的位置
        # Checkpoint 有两种常见结构：
        #   (A) 整个文件就是一个 state_dict（全是参数）。
        #   (B) 文件是一个包含 'epoch', 'optimizer', 'state_dict' 等键的字典。
        
        params_dict = {}
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                print("检测到 'state_dict' 键，将提取其中的参数。")
                params_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                print("检测到 'model' 键，将提取其中的参数。")
                params_dict = checkpoint['model']
            else:
                print("未检测到嵌套结构，假设整个 checkpoint 即为参数字典。")
                params_dict = checkpoint
        else:
            print("警告:Checkpoint 不是字典格式，无法提取键值。")
            return

        # 3. 将参数名写入文件
        with open(output_file, "w", encoding="utf-8") as f:
            # 按字母顺序排序，方便查看
            sorted_keys = sorted(list(params_dict.keys()))
            
            for key in sorted_keys:
                # 获取对应的 tensor 形状，方便分析 (可选)
                shape = params_dict[key].shape if hasattr(params_dict[key], 'shape') else "No Shape"
                
                # 写入格式：参数名
                f.write(f"{key}\n")
                
                # 如果你也想看参数的维度，可以将上面一行改成：
                # f.write(f"{key} \t {shape}\n")

        print(f"成功！共提取 {len(sorted_keys)} 个参数。")
        print(f"结果已保存至: {output_file}")

    except Exception as e:
        print(f"发生错误: {e}")

def save_point_cloud_ply(filepath, points, colors):
    """
    保存点云为 PLY 文件
    points: [N, 3] numpy array
    colors: [N, 3] numpy array, range [0, 1]
    """
    # 确保颜色在 0-255 之间并转为整数
    colors = (colors * 255).clip(0, 255).astype(np.uint8)
    points = points.astype(np.float32)
    
    num_points = points.shape[0]
    
    # PLY 文件头
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # 将点和颜色合并: [x, y, z, r, g, b]
    data = np.hstack([points, colors])
    
    # 写入文件
    with open(filepath, 'w') as f:
        f.write(header)
        np.savetxt(f, data, fmt='%f %f %f %d %d %d')
    
    print(f"Point cloud saved to {filepath}")


def save_camera_frustums_ply(filepath, extrinsics, intrinsics, scale=1.0):
    """
    将相机外参保存为可视化的视锥线框 PLY 文件 (edges)
    Args:
        filepath: 保存路径 (.ply)
        extrinsics: [N, 4, 4] World-to-Camera 矩阵
        intrinsics: [N, 3, 3] 相机内参
        scale: 视锥体的大小缩放，根据场景大小调整，通常 0.1 ~ 1.0
    """
    if torch.is_tensor(extrinsics):
        extrinsics = extrinsics.detach().cpu().numpy()
    if torch.is_tensor(intrinsics):
        intrinsics = intrinsics.detach().cpu().numpy()
    num_cams = extrinsics.shape[0]
    w, h = 0.8, 0.45
    local_points = np.array([
        [0, 0, 0],
        [-w, -h, 1], [w, -h, 1], [w, h, 1], [-w, h, 1]
    ]) * scale
    all_vertices = []
    all_edges = []
    current_vertex_count = 0
    for i in range(num_cams):
        w2c = extrinsics[i]
        c2w = np.linalg.inv(w2c)
        local_points_h = np.hstack([local_points, np.ones((5, 1))])
        world_points = (c2w @ local_points_h.T).T[:, :3]
        all_vertices.append(world_points)
        edges_indices = np.array([
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1]
        ]) + current_vertex_count
        all_edges.append(edges_indices)
        current_vertex_count += 5
    all_vertices = np.vstack(all_vertices)
    all_edges = np.vstack(all_edges)
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {all_vertices.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {all_edges.shape[0]}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        np.savetxt(f, all_vertices, fmt="%.6f %.6f %.6f")
        np.savetxt(f, all_edges, fmt="%d %d")
    print(f"Cameras saved to {filepath}")


def save_point_cloud_with_cameras_ply(filepath, points, colors, extrinsics, intrinsics, frustum_scale=1.0):
    """
    保存点云（使用输入图像对应像素颜色）+ 相机视锥线框到同一个 PLY 文件。
    points: [N, 3] numpy, colors: [N, 3] numpy [0,1], extrinsics: [num_cams, 4, 4], intrinsics: [num_cams, 3, 3]
    """
    points = np.asarray(points, dtype=np.float32)
    colors = (np.asarray(colors) * 255).clip(0, 255).astype(np.uint8)
    if torch.is_tensor(extrinsics):
        extrinsics = extrinsics.detach().cpu().numpy()
    if torch.is_tensor(intrinsics):
        intrinsics = intrinsics.detach().cpu().numpy()
    num_cams = extrinsics.shape[0]
    w, h = 0.8, 0.45
    local_points = np.array([
        [0, 0, 0], [-w, -h, 1], [w, -h, 1], [w, h, 1], [-w, h, 1]
    ]) * frustum_scale
    frustum_vertices = []
    frustum_edges = []
    current_vertex_count = points.shape[0]
    for i in range(num_cams):
        w2c = extrinsics[i]
        c2w = np.linalg.inv(w2c)
        local_points_h = np.hstack([local_points, np.ones((5, 1))])
        world_pts = (c2w @ local_points_h.T).T[:, :3]
        frustum_vertices.append(world_pts)
        edges_indices = np.array([
            [0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]
        ]) + current_vertex_count
        frustum_edges.append(edges_indices)
        current_vertex_count += 5
    frustum_vertices = np.vstack(frustum_vertices)
    frustum_edges = np.vstack(frustum_edges)
    # 点云顶点颜色；视锥顶点用红色 (255,0,0) 便于区分
    frustum_colors = np.tile(np.array([255, 0, 0], dtype=np.uint8), (frustum_vertices.shape[0], 1))
    all_vertices = np.vstack([points, frustum_vertices])
    all_colors = np.vstack([colors, frustum_colors])
    num_vertices = all_vertices.shape[0]
    num_edges = frustum_edges.shape[0]
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {num_edges}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        data = np.hstack([all_vertices, all_colors])
        np.savetxt(f, data, fmt='%f %f %f %d %d %d')
        np.savetxt(f, frustum_edges, fmt='%d %d')
    print(f"Point cloud + camera frustums saved to {filepath}")


def main(args):
    print("Start training...")
    dist.init_process_group(backend='nccl')
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dtype = torch.float32
    
    #原先scene_names:[str(i).zfill(3) for i in range(300,600)]
    print("Start prepraring dataset...")
    dataset = WaymoOpenDataset(args.image_dir, scene_names=[str(i).zfill(3) for i in range(0,args.dataset_length)], sequence_length=args.sequence_length, mode=1, views=1)
    sampler = DistributedSampler(dataset,shuffle=True)
    print(f"Dataset length: {len(dataset)}")
    print(f"Image paths length: {len(dataset.image_paths)}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    print("Dataset preparation done!")

    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "ckpt"), exist_ok=True)

    cfg = None
    if args.use_vdpm_backbone:
        class Config:
            class model:
                decoder_depth = args.decoder_depth
        cfg = Config()
        model = VGGT(use_vdpm_backbone=True, cfg=cfg).to(device)
    else:
        model = VGGT().to(device)
    
    if args.resume_ckpt is not None:
        checkpoint = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded checkpoint from {args.resume_ckpt}")
    else:
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from {args.ckpt_path}")
        # 存下的就是aggregator,decoder和point_head
    export_checkpoint_keys(checkpoint, os.path.join(args.log_dir, "load_keys.txt"))

    print("\n" + "="*20 + " Checking Gradient Status " + "="*20)
    
    # 获取 checkpoint 中真实的参数字典 (处理嵌套情况)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            ckpt_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            ckpt_dict = checkpoint['model']
        else:
            ckpt_dict = checkpoint
    else:
        ckpt_dict = checkpoint

    # 预处理 checkpoint 的 key (移除可能的 'module.' 前缀以匹配当前 model)
    # 因为 checkpoint 可能是 DDP 训练出来的，key 会带 module.，而当前 model 还没 wrap DDP
    loaded_keys = set()
    for k in ckpt_dict.keys():
        clean_k = k.replace("module.", "") if k.startswith("module.") else k
        loaded_keys.add(clean_k)

    frozen_new_params = []
    
    for name, param in model.named_parameters():
        if name not in loaded_keys:
            if not param.requires_grad:
                frozen_new_params.append(name)
                print(f"[WARNING] 新参数未激活梯度: {name}")
            else:
                pass
    if len(frozen_new_params) > 0:
        print(f"\n[FAIL] 检测到 {len(frozen_new_params)} 个新加入的参数处于冻结状态！请检查 requires_grad 设置。")
        # 可以在这里 raise ValueError("Training setup error") 强制中断
    else:
        print("[PASS] 所有新加入的参数均已正确开启梯度 (requires_grad=True)。")
        
    print("="*64 + "\n")
    
    #===============加载模型结束===============
    
    model.train()
    model = DDP(model, device_ids=[args.local_rank]) #, find_unused_parameters=True)
    model._set_static_graph()
    
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)


    binary_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    semantic_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    for param in model.module.parameters():
        param.requires_grad = False
    for head_name in ["gs_head","instance_head","sky_model" ]: #, "gs_head","instance_head","sky_model", "semantic_head"
        for param in getattr(model.module, head_name).parameters():
            param.requires_grad = True

    optimizer = AdamW([
        {'params': model.module.gs_head.parameters(), 'lr': 4e-5},
        # {'params': model.module.semantic_head.parameters(), 'lr': 1e-4},
        # {'params': model.module.instance_head.parameters(), 'lr': 4e-5},
        {'params': model.module.sky_model.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)

    # 学习率调度：按 batch 步数计算，避免几百个 batch 后 lr 归零导致参数不更新
    num_batches_per_epoch = len(dataloader)
    total_steps = args.max_epoch * num_batches_per_epoch
    warmup_iterations = min(1000, total_steps // 10)

    def lr_lambda(global_step):
        if global_step < warmup_iterations:
            return (global_step + 1) / warmup_iterations
        progress = (global_step - warmup_iterations) / max(1, total_steps - warmup_iterations)
        return 0.5 * (1.0 + float(torch.cos(torch.tensor(torch.pi * progress)).item()))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    global_step = 0
    for step in tqdm(range(args.max_epoch)):
        sampler.set_epoch(step)        
        for batch in dataloader:
            images = batch['images'].to(device)
            # sky_mask = torch.zeros_like(images).permute(0,1,3,4,2) 
            # bg_mask = torch.ones(sky_mask.shape[:-1], dtype=torch.bool, device=device)#[B,S,H,W]
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            bg_mask = (sky_mask == 0).any(dim=-1)

            timestamps = batch['timestamps'][0].to(device)
            # print("timestamps:", timestamps.shape)#[S]
            if 'dynamic_mask' in batch:
                dynamic_masks = batch['dynamic_mask'].to(device)[:, :, 0, :, :]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=dtype):    
                chunked_renders, chunked_alphas = [], []
                S = images.shape[1]

                # 预处理：point_map, predictions, dynamic_mask
                aggregated_tokens_list, image_tokens_list, patch_start_idx, time_varient_pts, _ = model.module.get_time_varient_pts(images)
                
                # 得到每个时间戳下的time_invarient_pts
                time_invarient_pts_list = []
                decoded_tokens_list_list = []
                for idx in range(S):
                    decoded_tokens_list, time_invarient_pts, _ = model.module.get_time_invarient_pts(images, aggregated_tokens_list, patch_start_idx, idx)
                    time_invarient_pts_list.append(time_invarient_pts)
                    decoded_tokens_list_list.append(decoded_tokens_list)
                
                # 模型输入images，输出pos_enc,depth,gs_map,gs_conf,dynamic_conf,semantic_logits
                predictions = model(images=images,
                                    aggregated_tokens_list=aggregated_tokens_list,
                                    image_tokens_list=image_tokens_list,
                                    decoded_tokens_list=decoded_tokens_list_list[idx],
                                    patch_start_idx=patch_start_idx,
                                    timestamp=idx
                                    )
                H, W = images.shape[-2:]
                # pose encoding: B,S,9（平移向量T 旋转四元数 fov）
                # 内外参,一般这种训练都是Batchsize=1
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                # 外参最后一行的[0,0,0,1]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)  #[B,1,4]
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]
                
                # infer的时候先出来aggregated token list，
                # 然后给相同的时间戳decode出来N个decoded_token_list，
                # 因此出来的pts，gs都是有N套独立的
                # 但是不会独立渲染，因为防止闪烁，但是alpha_t的参数可以调大？
                # for i in range(len(predictions)):
                
                # TBD:这里直接用了point_head的输出作为点的初始化
                # 因为原文We remove the redundant depth map prediction and fine-tune the rest of the network.
                gs_map = predictions["gs_map"]  #[B,S,H,W,11]
                gs_conf = predictions["gs_conf"]  #[B,S,H,W,1]
                
                # threshold 参考 diff_dist 统计：p90≈0.014, p95≈0.033；约 6% 动态用 0.02~0.03；temperature 越小越陡
                dynamic_point, dy_map = get_dynamic_mask(time_invarient_pts_list, time_varient_pts, threshold=0.033, temperature=0.01)  # [B,S,H,W]

                for idx in range(S):
                    # load一下当前帧的预处理信息
                    point_map = time_invarient_pts_list[idx]

                    static_mask = torch.ones_like(bg_mask)
                    static_points = point_map[static_mask].reshape(-1, 3)
                    gs_dynamic_list = dy_map[static_mask] 
                    # 注意get_split_gs将下面所有都展平成[N,3], [N,1], [N,3], [N,4]这样的了
                    static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
                    static_opacity = static_opacity * (1 - gs_dynamic_list)
                    static_gs_conf = gs_conf[static_mask]
                    frame_idx = torch.nonzero(static_mask, as_tuple=False)[:,1]  #[N],表示了每个点属于的帧数（1对应S维度）
                    gs_timestamps = timestamps[frame_idx]     # 生成static_points中每个像素点对应的timestamp
                    # print("gs_timestamps:", gs_timestamps.shape) #[N][725200]=H*W*S 

                    point_map_i = point_map[:, idx] #[B,H,W,3]
                    bg_mask_i = bg_mask[:, idx]  # [B,H,W]
                    dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
                    dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, idx], bg_mask_i)
                    gs_dynamic_list_i = dy_map[:, idx][bg_mask_i]
                    dynamic_opacity = dynamic_opacity * gs_dynamic_list_i
                    
                    # t0为对应帧的时间戳
                    t0 = timestamps[idx]
                    static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0 = static_gs_conf)
                    static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                    # 因为不需要dynamic的了，所以全局这些元素都直接用静态的表示
                    world_points, rgbs, opacity, scales, rotation = concat_list(
                        static_gs_list,
                        [dynamic_point, dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation]
                    )
                    
                    renders_chunk, alphas_chunk, _ = rasterization(
                        means=world_points, 
                        quats=rotation, 
                        scales=scales, 
                        opacities=opacity, 
                        colors=rgbs,
                        viewmats=extrinsic[idx][None], 
                        Ks=intrinsic[idx][None],
                        width=W, 
                        height=H, 
                    )
                    # renders_chunk: [1,H,W,3], alphas_chunk: [1,H,W,1]
                    chunked_renders.append(renders_chunk)
                    chunked_alphas.append(alphas_chunk)


                renders = torch.cat(chunked_renders, dim=0) #[S,H,W,3]
                alphas = torch.cat(chunked_alphas, dim=0)
                bg_render = model.module.sky_model(images, extrinsic, intrinsic)
                renders = alphas * renders + (1 - alphas) * bg_render
                non_sky_renders = alphas * renders
                sky_renders = alphas * bg_render
                

                rendered_image = renders.permute(0, 3, 1, 2) #[S,3,H,W]
                non_sky_rendered_image = non_sky_renders.permute(0, 3, 1, 2) #[S,3,H,W]
                sky_rendered_image = sky_renders.permute(0, 3, 1, 2) #[S,3,H,W]
                target_image = images[0]


                ####################### Loss ###########################
                loss = F.l1_loss(rendered_image, target_image)

                sky_mask_loss = F.l1_loss(alphas, 1 - sky_mask[0, ..., 0][..., None])
                loss +=  sky_mask_loss

                gs_conf_loss = compute_lifespan_loss(static_gs_conf)
                loss += 0.01 * gs_conf_loss

                if step >= 0:
                    lpips_val = lpips_loss_fn(rendered_image, target_image)
                    loss += 0.05 * min(step / 1000, 1.0) * lpips_val.mean() # *

            loss.backward()

            # 在 step 前检查 gs_head 梯度（step 后 grad 会被清空）
            gs_head_grad_norm = 0.0
            for p in model.module.gs_head.parameters():
                if p.grad is not None:
                    gs_head_grad_norm += p.grad.data.norm(2).item() ** 2
            gs_head_grad_norm = gs_head_grad_norm ** 0.5

            optimizer.step()
            scheduler.step()
            global_step += 1

        # 同步：避免 rank0 在下面做 I/O 时，其他 rank 已进入下一轮导致 NCCL 超时
        dist.barrier()
        if args.local_rank == 0 and step % 1 == 0:
            if gs_head_grad_norm == 0:
                print(f"[WARNING] gs_head gradient norm is 0 — 梯度未传到 gs_head，请检查计算图或 rasterization 是否可导")
            print(f"[{step}/{args.max_epoch}] Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | gs_head_grad_norm: {gs_head_grad_norm:.2e}")
            print(f"[{step}/{args.max_epoch}]   sky Loss: {sky_mask_loss.item():.4f} | LR: {scheduler.get_last_lr()}")
            # print(f"[{step}/{args.max_epoch}]   dynamic Loss: {dynamic_loss.item():.4f} | LR: {scheduler.get_last_lr()}")
            print(f"[{step}/{args.max_epoch}]   gs_conf Loss: {gs_conf_loss.item():.4f} | LR: {scheduler.get_last_lr()}")
        if args.local_rank == 0 and step % args.save_image == 0:
            random_frame_idx = random.randint(0, rendered_image.shape[0] - 1)

            rendered = rendered_image[random_frame_idx].detach().cpu().clamp(0, 1)
            target = target_image[random_frame_idx].detach().cpu().clamp(0, 1)
            non_sky_rendered = non_sky_rendered_image[random_frame_idx].detach().cpu().clamp(0, 1)
            sky_rendered = sky_rendered_image[random_frame_idx].detach().cpu().clamp(0, 1)

            # 还没有sky 所以暂时不用记录sem_rgb
            sem_rgb = alphas[random_frame_idx, ..., 0].unsqueeze(0).repeat(3, 1, 1).cpu()  # [3, H, W]

            combined = torch.cat([target, sem_rgb, rendered], dim=-1) 
            non_sky_combined = torch.cat([target, sem_rgb, non_sky_rendered], dim=-1)
            sky_combined = torch.cat([target, sem_rgb, sky_rendered], dim=-1)

            T.ToPILImage()(combined).save(os.path.join(args.log_dir, "images", f"step_{step}_frame_{random_frame_idx}.png"))
            T.ToPILImage()(non_sky_combined).save(os.path.join(args.log_dir, "images", f"step_{step}_non_sky_frame_{random_frame_idx}.png"))
            T.ToPILImage()(sky_combined).save(os.path.join(args.log_dir, "images", f"step_{step}_sky_frame_{random_frame_idx}.png"))
            
            last_render = rendered_image[-1].detach().cpu().clamp(0, 1)
            last_target = target_image[-1].detach().cpu().clamp(0, 1)
            last_combined = torch.cat([last_target, last_render], dim=-1) 

            # 单独保存这张渲染图
            render_path = os.path.join(args.log_dir, "images", f"step_{step}_last_frame.png")
            T.ToPILImage()(last_combined).save(render_path)
            print(f"Rendered image saved to {render_path}")

            # 随机选一帧：原图 | dynamic_mask GT | 预测的 dynamic_mask 拼在一起输出
            target_frame = target_image[random_frame_idx].detach().cpu().clamp(0, 1)  # [3, H, W]
            pred_mask_vis = dy_map[0, random_frame_idx].detach().cpu().clamp(0, 1).unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
            if 'dynamic_mask' in batch:
                gt_mask_vis = dynamic_masks[0, random_frame_idx].detach().cpu().clamp(0, 1).unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
                dynamic_gt_combined = torch.cat([target_frame, gt_mask_vis, pred_mask_vis], dim=-1)  # [3, H, 3*W]
            else:
                dynamic_gt_combined = torch.cat([target_frame, pred_mask_vis], dim=-1)  # [3, H, 2*W]
            T.ToPILImage()(dynamic_gt_combined).save(
                os.path.join(args.log_dir, "images", f"step_{step}_dynamic_mask_frame_{random_frame_idx}.png")
            )

            # 随机选一帧：数据集 RGB | sky_mask 拼在一起输出
            sky_mask_frame = sky_mask[0, random_frame_idx]  # [H, W, C]
            if sky_mask_frame.shape[-1] == 1:
                sky_mask_vis = sky_mask_frame[..., 0].detach().cpu().clamp(0, 1).unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
            else:
                sky_mask_vis = sky_mask_frame.detach().cpu().clamp(0, 1).permute(2, 0, 1)  # [3, H, W]
            sky_combined = torch.cat([target_frame, sky_mask_vis], dim=-1)  # [3, H, 2*W]
            T.ToPILImage()(sky_combined).save(
                os.path.join(args.log_dir, "images", f"step_{step}_sky_mask_frame_{random_frame_idx}.png")
            )

            # if step % (args.save_image*5) ==0:
            #     ply_points = world_points.detach().cpu().numpy()
            #     # 使用输入图像中对应像素的颜色（与 point_map 的 (frame, h, w) 一一对应）
            #     if save_pixel_frames is not None:
            #         ply_colors = images[0, save_pixel_frames, :, save_pixel_rows, save_pixel_cols].detach().cpu().numpy()
            #         ply_colors = np.clip(ply_colors, 0.0, 1.0)
            #     else:
            #         ply_colors = rgbs.detach().cpu().numpy()
            #     ply_path = os.path.join(args.log_dir, "images", f"step_{step}_last_frame.ply")
            #     # 点云 + 相机视锥线框写入同一 PLY；视锥缩放可根据场景调整
            #     save_point_cloud_with_cameras_ply(
            #         ply_path, ply_points, ply_colors,
            #         extrinsic.detach(), intrinsic.detach(),
            #         frustum_scale=1.0
            #     )
            
        if args.local_rank == 0 and step > 0 and step % args.save_ckpt == 0:
            ckpt_path = os.path.join(args.log_dir, "ckpt", f"model_latest.pt")
            torch.save(model.module.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model at step {step} to {ckpt_path}")
        dist.barrier()

if __name__ == "__main__":
    args = parse_args()
    main(args)