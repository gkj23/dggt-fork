"""
对多组 case（不同 batch）将该帧的 time_invariant 点云、time_variant 点云以及每帧相机视锥保存到 PLY。
时变/时不变点云使用原图对应像素颜色着色，视锥顶点为红色。time_variant 对每帧 s 用该帧 c2w 反变换到世界系。
每个 case 输出：合并(单帧)、仅 inv(单帧)、S 个 var_cam_frame*、以及一个「本场景所有 S 帧时不变点云+相机」合在一个 PLY 里
（*_all_frames_{case}.ply）。单卡运行，不启用 DDP。
"""
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from datasets.dataset import WaymoOpenDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Export time_invariant + time_variant point clouds and camera frustums to one PLY")
    parser.add_argument('--image_dir', type=str,
                        default="/DATA_EDS2/AIGC/2312/xuhr2312/gkj/dggt-fork/data/waymo/processed/training")
    parser.add_argument('--ckpt_path', type=str,
                        default='/DATA_EDS2/AIGC/2312/xuhr2312/gkj/dggt-fork/pretrain/vdpm_model.pt')
    parser.add_argument('--out_dir', type=str, default='logs/export_ply_test')
    parser.add_argument('--out_name', type=str, default='inv_var_cameras.ply')
    parser.add_argument('--sequence_length', type=int, default=4)
    parser.add_argument('--scene_range', type=str, default='0,50')
    parser.add_argument('--use_vdpm_backbone', action='store_true')
    parser.add_argument('--decoder_depth', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--frustum_scale', type=float, default=1.0, help='Scale of camera frustum size')
    parser.add_argument('--num_cases', type=int, default=5, help='Number of PLY files to export (different batches, each with one random frame)')
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def camera_to_world_points(pts_cam, w2c_4x4):
    """
    将相机坐标系下的点变换到世界坐标系。pts_cam: [N, 3], w2c_4x4: [4, 4] world-to-camera。
    返回 [N, 3] 世界坐标。inf/nan 会保留，仅对有限点做变换。
    """
    pts_cam = np.asarray(pts_cam, dtype=np.float32)
    c2w = np.linalg.inv(np.asarray(w2c_4x4, dtype=np.float64))
    valid = np.isfinite(pts_cam).all(axis=1)
    out = np.full_like(pts_cam, np.nan)
    if valid.any():
        pts_h = np.hstack([pts_cam[valid], np.ones((valid.sum(), 1), dtype=np.float32)])
        out[valid] = (c2w @ pts_h.T).T[:, :3].astype(np.float32)
    return out


def get_frustum_vertices_edges(extrinsics_4x4, scale=1.0):
    """
    extrinsics_4x4: [N, 4, 4] World-to-Camera
    Returns:
        vertices: [N*5, 3] numpy
        edges: [N*8, 2] numpy, indices into vertices
    """
    if torch.is_tensor(extrinsics_4x4):
        extrinsics_4x4 = extrinsics_4x4.detach().cpu().numpy()
    num_cams = extrinsics_4x4.shape[0]
    w, h = 0.8, 0.45
    local_points = np.array([
        [0, 0, 0],
        [-w, -h, 1], [w, -h, 1], [w, h, 1], [-w, h, 1]
    ], dtype=np.float64) * scale
    all_vertices = []
    all_edges = []
    current_vertex_count = 0
    for i in range(num_cams):
        w2c = extrinsics_4x4[i]
        c2w = np.linalg.inv(w2c)
        local_points_h = np.hstack([local_points, np.ones((5, 1))])
        world_points = (c2w @ local_points_h.T).T[:, :3]
        all_vertices.append(world_points)
        edges_indices = np.array([
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1]
        ], dtype=np.int64) + current_vertex_count
        all_edges.append(edges_indices)
        current_vertex_count += 5
    return np.vstack(all_vertices).astype(np.float32), np.vstack(all_edges)


def save_inv_var_frustum_ply(filepath, inv_pts, var_pts, frustum_vertices, frustum_edges,
                             color_inv=(0, 255, 0), color_var=(255, 255, 0), color_frustum=(255, 0, 0),
                             inv_colors=None, var_colors=None):
    """
    保存到单个 PLY：time_invariant 点云 + time_variant 点云 + 视锥顶点(红)，带 edge 仅对视锥。
    inv_pts, var_pts: [N, 3] float32, 可含 inf/nan，会过滤掉
    frustum_vertices: [M, 3], frustum_edges: [E, 2] 索引为在 frustum_vertices 内的局部索引
    inv_colors, var_colors: 可选 [N, 3] 与 inv_pts/var_pts 同长度，取值 [0,1]；若提供则用原图颜色，否则用 color_inv/color_var 单色
    """
    inv_pts = np.asarray(inv_pts, dtype=np.float32)
    var_pts = np.asarray(var_pts, dtype=np.float32)
    valid_inv = np.isfinite(inv_pts).all(axis=1)
    valid_var = np.isfinite(var_pts).all(axis=1)
    inv_pts = inv_pts[valid_inv]
    var_pts = var_pts[valid_var]

    n_inv, n_var = inv_pts.shape[0], var_pts.shape[0]
    n_frustum = frustum_vertices.shape[0]
    all_vertices = np.vstack([inv_pts, var_pts, frustum_vertices])

    if inv_colors is not None:
        inv_colors = np.asarray(inv_colors, dtype=np.float32)
        inv_colors = (inv_colors[valid_inv] * 255).clip(0, 255).astype(np.uint8)
    else:
        inv_colors = np.tile(np.array(color_inv, dtype=np.uint8), (n_inv, 1))
    if var_colors is not None:
        var_colors = np.asarray(var_colors, dtype=np.float32)
        var_colors = (var_colors[valid_var] * 255).clip(0, 255).astype(np.uint8)
    else:
        var_colors = np.tile(np.array(color_var, dtype=np.uint8), (n_var, 1))

    color_frustum_arr = np.tile(np.array(color_frustum, dtype=np.uint8), (n_frustum, 1))
    all_colors = np.vstack([inv_colors, var_colors, color_frustum_arr])

    # edge 顶点索引：视锥顶点在 all_vertices 中从 n_inv + n_var 开始
    edge_offset = n_inv + n_var
    frustum_edges_global = frustum_edges + edge_offset

    num_vertices = all_vertices.shape[0]
    num_edges = frustum_edges_global.shape[0]
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
        np.savetxt(f, frustum_edges_global, fmt='%d %d')
    print(f"Saved: {filepath} (time_invariant: {n_inv}, time_variant: {n_var}, frustum: {n_frustum})")


def save_all_frames_points_frustum_ply(filepath, pts_list, colors_list, frustum_vertices, frustum_edges,
                                       color_frustum=(255, 0, 0), label="points"):
    """
    将多帧点云 + 视锥写入同一个 PLY（一个场景里包含所有帧的点云）。
    pts_list: list of [N_s, 3]，每帧的点云（世界系）
    colors_list: list of [N_s, 3] 取值 [0,1]，与 pts_list 逐帧对应
    """
    all_pts = []
    all_colors = []
    for pts, colors in zip(pts_list, colors_list):
        pts = np.asarray(pts, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        colors = (colors[valid] * 255).clip(0, 255).astype(np.uint8) if colors.size else np.zeros((0, 3), dtype=np.uint8)
        all_pts.append(pts)
        all_colors.append(colors)
    n_pts_total = sum(p.shape[0] for p in all_pts)
    n_frustum = frustum_vertices.shape[0]
    vertices = np.vstack(all_pts + [frustum_vertices])
    colors = np.vstack(all_colors + [np.tile(np.array(color_frustum, dtype=np.uint8), (n_frustum, 1))])
    edge_offset = n_pts_total
    frustum_edges_global = frustum_edges + edge_offset
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {frustum_edges_global.shape[0]}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        data = np.hstack([vertices, colors])
        np.savetxt(f, data, fmt='%f %f %f %d %d %d')
        np.savetxt(f, frustum_edges_global, fmt='%d %d')
    print(f"Saved (all frames in one scene, {label}): {filepath} (frames: {len(pts_list)}, total pts: {n_pts_total}, frustum: {n_frustum})")


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    start, end = map(int, args.scene_range.split(','))
    scene_names = [str(i).zfill(3) for i in range(start, end)]

    print("Loading dataset...")
    dataset = WaymoOpenDataset(
        args.image_dir,
        scene_names=scene_names,
        sequence_length=args.sequence_length,
        mode=1,
        views=1,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("Loading model...")
    cfg = None
    if args.use_vdpm_backbone:
        class Config:
            class model:
                decoder_depth = args.decoder_depth
        cfg = Config()
        model = VGGT(use_vdpm_backbone=True, cfg=cfg).to(device)
    else:
        model = VGGT().to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    out_base = args.out_name.replace('.ply', '') if args.out_name.endswith('.ply') else args.out_name

    for case_idx, batch in enumerate(dataloader):
        if case_idx >= args.num_cases:
            break
        images = batch['images'].to(device)
        B, S, C, H, W = images.shape

        with torch.no_grad():
            aggregated_tokens_list, image_tokens_list, patch_start_idx, time_varient_pts, _ = \
                model.get_time_varient_pts(images)

            time_invarient_pts_list = []
            decoded_tokens_list_list = []
            for idx in range(S):
                decoded_tokens_list, time_invarient_pts, _ = model.get_time_invarient_pts(
                    images, aggregated_tokens_list, patch_start_idx, idx
                )
                time_invarient_pts_list.append(time_invarient_pts)
                decoded_tokens_list_list.append(decoded_tokens_list)

            # 随机选一个时间戳（帧索引），用于单帧的 inv/合并 导出
            frame_idx = random.randint(0, S - 1) if S > 1 else 0

            # 该帧的 time_invariant 点云与颜色
            time_invarient_at_t = time_invarient_pts_list[frame_idx][0, frame_idx]  # [H, W, 3]
            inv_pts_np = time_invarient_at_t.cpu().float().numpy().reshape(-1, 3)
            frame_rgb = images[0, frame_idx].permute(1, 2, 0).cpu().float().numpy()
            pixel_colors = np.clip(frame_rgb, 0.0, 1.0).reshape(-1, 3)

            # 相机 pose -> 4x4 外参（一次得到所有 S 帧）
            pred = model(
                images,
                aggregated_tokens_list=aggregated_tokens_list,
                image_tokens_list=image_tokens_list,
                decoded_tokens_list=decoded_tokens_list_list[frame_idx],
                patch_start_idx=patch_start_idx,
                timestamp=frame_idx,
            )
            extrinsics_3x4, _ = pose_encoding_to_extri_intri(pred["pose_enc"], (H, W))
            bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            extrinsics_4x4 = []
            for s in range(extrinsics_3x4.shape[1]):
                e = extrinsics_3x4[0, s].cpu().numpy()
                e4 = np.vstack([e, bottom])
                extrinsics_4x4.append(e4)
            extrinsics_4x4 = np.stack(extrinsics_4x4, axis=0)

            # time_variant：每帧 s 的点云在该帧相机系下用 c2w_s 反变换到世界系
            var_pts_world_per_frame = []
            var_colors_per_frame = []
            for s in range(S):
                var_s = time_varient_pts[0, s].cpu().float().numpy().reshape(-1, 3)
                var_pts_world_per_frame.append(camera_to_world_points(var_s, extrinsics_4x4[s]))
                rgb_s = images[0, s].permute(1, 2, 0).cpu().float().numpy()
                var_colors_per_frame.append(np.clip(rgb_s, 0.0, 1.0).reshape(-1, 3))

            # 本 case 所有 S 帧的时不变点云（用于 all_frames PLY）
            inv_pts_per_frame = []
            inv_colors_per_frame = []
            for s in range(S):
                inv_s = time_invarient_pts_list[s][0, s].cpu().float().numpy().reshape(-1, 3)
                inv_pts_per_frame.append(inv_s)
                rgb_s = images[0, s].permute(1, 2, 0).cpu().float().numpy()
                inv_colors_per_frame.append(np.clip(rgb_s, 0.0, 1.0).reshape(-1, 3))

            var_pts_np = var_pts_world_per_frame[frame_idx]

            frustum_vertices, frustum_edges = get_frustum_vertices_edges(
                extrinsics_4x4, scale=args.frustum_scale
            )

        # # 1) 时不变 + 时变 + 相机视锥（单帧 frame_idx）
        # out_path = os.path.join(args.out_dir, f"{out_base}_{case_idx:03d}.ply")
        # save_inv_var_frustum_ply(
        #     out_path,
        #     inv_pts_np,
        #     var_pts_np,
        #     frustum_vertices,
        #     frustum_edges,
        #     color_frustum=(255, 0, 0),
        #     inv_colors=pixel_colors,
        #     var_colors=pixel_colors,
        # )
        # 2) 仅时不变点云 + 相机视锥（单帧）
        out_inv_cam = os.path.join(args.out_dir, f"{out_base}_inv_cam_{case_idx:03d}.ply")
        save_inv_var_frustum_ply(
            out_inv_cam,
            inv_pts_np,
            np.zeros((0, 3), dtype=np.float32),
            frustum_vertices,
            frustum_edges,
            color_frustum=(255, 0, 0),
            inv_colors=pixel_colors,
        )
        # # 3) 时变点云：S 个单帧文件
        # for s in range(S):
        #     out_var_cam = os.path.join(args.out_dir, f"{out_base}_var_cam_{case_idx:03d}_frame{s:02d}.ply")
        #     save_inv_var_frustum_ply(
        #         out_var_cam,
        #         np.zeros((0, 3), dtype=np.float32),
        #         var_pts_world_per_frame[s],
        #         frustum_vertices,
        #         frustum_edges,
        #         color_frustum=(255, 0, 0),
        #         var_colors=var_colors_per_frame[s],
        #     )
        # 4) 本 case 所有 S 帧的时不变点云 + 相机视锥 合在一个 PLY 里（一个场景）
        out_all_frames = os.path.join(args.out_dir, f"{out_base}_all_frames_{case_idx:03d}.ply")
        save_all_frames_points_frustum_ply(
            out_all_frames,
            inv_pts_per_frame,
            inv_colors_per_frame,
            frustum_vertices,
            frustum_edges,
            color_frustum=(255, 0, 0),
            label="time_invariant",
        )
        print(f"  case {case_idx}: frame_idx={frame_idx} (of 0..{S-1}), + all S frames (time_invariant) in one scene")

    n_cases = case_idx + 1
    n_per_case = 2 + 1  # inv_cam (单帧), all_frames (时不变 S 帧合一)
    print(f"Done. Exported {n_cases} cases × {n_per_case} PLY each = {n_cases * n_per_case} files to {args.out_dir}/")


if __name__ == "__main__":
    main()
