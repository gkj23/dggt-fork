"""
测试 dynamic_masks 的 threshold 设定：对若干样本跑 get_time_varient_pts 与 get_time_invarient_pts，
统计每个像素上两点云位置的 diff（x,y,z）和 dist（L2），输出统计量便于设定 threshold / temperature。
单卡或 CPU 运行，不启用 DDP。

坐标系说明（从代码推断，无显式标注）：
- time_varient_pts: point_head(aggregated_tokens_list)，聚合多帧 token 后预测，渲染时直接当 world 使用。
- time_invarient_pts_list[s]: point_head(decoded_tokens_list)，decoder 以 cond_view_idxs=s 解码后预测。
代码中两者均无相机/世界变换，设计上假定为同一 world 系；若统计异常可加 --unify_frame 0 在「统一到第 0 帧相机系」后再算 dist 对比。
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from datasets.dataset import WaymoOpenDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test diff/dist stats for dynamic mask threshold")
    parser.add_argument('--image_dir', type=str,
                        default="/DATA_EDS2/AIGC/2312/xuhr2312/gkj/dggt-fork/data/waymo/processed/training")
    parser.add_argument('--ckpt_path', type=str,
                        default='/DATA_EDS2/AIGC/2312/xuhr2312/gkj/dggt-fork/pretrain/vdpm_model.pt')
    parser.add_argument('--out_dir', type=str, default='logs/dynamic_threshold_test',
                        help='Output dir for stats and dist maps')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of batches (samples) to run')
    parser.add_argument('--sequence_length', type=int, default=4)
    parser.add_argument('--scene_range', type=str, default='0,50',
                        help='Comma-separated start,end for scene indices, e.g. 0,50')
    parser.add_argument('--use_vdpm_backbone', action='store_true')
    parser.add_argument('--decoder_depth', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--unify_frame', type=int, default=None,
                        help='If set (e.g. 0), convert both point maps to camera of this frame then compute dist; use to check coordinate mismatch.')
    return parser.parse_args()


def world_to_camera_batch(pts_world, extrinsics):
    """pts_world: [B, S, H, W, 3], extrinsics: [B, S, 3, 4] (world-to-camera [R|t]). Return [B, S, H, W, 3] in camera."""
    B, S, H, W, _ = pts_world.shape
    pts = pts_world.reshape(B * S, -1, 3)  # [B*S, H*W, 3]
    ones = torch.ones(pts.shape[0], pts.shape[1], 1, device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=-1).transpose(-1, -2)  # [B*S, 4, H*W]
    ext = extrinsics.reshape(B * S, 3, 4)
    cam = torch.bmm(ext, pts_h)  # [B*S, 3, H*W]
    return cam.transpose(-1, -2).reshape(B, S, H, W, 3)


def compute_diff_dist_stats(time_invarient_pts_list, time_varient_pts, valid_only=True):
    """
    与 get_dynamic_mask 相同的 diff/dist 计算，但返回所有像素的 diff 和 dist 用于统计。
    time_invarient_pts_list: list of [B, S, H, W, 3]
    time_varient_pts: [B, S, H, W, 3]
    Returns:
        all_diffs: [N, 3] 所有有效像素的 diff (inv - var)
        all_dists: [N] 所有有效像素的 L2 dist
    """
    device = time_varient_pts.device
    all_diffs = []
    all_dists = []

    for time_invarient_pts in time_invarient_pts_list:
        diff = time_invarient_pts - time_varient_pts  # [B, S, H, W, 3]
        dist = torch.norm(diff, dim=-1, p=2)  # [B, S, H, W]
        valid = (
            torch.isfinite(time_invarient_pts).all(dim=-1)
            & torch.isfinite(time_varient_pts).all(dim=-1)
        )
        if valid_only:
            diff_flat = diff[valid].reshape(-1, 3)
            dist_flat = dist[valid].reshape(-1)
        else:
            diff_flat = diff.reshape(-1, 3)
            dist_flat = dist.reshape(-1)
            valid_flat = valid.reshape(-1)
            diff_flat = diff_flat[valid_flat]
            dist_flat = dist_flat[valid_flat]
        all_diffs.append(diff_flat)
        all_dists.append(dist_flat)

    all_diffs = torch.cat(all_diffs, dim=0)
    all_dists = torch.cat(all_dists, dim=0)
    return all_diffs, all_dists


def main():
    args = parse_args()
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
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    print(f"Dataset length: {len(dataset)}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

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

    all_diffs_list = []
    all_dists_list = []
    all_dists_cam_list = []  # 仅当 --unify_frame 时使用
    per_sample_stats = []

    num_done = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Samples"):
            if num_done >= args.num_samples:
                break
            images = batch['images'].to(device)
            S = images.shape[1]

            aggregated_tokens_list, image_tokens_list, patch_start_idx, time_varient_pts, _ = \
                model.get_time_varient_pts(images)

            decoded_tokens_list_list = []
            time_invarient_pts_list = []
            for idx in range(S):
                decoded_tokens_list, time_invarient_pts, _ = model.get_time_invarient_pts(
                    images, aggregated_tokens_list, patch_start_idx, idx
                )
                decoded_tokens_list_list.append(decoded_tokens_list)
                time_invarient_pts_list.append(time_invarient_pts)

            diffs, dists = compute_diff_dist_stats(time_invarient_pts_list, time_invarient_pts_list[0], valid_only=True)
            diffs_np = diffs.cpu().float().numpy()
            dists_np = dists.cpu().float().numpy()

            all_diffs_list.append(diffs_np)
            all_dists_list.append(dists_np)

            # 单样本统计
            n = dists_np.size
            if n > 0:
                stats = {
                    'n_pixels': n,
                    'dist_min': float(np.min(dists_np)),
                    'dist_max': float(np.max(dists_np)),
                    'dist_mean': float(np.mean(dists_np)),
                    'dist_std': float(np.std(dists_np)),
                    'dist_p50': float(np.percentile(dists_np, 50)),
                    'dist_p90': float(np.percentile(dists_np, 90)),
                    'dist_p95': float(np.percentile(dists_np, 95)),
                    'dist_p99': float(np.percentile(dists_np, 99)),
                    'diff_x_mean': float(np.mean(np.abs(diffs_np[:, 0]))),
                    'diff_y_mean': float(np.mean(np.abs(diffs_np[:, 1]))),
                    'diff_z_mean': float(np.mean(np.abs(diffs_np[:, 2]))),
                }
                per_sample_stats.append(stats)

            num_done += 1

    # 全局统计
    all_diffs_np = np.concatenate(all_diffs_list, axis=0)
    all_dists_np = np.concatenate(all_dists_list, axis=0)
    n_total = all_dists_np.size

    txt_path = os.path.join(args.out_dir, 'diff_dist_stats.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== Dynamic mask 点云 diff/dist 统计（用于设定 threshold / temperature）===\n\n")
        f.write("坐标系来源（代码中无显式标注）：\n")
        f.write("  - time_varient_pts: point_head(aggregated_tokens_list)，聚合多帧，渲染时作 world 使用。\n")
        f.write("  - time_invarient_pts: point_head(decoded_tokens_list)，decoder 按帧 condition，设计上亦为 world。\n")
        f.write("  若 dist 统计与预期不符，可加 --unify_frame 0 在「统一到第 0 帧相机系」后重算 dist 对比。\n\n")
        f.write(f"样本数: {args.num_samples}, 总有效像素数: {n_total}\n\n")

        f.write("--- dist (L2 距离) 全局统计 ---\n")
        f.write(f"  min   = {all_dists_np.min():.6f}\n")
        f.write(f"  max   = {all_dists_np.max():.6f}\n")
        f.write(f"  mean  = {all_dists_np.mean():.6f}\n")
        f.write(f"  std   = {all_dists_np.std():.6f}\n")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            f.write(f"  p{p:02d}   = {np.percentile(all_dists_np, p):.6f}\n")

        f.write("\n--- diff (x,y,z) 全局统计（绝对值）---\n")
        for i, name in enumerate(['x', 'y', 'z']):
            col = np.abs(all_diffs_np[:, i])
            f.write(f"  |diff_{name}|: min={col.min():.6f}, max={col.max():.6f}, mean={col.mean():.6f}, std={col.std():.6f}\n")

        f.write("\n--- 每个样本的 dist 简要统计 ---\n")
        for i, s in enumerate(per_sample_stats):
            f.write(f"  sample {i}: n={s['n_pixels']}, dist mean={s['dist_mean']:.6f}, p95={s['dist_p95']:.6f}, p99={s['dist_p99']:.6f}\n")

        f.write("\n--- 建议 ---\n")
        f.write("  get_dynamic_mask(threshold=...) 可参考 dist 的百分位数：\n")
        f.write("  - 若希望少标 dynamic：threshold 取大一些（如 p95 或 p99 附近）。\n")
        f.write("  - 若希望多标 dynamic：threshold 取小一些（如 p50 或 p75 附近）。\n")
        f.write("  temperature 控制 sigmoid 在 threshold 附近的陡峭程度，可先用 0.01~0.02 试。\n")

        if args.unify_frame is not None and all_dists_cam_list:
            all_dists_cam_np = np.concatenate(all_dists_cam_list, axis=0)
            f.write(f"\n--- dist 在「统一到第 {args.unify_frame} 帧相机系」后的统计（用于检查坐标系是否一致）---\n")
            f.write(f"  min   = {all_dists_cam_np.min():.6f}\n")
            f.write(f"  max   = {all_dists_cam_np.max():.6f}\n")
            f.write(f"  mean  = {all_dists_cam_np.mean():.6f}\n")
            f.write(f"  p50   = {np.percentile(all_dists_cam_np, 50):.6f}, p95   = {np.percentile(all_dists_cam_np, 95):.6f}\n")
            f.write("  若与上面「原始 dist」分布差异很大，说明两路 point 可能不在同一坐标系，需在 get_dynamic_mask 前做坐标统一。\n")

    print(f"Stats written to {txt_path}")

    # 保存 dist 直方图数据，便于自行画图
    hist, bin_edges = np.histogram(all_dists_np, bins=100)
    np.savez(
        os.path.join(args.out_dir, 'dist_histogram.npz'),
        counts=hist,
        bin_edges=bin_edges,
        dist_min=all_dists_np.min(),
        dist_max=all_dists_np.max(),
        dist_mean=all_dists_np.mean(),
        percentiles={p: float(np.percentile(all_dists_np, p)) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
    )
    print(f"Histogram data saved to {args.out_dir}/dist_histogram.npz")

    # 在终端也打一份简要统计
    print("\n--- dist 全局 ---")
    print(f"  min={all_dists_np.min():.6f}, max={all_dists_np.max():.6f}, mean={all_dists_np.mean():.6f}")
    print(f"  p50={np.percentile(all_dists_np, 50):.6f}, p90={np.percentile(all_dists_np, 90):.6f}, "
          f"p95={np.percentile(all_dists_np, 95):.6f}, p99={np.percentile(all_dists_np, 99):.6f}")


if __name__ == "__main__":
    main()
