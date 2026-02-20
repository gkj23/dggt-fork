"""
从 smpl.pkl 的 SMPL 参数恢复人体 mesh，并渲染到对应相机图像上。

依赖（需单独安装）:
  pip install smplx pytorch3d opencv-python imageio
  # SMPL 模型文件: 从 https://smpl.is.tue.mpg.de/ 注册下载，或使用 smplx 自带的
  # 将 SMPL_NEUTRAL.pkl 放到某路径，下面 SMPL_MODEL_PATH 指向该路径

用法:
  python render_smpl_on_image.py --scene_dir data/waymo/processed/training/640 --frame_id 0 --human_id 0 --cam_id 0 --out out.png
"""

import os
import json
import argparse
import numpy as np
import torch
import joblib
import cv2

# 若没有 smplx，会在此处报错，请 pip install smplx
import smplx
from pytorch3d.transforms import matrix_to_axis_angle

# Waymo 与 OpenCV 坐标系转换 (与 waymo_sourceloader 一致)
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
)
AVAILABLE_CAM_LIST = [0, 1, 2, 3, 4]


def load_scene_cameras(scene_dir, start_timestep=0, end_timestep=None):
    """加载场景内所有相机的内参和每帧的 c2w。"""
    if end_timestep is None:
        ego_dir = os.path.join(scene_dir, "ego_pose")
        n = len([f for f in os.listdir(ego_dir) if f.endswith(".txt")])
        end_timestep = n
    intrinsics = {}
    cam_to_ego = {}
    for cam_id in AVAILABLE_CAM_LIST:
        intrinsic = np.loadtxt(
            os.path.join(scene_dir, "intrinsics", f"{cam_id}.txt")
        )
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsics[cam_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        cam_to_ego[cam_id] = np.loadtxt(
            os.path.join(scene_dir, "extrinsics", f"{cam_id}.txt")
        )
        cam_to_ego[cam_id] = cam_to_ego[cam_id] @ OPENCV2DATASET

    ego_to_world_start = np.loadtxt(
        os.path.join(scene_dir, "ego_pose", f"{start_timestep:03d}.txt")
    )
    cam_to_worlds = {cam_id: [] for cam_id in AVAILABLE_CAM_LIST}
    for t in range(start_timestep, end_timestep):
        ego_to_world = np.linalg.inv(ego_to_world_start) @ np.loadtxt(
            os.path.join(scene_dir, "ego_pose", f"{t:03d}.txt")
        )
        for cam_id in AVAILABLE_CAM_LIST:
            c2w = ego_to_world @ cam_to_ego[cam_id]
            cam_to_worlds[cam_id].append(c2w)
    for cam_id in AVAILABLE_CAM_LIST:
        cam_to_worlds[cam_id] = np.stack(cam_to_worlds[cam_id], axis=0)
    return intrinsics, cam_to_worlds, start_timestep, end_timestep


def load_instances_info(scene_dir):
    """加载 instances_info，用于获取每帧每个人的 obj_to_world（平移等）。"""
    path = os.path.join(scene_dir, "instances", "instances_info.json")
    with open(path, "r") as f:
        return json.load(f)


def smpl_params_to_vertices(
    global_orient, body_pose, betas, transl,
    smpl_model_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    用 SMPL 模型根据参数得到顶点坐标 (N, 3)。

    global_orient: (1, 3, 3) 或 (3, 3) 旋转矩阵
    body_pose: (23, 3, 3) 旋转矩阵
    betas: (10,)
    transl: (3,) 世界系平移
    """
    if isinstance(global_orient, np.ndarray):
        global_orient = torch.from_numpy(global_orient).float()
    if isinstance(body_pose, np.ndarray):
        body_pose = torch.from_numpy(body_pose).float()
    if isinstance(betas, np.ndarray):
        betas = torch.from_numpy(betas).float()
    if isinstance(transl, np.ndarray):
        transl = torch.from_numpy(transl).float()

    global_orient = global_orient.to(device)
    body_pose = body_pose.to(device)
    betas = betas.to(device).unsqueeze(0)
    transl = transl.to(device).unsqueeze(0)

    # 旋转矩阵 -> axis-angle (smplx 默认用 axis-angle)
    go = matrix_to_axis_angle(global_orient.view(1, 3, 3))   # (1, 3)
    bp = matrix_to_axis_angle(body_pose.view(23, 3, 3))      # (23, 3)
    body_pose_aa = bp.unsqueeze(0)                           # (1, 23, 3)

    # smplx 需要传模型所在目录（目录内放 SMPL_NEUTRAL.pkl）
    if os.path.isfile(smpl_model_path):
        model_dir = os.path.dirname(smpl_model_path)
    else:
        model_dir = smpl_model_path
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"SMPL model dir not found: {model_dir}. "
            "Download from https://smpl.is.tue.mpg.de/ and set --smpl_model_path to the folder containing SMPL_NEUTRAL.pkl"
        )
    body_model = smplx.create(
        model_dir,
        model_type="smpl",
        gender="neutral",
        batch_size=1,
    ).to(device)
    out = body_model(
        body_pose=body_pose_aa,
        global_orient=go,
        betas=betas,
        transl=transl,
    )
    verts = out.vertices.squeeze(0)  # (V, 3)
    return verts.cpu().numpy()


def project_vertices(vertices_world, K, w2c):
    """
    vertices_world: (N, 3)
    K: (3, 3) 内参
    w2c: (4, 4) world to camera
    return: (N, 2) 像素坐标, (N,) 深度
    """
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    verts_cam = (R @ vertices_world.T).T + t
    z = verts_cam[:, 2]
    uv = (K @ verts_cam.T).T
    uv = uv[:, :2] / (uv[:, 2:3] + 1e-8)
    return uv, z


def render_mesh_on_image(
    image,
    vertices_world,
    faces,
    K,
    w2c,
    color=(0, 255, 0),
    line_width=1,
):
    """将 mesh 的边画到图像上（简单线框）。"""
    uv, z = project_vertices(vertices_world, K, w2c)
    h, w = image.shape[:2]
    # 只画在相机前方的点
    valid = z > 0.1
    uv = np.round(uv).astype(np.int32)
    for i in range(faces.shape[0]):
        a, b, c = faces[i]
        if not (valid[a] and valid[b] and valid[c]):
            continue
        pa, pb, pc = uv[a], uv[b], uv[c]
        if 0 <= pa[0] < w and 0 <= pa[1] < h or 0 <= pb[0] < w and 0 <= pb[1] < h or 0 <= pc[0] < w and 0 <= pc[1] < h:
            cv2.line(image, tuple(pa), tuple(pb), color, line_width)
            cv2.line(image, tuple(pb), tuple(pc), color, line_width)
            cv2.line(image, tuple(pc), tuple(pa), color, line_width)
    return image


def main():
    parser = argparse.ArgumentParser(description="Render SMPL mesh from pkl on image")
    parser.add_argument("--scene_dir", type=str, required=True, help="e.g. data/waymo/processed/training/640")
    parser.add_argument("--frame_id", type=int, default=0, help="frame index")
    parser.add_argument("--human_id", type=int, default=0, help="human instance id in pkl")
    parser.add_argument("--cam_id", type=int, default=0, help="camera to render onto")
    parser.add_argument("--smpl_model_path", type=str, default=".", help="dir containing SMPL_NEUTRAL.pkl or path to .pkl")
    parser.add_argument("--out", type=str, default="render_out.png", help="output image path")
    parser.add_argument("--no_instances", action="store_true", help="skip instances_info (use transl=0)")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    frame_id = args.frame_id
    human_id = args.human_id
    cam_id = args.cam_id

    # 1. 加载 pkl
    pkl_path = os.path.join(scene_dir, "humanpose", "smpl.pkl")
    smpl_dict = joblib.load(pkl_path)
    if human_id not in smpl_dict:
        raise KeyError(f"human_id {human_id} not in pkl. Keys: {list(smpl_dict.keys())}")
    ins = smpl_dict[human_id]
    if not ins["valid_mask"][frame_id]:
        print(f"Warning: frame {frame_id} has valid_mask=False for human_id {human_id}, rendering anyway.")
    global_orient = ins["smpl"]["global_orient"][frame_id]
    body_pose = ins["smpl"]["body_pose"][frame_id]
    betas = ins["smpl"]["betas"][frame_id]

    # 2. 世界系朝向与平移（与 waymo_sourceloader 一致）
    intrinsics, cam_to_worlds, start_t, end_t = load_scene_cameras(scene_dir)
    cam_depend = ins["selected_cam_idx"][frame_id].item()
    c2w = cam_to_worlds[cam_depend][frame_id - start_t]
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).float()
    global_orient_cam = torch.from_numpy(
        np.asarray(global_orient)
    ).float().view(3, 3)
    world_orient = (c2w[:3, :3] @ global_orient_cam).numpy()

    if args.no_instances:
        transl = np.zeros(3, dtype=np.float32)
    else:
        instances_info = load_instances_info(scene_dir)
        sid = str(human_id)
        if sid not in instances_info:
            transl = np.zeros(3, dtype=np.float32)
        else:
            frame_idx_list = instances_info[sid]["frame_annotations"]["frame_idx"]
            obj_to_world_list = instances_info[sid]["frame_annotations"]["obj_to_world"]
            if frame_id not in frame_idx_list:
                transl = np.zeros(3, dtype=np.float32)
            else:
                ii = frame_idx_list.index(frame_id)
                o2w = np.array(obj_to_world_list[ii]).reshape(4, 4)
                ego_start = np.loadtxt(
                    os.path.join(scene_dir, "ego_pose", f"{start_t:03d}.txt")
                )
                o2w = np.linalg.inv(ego_start) @ o2w
                transl = o2w[:3, 3].astype(np.float32)

    # 3. SMPL -> 顶点（世界系）
    verts = smpl_params_to_vertices(
        world_orient,
        np.asarray(body_pose),
        np.asarray(betas),
        transl,
        args.smpl_model_path,
    )
    # 获取 faces（smplx 模型里有）
    model_dir = os.path.dirname(args.smpl_model_path) if os.path.isfile(args.smpl_model_path) else args.smpl_model_path
    body_model = smplx.create(model_dir, model_type="smpl", gender="neutral", batch_size=1)
    faces = np.array(body_model.faces)  # (F, 3)

    # 4. 投影并渲染
    K = intrinsics[cam_id]
    w2c = np.linalg.inv(cam_to_worlds[cam_id][frame_id - start_t])
    img_path = os.path.join(scene_dir, "images", f"{frame_id:03d}_{cam_id}.jpg")
    if not os.path.isfile(img_path):
        img_path = os.path.join(scene_dir, "images", f"{frame_id:03d}_{cam_id}.png")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"No image found like {img_path}")
    image = cv2.imread(img_path)
    if image is None:
        raise RuntimeError(f"Failed to read {img_path}")
    image = render_mesh_on_image(image, verts, faces, K, w2c, color=(0, 255, 0), line_width=1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, image)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
