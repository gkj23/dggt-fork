"""
@file   extract_masks.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Extract semantic mask

Using SegFormer, 2021. Cityscapes 83.2%
Relies on timm==0.3.2 & pytorch 1.8.1 (buggy on pytorch >= 1.9)

Installation:
    NOTE: mmcv-full==1.2.7 requires another pytorch version & conda env.
        Currently mmcv-full==1.2.7 does not support pytorch>=1.9; 
            will raise AttributeError: 'super' object has no attribute '_specify_ddp_gpu_num'
        Hence, a seperate conda env is needed.

    git clone https://github.com/NVlabs/SegFormer

    conda create -n segformer python=3.8
    conda activate segformer
    # conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

    pip install timm==0.3.2 pylint debugpy opencv-python attrs ipython tqdm imageio scikit-image omegaconf
    pip install mmcv-full==1.2.7 --no-cache-dir
    
    cd SegFormer
    pip install .

Usage:
    Direct run this script in the newly set conda env.

Multi-GPU:
    python extract_masks.py --num_workers 4 --devices cuda:0,cuda:1,cuda:2,cuda:3 ...
    不指定 --devices 时默认使用 cuda:0, cuda:1, ... 与 num_workers 对应。
"""

from mmseg.apis import inference_segmentor, init_segmentor
import os
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
import multiprocessing
from multiprocessing import Process

semantic_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

dataset_classes_in_sematic = {
    'Road': [0],
    'Building': [2],
    'Vegetation': [8],
    'Vehicle': [13, 14, 15],
    'Person': [11],
    'Cyclist': [12, 17, 18],
    'Traffic Sign': [9],
    'Sidewalk': [1],
    'Sky': [10],
    'Other': []
}

CLASS_VALUE_MAP = {
    'Road': 10,
    'Building': 20,
    'Vegetation': 30,
    'Vehicle': 40,
    'Person': 50,
    'Cyclist': 60,
    'Traffic Sign': 70,
    'Sidewalk': 80,
    'Sky': 90,
    'Other': 255
}


def _process_one_scene(scene_id, model, w):
    """处理单个 scene 下的所有图像。w 为 worker 参数字典。"""
    scene_id = str(scene_id).zfill(3)
    data_root = w["data_root"]
    rgb_dirname = w["rgb_dirname"]
    process_dynamic_mask = w["process_dynamic_mask"]
    ignore_existing = w["ignore_existing"]

    img_dir = os.path.join(data_root, scene_id, rgb_dirname)
    flist = sorted(glob(os.path.join(img_dir, '*')))
    if not flist:
        return

    sky_mask_dir = os.path.join(data_root, scene_id, "sky_masks")
    os.makedirs(sky_mask_dir, exist_ok=True)

    if process_dynamic_mask:
        rough_human_mask_dir = os.path.join(data_root, scene_id, "dynamic_masks", "human")
        rough_vehicle_mask_dir = os.path.join(data_root, scene_id, "dynamic_masks", "vehicle")
        all_mask_dir = os.path.join(data_root, scene_id, "fine_dynamic_masks", "all")
        human_mask_dir = os.path.join(data_root, scene_id, "fine_dynamic_masks", "human")
        vehicle_mask_dir = os.path.join(data_root, scene_id, "fine_dynamic_masks", "vehicle")
        os.makedirs(all_mask_dir, exist_ok=True)
        os.makedirs(human_mask_dir, exist_ok=True)
        os.makedirs(vehicle_mask_dir, exist_ok=True)

    custom_mask_dir = os.path.join(data_root, scene_id, "custom_masks")
    os.makedirs(custom_mask_dir, exist_ok=True)

    for fpath in flist:
        fbase = os.path.splitext(os.path.basename(fpath))[0]
        mask_fpath = os.path.join(custom_mask_dir, f"{fbase}.png")
        if ignore_existing and os.path.exists(mask_fpath):
            continue

        result = inference_segmentor(model, fpath)
        mask = result[0].astype(np.uint8)

        sky_mask = np.isin(mask, [10])
        imageio.imwrite(os.path.join(sky_mask_dir, f"{fbase}.png"), sky_mask.astype(np.uint8) * 255)

        custom_mask = np.full_like(mask, CLASS_VALUE_MAP['Other'], dtype=np.uint8)
        for class_name, ids in dataset_classes_in_sematic.items():
            if class_name == 'Other':
                continue
            for city_id in ids:
                custom_mask[mask == city_id] = CLASS_VALUE_MAP[class_name]
        imageio.imwrite(mask_fpath, custom_mask)

        if process_dynamic_mask:
            rough_human_mask_path = os.path.join(rough_human_mask_dir, f"{fbase}.png")
            rough_vehicle_mask_path = os.path.join(rough_vehicle_mask_dir, f"{fbase}.png")

            if os.path.exists(rough_human_mask_path):
                rough_human_mask = imageio.imread(rough_human_mask_path) > 0
                human_mask = np.isin(mask, dataset_classes_in_sematic['Person'] + dataset_classes_in_sematic['Cyclist'])
                valid_human_mask = np.logical_and(human_mask, rough_human_mask)
                imageio.imwrite(os.path.join(human_mask_dir, f"{fbase}.png"), valid_human_mask.astype(np.uint8) * 255)
            else:
                valid_human_mask = np.zeros_like(mask, dtype=bool)

            if os.path.exists(rough_vehicle_mask_path):
                rough_vehicle_mask = imageio.imread(rough_vehicle_mask_path) > 0
                vehicle_mask = np.isin(mask, dataset_classes_in_sematic['Vehicle'])
                valid_vehicle_mask = np.logical_and(vehicle_mask, rough_vehicle_mask)
                imageio.imwrite(os.path.join(vehicle_mask_dir, f"{fbase}.png"), valid_vehicle_mask.astype(np.uint8) * 255)
            else:
                valid_vehicle_mask = np.zeros_like(mask, dtype=bool)

            valid_all_mask = np.logical_or(valid_human_mask, valid_vehicle_mask)
            imageio.imwrite(os.path.join(all_mask_dir, f"{fbase}.png"), valid_all_mask.astype(np.uint8) * 255)


def _worker_process(scene_ids_chunk, device, worker_args):
    """子进程入口：在指定 device 上加载模型并处理分配到的 scene。
    依赖主进程在 spawn 前已设置 CUDA_VISIBLE_DEVICES，子进程内只用 cuda:0。
    """
    gpu_id = worker_args.get("gpu_id", 0)
    # 子进程内只看到一张卡（由主进程设好环境变量），统一用 cuda:0
    model = init_segmentor(worker_args["config"], worker_args["checkpoint"], device="cuda:0")
    for scene_id in tqdm(scene_ids_chunk, desc=f"Worker GPU{gpu_id}", position=worker_args.get("worker_rank", 0)):
        _process_one_scene(scene_id, model, worker_args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/waymo/processed/training')
    parser.add_argument("--scene_ids", default=None, type=int, nargs="+")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_scenes", type=int, default=200)
    parser.add_argument('--process_dynamic_mask', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--no_compress', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--mask_dirname', type=str, default="fine_dynamic_masks")
    parser.add_argument('--segformer_path', type=str, default='/home/guojianfei/ai_ws/SegFormer')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--palette', default='cityscapes')
    parser.add_argument('--num_workers', type=int, default=1, help='并行进程数（多卡时建议等于 GPU 数）')
    parser.add_argument('--devices', type=str, default=None, help='逗号分隔的 GPU 列表，如 cuda:0,cuda:1；默认根据 num_workers 用 cuda:0, cuda:1, ...')

    args = parser.parse_args()

    if args.config is None:
        args.config = os.path.join(args.segformer_path, 'local_configs', 'segformer', 'B5', 'segformer.b5.1024x1024.city.160k.py')
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.segformer_path, 'pretrained', 'segformer.b5.1024x1024.city.160k.pth')

    if args.scene_ids is not None:
        scene_ids_list = list(args.scene_ids)
    elif args.split_file is not None:
        lines = open(args.split_file, "r").readlines()[1:]
        if "kitti" in args.split_file or "nuplan" in args.split_file:
            scene_ids_list = [line.strip().split(",")[0] for line in lines]
        else:
            scene_ids_list = [int(line.strip().split(",")[0]) for line in lines]
    else:
        scene_ids_list = list(np.arange(args.start_idx, args.start_idx + args.num_scenes))

    num_workers = max(1, args.num_workers)
    if args.devices is not None:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        devices = [f"cuda:{i}" for i in range(num_workers)]
    # 解析 GPU 编号：cuda:0 -> 0, cuda:1 -> 1
    def _device_to_id(d):
        if d.startswith("cuda:"):
            return int(d.split(":")[1])
        return 0
    gpu_ids = [_device_to_id(d) for d in devices]
    num_workers = min(num_workers, len(devices), len(scene_ids_list))
    if num_workers > len(scene_ids_list):
        num_workers = len(scene_ids_list)
    devices = devices[:num_workers]
    gpu_ids = gpu_ids[:num_workers]

    worker_args = {
        "data_root": args.data_root,
        "rgb_dirname": args.rgb_dirname,
        "process_dynamic_mask": args.process_dynamic_mask,
        "ignore_existing": args.ignore_existing,
        "config": args.config,
        "checkpoint": args.checkpoint,
    }

    if num_workers <= 1:
        # 单进程：与原逻辑一致
        print("Start initializing model...")
        model = init_segmentor(args.config, args.checkpoint, device=args.device)
        for scene_id in tqdm(scene_ids_list, desc="Processing Scenes"):
            _process_one_scene(scene_id, model, worker_args)
    else:
        # 多进程：spawn 前为每个子进程设置 CUDA_VISIBLE_DEVICES，否则子进程 import 时 CUDA 已初始化会报错
        multiprocessing.set_start_method("spawn", force=True)
        chunk_size = (len(scene_ids_list) + num_workers - 1) // num_workers
        chunks = [
            scene_ids_list[i * chunk_size:(i + 1) * chunk_size]
            for i in range(num_workers)
        ]
        chunks = [c for c in chunks if c]
        procs = []
        for rank, (chunk, gpu_id) in enumerate(zip(chunks, gpu_ids)):
            w = dict(worker_args, worker_rank=rank, gpu_id=gpu_id)
            # 必须在 start() 前设置，子进程继承后 import torch 时只能看到这一张卡
            env_backup = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            p = Process(target=_worker_process, args=(chunk, None, w))
            p.start()
            if env_backup is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = env_backup
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            procs.append(p)
        for p in procs:
            p.join()
        print("All workers finished.")