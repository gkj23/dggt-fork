import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_vdpm_local():
    # 1. 设置目标路径为当前目录下的 pretrain 文件夹
    save_dir = Path("./pretrain")
    save_dir.mkdir(parents=True, exist_ok=True)
    target_path = save_dir / "vdpm_model.pt"
    
    url = "https://huggingface.co/edgarsucar/vdpm/resolve/main/model.pt"
    
    if target_path.exists():
        print(f"文件已存在: {target_path}")
        return

    print(f"正在下载到本地目录: {target_path}")
    try:
        # 增加 stream=True 处理大文件
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f, tqdm(
            desc="下载进度",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = f.write(data)
                bar.update(size)
                
        print(f"\n下载完成！路径: {target_path}")
        
    except Exception as e:
        print(f"\n下载出错: {e}")
        if target_path.exists():
            target_path.unlink()

if __name__ == "__main__":
    download_vdpm_local()