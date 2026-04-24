import os
import numpy as np
import struct
from pathlib import Path

def read_colmap_depth_v4(path):
    with open(path, "rb") as f:
        header = b""
        amp_count = 0
        while amp_count < 3:
            char = f.read(1)
            if not char: break
            header += char
            if char == b"&": amp_count += 1
        parts = header.decode("ascii").split("&")
        width, height, channels = int(parts[0]), int(parts[1]), int(parts[2])
        data = np.fromfile(f, dtype=np.float32)
        return data.reshape((height, width))

def compare_densities():
    depth_dir = Path(r"c:\modelPlate\data\reconstructions\job_cap_24b4136c\attempt_1_denser_frames\dense\stereo\depth_maps")
    if not depth_dir.exists(): return
    
    photo_files = sorted(list(depth_dir.glob("*.photometric.bin")))
    geo_files = sorted(list(depth_dir.glob("*.geometric.bin")))
    
    print(f"{'Image':<20} | {'Photo%':<8} | {'Geo%':<8} | {'Ratio'}")
    print("-" * 50)
    
    for p_path, g_path in zip(photo_files[:10], geo_files[:10]):
        p_data = read_colmap_depth_v4(p_path)
        g_data = read_colmap_depth_v4(g_path)
        
        p_pct = np.sum(p_data > 0) / p_data.size
        g_pct = np.sum(g_data > 0) / g_data.size
        ratio = g_pct / p_pct if p_pct > 0 else 0
        
        print(f"{p_path.name.replace('.photometric.bin',''):<20} | {p_pct:>7.2%} | {g_pct:>7.2%} | {ratio:>7.2f}")

if __name__ == "__main__":
    compare_densities()
