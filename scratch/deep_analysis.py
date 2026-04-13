import os
import numpy as np
from pathlib import Path
from PIL import Image

def read_colmap_depth_v4(path):
    """Reads COLMAP 4.x binary depth map format with '&' string header."""
    with open(path, "rb") as f:
        # Read header until we have 3 '&' characters
        header = b""
        amp_count = 0
        while amp_count < 3:
            char = f.read(1)
            if not char:
                break
            header += char
            if char == b"&":
                amp_count += 1
        
        parts = header.decode("ascii").split("&")
        width = int(parts[0])
        height = int(parts[1])
        channels = int(parts[2])
        
        data = np.fromfile(f, dtype=np.float32)
        return data.reshape((height, width))

def analyze_session():
    workspace = Path(r"c:\modelPlate\data\reconstructions\job_cap_24b4136c\attempt_1_denser_frames")
    dense_path = workspace / "dense"
    
    print("--- 1. Image Quality Analysis ---")
    undistorted_images = list((dense_path / "images").glob("*.jpg"))
    if not undistorted_images:
        print("No undistorted images found.")
    else:
        results = []
        for img_path in sorted(undistorted_images):
            with Image.open(img_path) as img:
                arr = np.array(img.convert("L"))
                # Blur estimation using Laplacian variance
                try:
                    import cv2
                    laplacian = cv2.Laplacian(arr, cv2.CV_64F).var()
                except:
                    # Fallback blur estimate if cv2 not available
                    laplacian = -1
                
                brightness = np.mean(arr)
                std = np.std(arr)
                results.append((img_path.name, brightness, std, laplacian))
        
        print(f"{'Image':<20} | {'Bright':<6} | {'Std':<6} | {'Blur':<6}")
        print("-" * 50)
        for name, b, s, l in results[:10]: # Show first 10
            print(f"{name:<20} | {b:>6.2f} | {s:>6.2f} | {l:>6.2f}")

    print("\n--- 2. Depth Map Analysis ---")
    depth_dir = dense_path / "stereo" / "depth_maps"
    if not depth_dir.exists():
        print("Depth maps directory not found.")
    else:
        geometric = list(depth_dir.glob("*.geometric.bin"))
        print(f"Analyzing {len(geometric)} geometric depth maps...")
        
        valid_counts = []
        for dmap_path in sorted(geometric):
            try:
                data = read_colmap_depth_v4(dmap_path)
                valid = np.sum(data > 0)
                total = data.size
                valid_counts.append(valid / total)
            except Exception as e:
                print(f"Error reading {dmap_path.name}: {e}")
        
        if valid_counts:
            print(f"Avg Valid Pct: {np.mean(valid_counts):.2%}")
            print(f"Min Valid Pct: {np.min(valid_counts):.2%}")
            print(f"Max Valid Pct: {np.max(valid_counts):.2%}")
            
            # Detailed sample
            first_map = read_colmap_depth_v4(sorted(geometric)[0])
            valid_depths = first_map[first_map > 0]
            if len(valid_depths) > 0:
                print(f"Sample Depth Range: [{np.min(valid_depths):.4f}, {np.max(valid_depths):.4f}]")

if __name__ == "__main__":
    analyze_session()
