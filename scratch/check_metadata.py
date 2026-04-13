import numpy as np
from PIL import Image
from pathlib import Path
import json

def compare_and_metadata():
    workspace = Path(r"c:\modelPlate\data\reconstructions\job_cap_24b4136c\attempt_1_denser_frames")
    
    orig_img = workspace / "extracted_frames" / "frame_0000.jpg"
    undist_img = workspace / "dense" / "images" / "frame_0000.jpg"
    
    print("--- Brightness Comparison ---")
    if orig_img.exists():
        with Image.open(orig_img) as img:
            arr = np.array(img.convert("L"))
            print(f"Original {orig_img.name}: Mean brightness {np.mean(arr):.2f}")
    
    if undist_img.exists():
        with Image.open(undist_img) as img:
            arr = np.array(img.convert("L"))
            print(f"Undistorted {undist_img.name}: Mean brightness {np.mean(arr):.2f}")

    print("\n--- Metadata Analysis ---")
    meta_path = workspace / "extracted_frames" / "masks" / "frame_0000.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            data = json.load(f)
            print(json.dumps(data, indent=2))

if __name__ == "__main__":
    compare_and_metadata()
