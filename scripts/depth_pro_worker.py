import os
import sys
import json
import argparse
from pathlib import Path

def run_inference(image_path: str, output_dir: str, device: str = "cpu", checkpoint: str = None):
    try:
        import depth_pro
        import numpy as np
        from PIL import Image
        import cv2
        import torch

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        depth_file_path = output_path / "depth_16.png"

        # Handle custom config if checkpoint is provided
        if checkpoint:
            from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
            import dataclasses
            config = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=checkpoint)
            model, transform = depth_pro.create_model_and_transforms(config=config, device=torch.device(device))
        else:
            model, transform = depth_pro.create_model_and_transforms(device=torch.device(device))
            
        model.eval()

        image, _, f_px = depth_pro.load_rgb(image_path)
        prediction = model.infer(transform(image), f_px=f_px)
        depth = prediction["depth"].detach().cpu().numpy()

        d_min, d_max = depth.min(), depth.max()
        depth16 = ((depth - d_min) / (d_max - d_min + 1e-8) * 65535).astype("uint16")

        cv2.imwrite(str(depth_file_path), depth16)

        result = {
            "status": "ok",
            "depth_map_path": str(depth_file_path),
            "depth_format": "png16",
            "model_name": "apple/depth-pro"
        }
        print(json.dumps(result))

    except Exception as e:
        error_result = {
            "status": "error",
            "message": str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Pro Isolated Worker")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--checkpoint", help="Optional checkpoint path")
    
    args = parser.parse_args()
    run_inference(args.image, args.output, args.device, args.checkpoint)
