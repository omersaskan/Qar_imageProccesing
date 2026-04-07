import os
from pathlib import Path
import json

def create_placeholders():
    """Creates dummy .glb files in the blobs directory for demo products."""
    blobs_dir = Path("data/registry/blobs")
    blobs_dir.mkdir(parents=True, exist_ok=True)
    
    meta_dir = Path("data/registry/meta")
    if not meta_dir.exists():
        print("Meta directory not found. Please run populate_demo_data.py first.")
        return

    for file in meta_dir.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for asset_id in data.get("assets", {}):
                asset_path = blobs_dir / f"{asset_id}.glb"
                if not asset_path.exists():
                    # Create a 0-byte placeholder
                    asset_path.touch()
                    print(f"Created placeholder: {asset_id}.glb")

if __name__ == "__main__":
    create_placeholders()
