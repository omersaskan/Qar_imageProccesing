
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from modules.operations.settings import settings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline", type=str)
args = parser.parse_args(["--pipeline", "colmap_openmvs"])

if args.pipeline:
    settings.recon_pipeline = args.pipeline
    print(f"Applied CLI pipeline override: {args.pipeline}")

print(f"Pipeline:         {settings.recon_pipeline}")
print(f"Environment:      {str(settings.env)}")
