
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from modules.operations.settings import settings

print("=" * 50)
print(f"Project Root:     {project_root}")
print(f"Environment:      {getattr(settings.env, 'value', settings.env)}")
print(f"Pipeline (pre):   {settings.recon_pipeline}")

settings.recon_pipeline = "colmap_openmvs"
print(f"Pipeline (post):  {settings.recon_pipeline}")
print("=" * 50)
