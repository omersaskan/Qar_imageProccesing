
import traceback
from modules.operations.settings import settings, AppEnvironment
from modules.reconstruction_engine.runner import ReconstructionRunner

print(f"Current settings.env: {settings.env} (type: {type(settings.env)})")
settings.env = "pilot"
print(f"New settings.env: {settings.env} (type: {type(settings.env)})")

runner = ReconstructionRunner()
try:
    adapter = runner.adapter
    print("Runner.adapter accessed successfully")
except Exception:
    traceback.print_exc()
