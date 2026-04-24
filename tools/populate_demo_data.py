from modules.asset_registry.registry import AssetRegistry
from modules.shared_contracts.models import AssetMetadata
from modules.operations.telemetry import OperationalTelemetry, FailureCodes
import time

def populate():
    registry = AssetRegistry()
    telemetry = OperationalTelemetry()
    
    products = [
        ("chair_eames", "v1.0", "v1.1"),
        ("table_nordic", "v1.0", "v2.0"),
        ("lamp_industrial", "v0.9", "v1.0")
    ]
    
    for pid, v1, v2 in products:
        print(f"Populating {pid}...")
        
        # Version 1
        aid1 = f"{pid}_{int(time.time())}_1"
        meta1 = AssetMetadata(asset_id=aid1, product_id=pid, version=v1)
        registry.register_asset(meta1)
        registry.set_active_version(pid, aid1)
        registry.update_publish_state(aid1, "published")
        
        # Version 2
        time.sleep(1)
        aid2 = f"{pid}_{int(time.time())}_2"
        meta2 = AssetMetadata(asset_id=aid2, product_id=pid, version=v2)
        registry.register_asset(meta2)
        registry.update_publish_state(aid2, "review")
        
        # Log some actions
        telemetry.log_action(aid1, "production_export", {"format": "glb"})
        if "lamp" in pid:
            telemetry.log_failure("reconstruction", aid2, FailureCodes.ERR_RECON_RUNTIME, "Out of memory on voxelization")

    print("Demo data populated successfully!")

if __name__ == "__main__":
    populate()
