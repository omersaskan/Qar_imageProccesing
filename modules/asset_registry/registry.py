import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from modules.shared_contracts.models import AssetMetadata
from modules.shared_contracts.errors import DuplicateAssetError
from modules.utils.file_persistence import atomic_write_json, FileLock
from modules.utils.path_safety import ensure_dir, validate_identifier

class AssetRegistry:
    def __init__(self, data_root: str = "data/registry"):
        self.data_root = Path(data_root).resolve()
        self.meta_dir = self.data_root / "meta"
        self.active_dir = self.data_root / "active"
        self._ensure_dirs()

    def _ensure_dirs(self):
        ensure_dir(self.meta_dir)
        ensure_dir(self.active_dir)

    def _get_product_file(self, product_id: str) -> Path:
        return self.meta_dir / f"{product_id}.json"

    def _get_active_file(self, product_id: str) -> Path:
        return self.active_dir / f"{product_id}.ptr"

    def _load_product_data(self, product_id: str) -> Dict[str, Any]:
        file_path = self._get_product_file(product_id)
        if not file_path.exists():
            return {"assets": {}, "audit_logs": []}
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_no_lock(self, product_id: str, data: Dict[str, Any]):
        """Internal save without locking (caller must hold the lock)."""
        file_path = self._get_product_file(product_id)
        atomic_write_json(file_path, data)

    def _save_product_data(self, product_id: str, data: Dict[str, Any]):
        """Public save with locking."""
        file_path = self._get_product_file(product_id)
        with FileLock(file_path):
            self._save_no_lock(product_id, data)

    def _next_version_from_data(self, data: Dict[str, Any]) -> str:
        max_seen = 0
        for info in data.get("assets", {}).values():
            version = str(info.get("metadata", {}).get("version", "") or "").strip().lower()
            match = re.search(r"(\d+)", version)
            if match:
                max_seen = max(max_seen, int(match.group(1)))
        return f"v{max_seen + 1}"

    def register_asset(self, metadata: AssetMetadata) -> AssetMetadata:
        """Registers a new asset version. Atomic operation."""
        product_id = validate_identifier(metadata.product_id, "Product ID")
        asset_id = validate_identifier(metadata.asset_id, "Asset ID")
        
        file_path = self._get_product_file(product_id)
        with FileLock(file_path):
            data = self._load_product_data(product_id)
            if asset_id in data["assets"]:
                raise DuplicateAssetError(f"Asset ID {asset_id} already exists in registry.")

            stored_metadata = metadata
            if not metadata.version:
                stored_metadata = metadata.model_copy(update={"version": self._next_version_from_data(data)})

            data["assets"][asset_id] = {
                "metadata": stored_metadata.model_dump(mode="json"),
                "publish_state": "draft",
                "is_approved": False,
                "rework_required": False,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Add audit entry within the same lock
            data["audit_logs"].append({
                "asset_id": asset_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "registered",
                "details": {"version": stored_metadata.version}
            })
            
            self._save_no_lock(product_id, data)
            return stored_metadata

    def grant_approval(self, asset_id: str, validation_status: str) -> None:
        """Grants manual approval. Atomic operation."""
        if validation_status != "review":
            raise ValueError(f"Approval failed: Status '{validation_status}', not 'review'.")
        
        product_id = self._find_product_by_asset(asset_id)
        if not product_id: raise ValueError(f"Asset {asset_id} not found.")

        file_path = self._get_product_file(product_id)
        with FileLock(file_path):
            data = self._load_product_data(product_id)
            data["assets"][asset_id]["is_approved"] = True
            
            # Add audit entry within the same lock
            data["audit_logs"].append({
                "asset_id": asset_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "approved",
                "details": {"status_at_approval": validation_status}
            })
            
            self._save_no_lock(product_id, data)

    def _find_product_by_asset(self, asset_id: str) -> Optional[str]:
        for file in self.meta_dir.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if asset_id in data["assets"]:
                    return file.stem
        return None

    def has_approval(self, asset_id: str) -> bool:
        product_id = self._find_product_by_asset(asset_id)
        if not product_id: return False
        data = self._load_product_data(product_id)
        return data["assets"][asset_id].get("is_approved", False)

    def mark_for_rework(self, asset_id: str) -> None:
        product_id = self._find_product_by_asset(asset_id)
        if not product_id: return
        data = self._load_product_data(product_id)
        data["assets"][asset_id]["rework_required"] = True
        self._save_product_data(product_id, data)
        self._log_audit(asset_id, product_id, "rework_requested")

    def get_history(self, product_id: str) -> List[Dict[str, Any]]:
        """Returns full history for a product."""
        data = self._load_product_data(product_id)
        active_id = self._get_active_id(product_id)
        
        history = []
        # Sort assets by creation time
        sorted_assets = sorted(data["assets"].items(), key=lambda x: x[1]["created_at"])
        
        for asset_id, info in sorted_assets:
            asset_logs = [log for log in data["audit_logs"] if log["asset_id"] == asset_id]
            history.append({
                "asset_id": asset_id,
                "version": info["metadata"]["version"],
                "status": info["publish_state"],
                "is_active": (asset_id == active_id),
                "approved": info["is_approved"],
                "audit": asset_logs
            })
        return history

    def _get_active_id(self, product_id: str) -> Optional[str]:
        ptr_file = self._get_active_file(product_id)
        if not ptr_file.exists():
            return None
        return ptr_file.read_text(encoding="utf-8").strip()

    def set_active_version(self, product_id: str, asset_id: str) -> None:
        """Sets the active version for a product. Atomic operation."""
        data = self._load_product_data(product_id)
        if asset_id not in data["assets"]:
            raise ValueError(f"Asset {asset_id} does not belong to product {product_id} or doesn't exist.")
        
        asset_info = data["assets"][asset_id]
        metadata = asset_info.get("metadata", {})
        
        # --- Hardening Guards (Phase B Safety) ---
        if metadata.get("ai_generated"):
            raise ValueError(f"Active Pointer Rejection: Asset {asset_id} is AI-generated and cannot be set as the active production pointer.")
            
        if metadata.get("requires_manual_review") and not asset_info.get("is_approved"):
            raise ValueError(f"Active Pointer Rejection: Asset {asset_id} requires manual review and has not been approved.")
            
        if metadata.get("geometry_source", "photogrammetry") != "photogrammetry":
            raise ValueError(f"Active Pointer Rejection: Asset {asset_id} source is '{metadata.get('geometry_source')}', not 'photogrammetry'.")

        ptr_file = self._get_active_file(product_id)
        # Ensure dir exists before write
        ensure_dir(ptr_file.parent)
        with FileLock(ptr_file):
            ptr_file.write_text(asset_id, encoding="utf-8")

    def publish_asset(self, product_id: str, asset_id: str) -> None:
        """
        Atomically marks an asset as published and sets it as active.
        
        Hardening Guards:
        1. Rejects metadata.ai_generated=True (Phase B safety).
        2. Rejects requires_manual_review=True unless is_approved is True.
        """
        product_file = self._get_product_file(product_id)
        ptr_file = self._get_active_file(product_id)
        
        # We lock the product metadata file as the primary source of truth
        with FileLock(product_file):
            data = self._load_product_data(product_id)
            if asset_id not in data["assets"]:
                raise ValueError(f"Asset {asset_id} not found in product {product_id}")
            
            asset_info = data["assets"][asset_id]
            metadata = asset_info.get("metadata", {})
            
            # --- Hardening Guards ---
            # 1. AI Generated Guard (Phase B safety)
            if metadata.get("ai_generated"):
                raise ValueError(f"Publishing rejected: Asset {asset_id} is AI-generated. Phase B assets are review-only and cannot be published to production.")
            
            # 2. Manual Review Guard
            if metadata.get("requires_manual_review") and not asset_info.get("is_approved"):
                raise ValueError(f"Publishing rejected: Asset {asset_id} requires manual review and has not been approved.")

            # 3. Geometry Source Guard
            if metadata.get("geometry_source", "photogrammetry") != "photogrammetry":
                 raise ValueError(f"Publishing rejected: Asset {asset_id} source is '{metadata.get('geometry_source')}', not 'photogrammetry'.")

            # 1. Update publish state
            data["assets"][asset_id]["publish_state"] = "published"
            
            # 2. Add audit entry
            data["audit_logs"].append({
                "asset_id": asset_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "published",
                "details": {"set_active": True}
            })
            
            # 3. Save metadata
            self._save_no_lock(product_id, data)
            
            # 4. Update active pointer (separate file, but done while holding metadata lock)
            ensure_dir(ptr_file.parent)
            ptr_file.write_text(asset_id, encoding="utf-8")

    def rollback_version(self, product_id: str) -> Optional[str]:
        """Rolls back to the previous version if available."""
        data = self._load_product_data(product_id)
        versions = sorted(data["assets"].keys(), key=lambda x: data["assets"][x]["created_at"])
        
        if len(versions) < 2:
            return None
            
        current_active = self._get_active_id(product_id)
        if not current_active:
            new_active = versions[-1]
        else:
            try:
                idx = versions.index(current_active)
                if idx > 0:
                    new_active = versions[idx - 1]
                else:
                    return None
            except ValueError:
                new_active = versions[-2]
                
        self.set_active_version(product_id, new_active)
        self._log_audit(new_active, product_id, "rollback_applied", {"product_id": product_id})
        return new_active

    def update_publish_state(self, asset_id: str, state: str) -> None:
        """Updates the publish state (e.g., 'published', 'draft')."""
        if state == "published":
             raise ValueError("Direct state update to 'published' is prohibited. Use publish_asset() to ensure safety guards are applied.")
             
        product_id = self._find_product_by_asset(asset_id)
        if not product_id:
            raise ValueError(f"Asset {asset_id} not found")

        file_path = self._get_product_file(product_id)
        with FileLock(file_path):
            data = self._load_product_data(product_id)
            data["assets"][asset_id]["publish_state"] = state
            self._save_no_lock(product_id, data)

    def get_asset(self, asset_id: str) -> Optional[AssetMetadata]:
        product_id = self._find_product_by_asset(asset_id)
        if not product_id: return None
        data = self._load_product_data(product_id)
        asset_info = data["assets"].get(asset_id)
        if asset_info:
            return AssetMetadata.model_validate(asset_info["metadata"])
        return None

    def get_active_asset(self, product_id: str) -> Optional[AssetMetadata]:
        active_id = self._get_active_id(product_id)
        if not active_id: return None
        return self.get_asset(active_id)

    def _log_audit(self, asset_id: str, product_id: str, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Internal helper to add an audit entry to a product's history."""
        file_path = self._get_product_file(product_id)
        with FileLock(file_path):
            data = self._load_product_data(product_id)
            data["audit_logs"].append({
                "asset_id": asset_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "details": details or {}
            })
            self._save_no_lock(product_id, data)
