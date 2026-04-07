from typing import List, Optional
from modules.asset_registry.registry import AssetRegistry
from modules.shared_contracts.models import AssetMetadata, ValidationReport

class ReviewManager:
    """
    Manages the manual review workflow for assets in 'review' status.
    """
    def __init__(self, registry: AssetRegistry):
        self.registry = registry

    def list_pending_reviews(self) -> List[AssetMetadata]:
        """
        Returns only assets that have a 'review' status (conceptually, in this factory,
        we check the registry's publish state and current validation context).
        For this simulation, we'll return assets where manual approval is False.
        """
        pending = []
        for asset_id, metadata in self.registry.assets.items():
            # In a real system, we'd check the last validation report.
            # Here we rely on the caller knowing which ones need review, 
            # or we simulate the state check.
            if not self.registry.has_approval(asset_id):
                pending.append(metadata)
        return pending

    def approve(self, asset_id: str, validation_report: ValidationReport) -> None:
        """
        Grants manual approval for an asset. 
        Only allowed if the validation report says 'review'.
        """
        if validation_report.final_decision != "review":
            raise ValueError(f"Approval failed: Asset {asset_id} is '{validation_report.final_decision}', not 'review'.")
        
        self.registry.grant_approval(asset_id, validation_report.final_decision)
        # Log action (Conceptual, telemetry will do this)

    def reject(self, asset_id: str, reason: str) -> None:
        """
        Marks an asset as failed/rejected.
        """
        # Conceptually marks the asset as terminal or needing rework
        self.registry.update_publish_state(asset_id, "rejected")
        # In this factory, rejection is a form of terminal fail for this version.

    def request_rework(self, asset_id: str, reason: str) -> None:
        """
        Marks an asset for rework.
        """
        self.registry.mark_for_rework(asset_id)
        self.registry.update_publish_state(asset_id, "needs_rework")
