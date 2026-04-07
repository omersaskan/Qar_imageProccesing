from typing import Dict, Any, Optional
from modules.shared_contracts.models import AssetPackage, AssetMetadata, ValidationReport, ProductPhysicalProfile
from .registry import AssetRegistry

class PackagePublisher:
    def __init__(self, registry: AssetRegistry):
        self.registry = registry

    def publish_package(
        self, 
        product_id: str, 
        asset_id: str, 
        validation_report: ValidationReport,
        export_urls: Dict[str, str], # glb_url, usdz_url, poster_url, thumb_url
        physical_profile: ProductPhysicalProfile
    ) -> AssetPackage:
        """
        Orchestrates AssetPackage creation with a strict validation gate.
        """
        # 1. Validation Gate
        status = validation_report.final_decision
        if status == "fail":
            raise ValueError(f"Publish failed: Asset {asset_id} failed validation. Failure is terminal.")
        
        if status == "review":
            if not self.registry.has_approval(asset_id):
                raise ValueError(f"Publish failed: Asset {asset_id} is in 'review' status and lacks manual approval.")
            
        # 2. Get Metadata from Registry
        metadata = self.registry.get_asset(asset_id)
        if not metadata:
            raise ValueError(f"Metadata not found in registry for {asset_id}")
            
        # 3. Create AssetPackage
        package = AssetPackage(
            product_id=product_id,
            asset_version=metadata.version,
            glb_url=export_urls.get("glb_url", "http://missing"),
            usdz_url=export_urls.get("usdz_url", "http://missing"),
            poster_image_url=export_urls.get("poster_url", "http://missing"),
            thumbnail_url=export_urls.get("thumb_url", "http://missing"),
            bbox=metadata.bbox,
            pivot_offset=metadata.pivot_offset,
            physical_profile=physical_profile,
            validation_status=validation_report.final_decision,
            package_status="ready_for_ar"
        )
        
        # 4. Update Registry State
        self.registry.update_publish_state(asset_id, "published")
        self.registry.set_active_version(product_id, asset_id)
        
        return package
