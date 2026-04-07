import os
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Dict, Any, Tuple
from .export_manifest import ExportArtifact

class PosterGenerator:
    def __init__(self, bg_color: Tuple[int, int, int] = (240, 240, 240)):
        self.bg_color = bg_color

    def generate_poster(self, product_id: str, output_path: str) -> ExportArtifact:
        """
        Generates a high-resolution poster (1024x1024) PNG.
        """
        return self._generate_image(
            product_id=product_id,
            output_path=output_path,
            artifact_type="poster",
            size=(1024, 1024),
            label=f"POSTER: {product_id}"
        )

    def generate_thumbnail(self, product_id: str, output_path: str) -> ExportArtifact:
        """
        Generates a low-resolution thumbnail (256x256) PNG.
        """
        return self._generate_image(
            product_id=product_id,
            output_path=output_path,
            artifact_type="thumbnail",
            size=(256, 256),
            label=f"THUMB: {product_id}"
        )

    def _generate_image(
        self, 
        product_id: str, 
        output_path: str, 
        artifact_type: str, 
        size: Tuple[int, int], 
        label: str
    ) -> ExportArtifact:
        # Create a real PNG image
        img = Image.new('RGB', size, color=self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw a simple border and text
        draw.rectangle([10, 10, size[0]-10, size[1]-10], outline=(100, 100, 100), width=5)
        # Note: In a headless environment, default font is used
        draw.text((size[0]//2 - 50, size[1]//2), label, fill=(50, 50, 50))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, "PNG")
        
        return ExportArtifact(
            name=os.path.basename(output_path),
            artifact_type=artifact_type,
            file_path=str(output_path),
            file_format="PNG",
            file_size_bytes=os.path.getsize(output_path),
            metadata={
                "product_id": product_id,
                "width": size[0],
                "height": size[1]
            }
        )
