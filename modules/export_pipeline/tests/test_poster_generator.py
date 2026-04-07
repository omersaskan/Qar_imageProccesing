import os
import shutil
import pytest
from pathlib import Path
from PIL import Image
from modules.export_pipeline.poster_generator import PosterGenerator

def test_generate_poster():
    generator = PosterGenerator()
    product_id = "test_product_123"
    output_path = "temp_test_outputs/poster.png"
    
    artifact = generator.generate_poster(product_id, output_path)
    
    assert os.path.exists(output_path)
    assert artifact.artifact_type == "poster"
    assert artifact.file_format == "PNG"
    assert artifact.file_size_bytes > 0
    
    # Verify it's a real PNG and correct size
    with Image.open(output_path) as img:
        assert img.size == (1024, 1024)
        assert img.format == "PNG"

def test_generate_thumbnail():
    generator = PosterGenerator()
    product_id = "test_product_123"
    output_path = "temp_test_outputs/thumb.png"
    
    artifact = generator.generate_thumbnail(product_id, output_path)
    
    assert os.path.exists(output_path)
    assert artifact.artifact_type == "thumbnail"
    
    with Image.open(output_path) as img:
        assert img.size == (256, 256)
        assert img.format == "PNG"

@pytest.fixture(autouse=True)
def cleanup():
    # Setup
    if os.path.exists("temp_test_outputs"):
        shutil.rmtree("temp_test_outputs")
    os.makedirs("temp_test_outputs", exist_ok=True)
    
    yield
    
    # Teardown
    if os.path.exists("temp_test_outputs"):
        shutil.rmtree("temp_test_outputs")
