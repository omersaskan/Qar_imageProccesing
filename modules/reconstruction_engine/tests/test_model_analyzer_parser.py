import pytest
from modules.reconstruction_engine.adapter import COLMAPAdapter

def test_parse_analyzer_output_legacy():
    adapter = COLMAPAdapter(engine_path="fake")
    output = """
    Registered images: 29
    Points3D: 4187
    """
    stats = adapter._parse_analyzer_output(output)
    assert stats["registered_images"] == 29
    assert stats["points_3d"] == 4187

def test_parse_analyzer_output_legacy_with_colon_whitespace():
    adapter = COLMAPAdapter(engine_path="fake")
    output = """
    Registered images : 29
    Points3D : 4187
    """
    stats = adapter._parse_analyzer_output(output)
    assert stats["registered_images"] == 29
    assert stats["points_3d"] == 4187

def test_parse_analyzer_output_colmap_403():
    adapter = COLMAPAdapter(engine_path="fake")
    output = """
    I20260413 06:11:14.326016  9288 model.cc:446] Registered images: 29
    I20260413 06:11:14.326023  9288 model.cc:448] Points: 4187
    """
    stats = adapter._parse_analyzer_output(output)
    assert stats["registered_images"] == 29
    assert stats["points_3d"] == 4187

def test_parse_analyzer_output_ignores_observations():
    adapter = COLMAPAdapter(engine_path="fake")
    output = """
    I20260413 06:11:14.326023  9288 model.cc:448] Points: 4187
    I20260413 06:11:14.326030  9288 model.cc:449] Observations: 26356
    """
    stats = adapter._parse_analyzer_output(output)
    assert stats["points_3d"] == 4187
    # Ensure it didn't overwrite with observations
    assert stats["points_3d"] != 26356

def test_parse_analyzer_output_mixed_and_messy():
    adapter = COLMAPAdapter(engine_path="fake")
    output = """
    random noise line
    [LOG] Registered  images   :    10
    Some other line
    Points3D: 500
    More noise
    """
    stats = adapter._parse_analyzer_output(output)
    assert stats["registered_images"] == 10
    assert stats["points_3d"] == 500

def test_parse_analyzer_output_empty():
    adapter = COLMAPAdapter(engine_path="fake")
    stats = adapter._parse_analyzer_output("")
    assert stats["registered_images"] == 0
    assert stats["points_3d"] == 0

def test_parse_analyzer_output_invalid_values():
    adapter = COLMAPAdapter(engine_path="fake")
    output = """
    Registered images: NaN
    Points: infinity
    """
    stats = adapter._parse_analyzer_output(output)
    assert stats["registered_images"] == 0
    assert stats["points_3d"] == 0
