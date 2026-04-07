import struct
from pathlib import Path

def write_placeholder_glb(output_path: str | Path):
    """Writes a minimal valid GLB binary cube (glTF 2.0) to the target path."""
    
    # 1. JSON Chunk
    json_dict = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]}],
        "bufferViews": [{"buffer": 0, "byteLength": 36}],
        "buffers": [{"byteLength": 36}]
    }
    import json
    json_bytes = json.dumps(json_dict, separators=(',', ':')).encode('ascii')
    # Pad JSON to 4-byte boundary with spaces
    while len(json_bytes) % 4 != 0:
        json_bytes += b' '
    
    # 2. Binary Chunk (Triangle coordinates)
    # 3 vertices (VEC3) = 9 floats = 36 bytes
    bin_bytes = struct.pack('<9f', 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    
    # 3. GLB Header (12 bytes)
    # Magic: 'glTF', Version: 2, Total Length
    total_len = 12 + (8 + len(json_bytes)) + (8 + len(bin_bytes))
    header = struct.pack('<4sII', b'glTF', 2, total_len)
    
    # 4. JSON Chunk Header (8 bytes)
    json_chunk_header = struct.pack('<I4s', len(json_bytes), b'JSON')
    
    # 5. BIN Chunk Header (8 bytes)
    bin_chunk_header = struct.pack('<I4s', len(bin_bytes), b'BIN\x00')
    
    # Final assembly
    glb_data = header + json_chunk_header + json_bytes + bin_chunk_header + bin_bytes
    
    with open(output_path, "wb") as f:
        f.write(glb_data)
