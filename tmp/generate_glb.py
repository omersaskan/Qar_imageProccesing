import struct
import base64

def create_valid_minimal_glb():
    # JSON Content MUST be 4-byte aligned
    json_dict = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1
                    }
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "max": [1.0, 1.0, 0.0],
                "min": [0.0, 0.0, 0.0]
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteLength": 36, "target": 34962},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963}
        ],
        "buffers": [{"byteLength": 44}]
    }
    
    import json
    json_content = json.dumps(json_dict, separators=(',', ':'))
    json_bytes = json_content.encode('utf-8')
    while len(json_bytes) % 4 != 0:
        json_bytes += b' '
    
    # Binary Content (3 vertices + 3 indices)
    # Positions: (0,0,0), (1,0,0), (0,1,0)
    pos = struct.pack('9f', 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    # Indices: 0, 1, 2
    ind = struct.pack('3H', 0, 1, 2)
    bin_bytes = pos + ind
    while len(bin_bytes) % 4 != 0:
        bin_bytes += b'\x00'
    
    # GLB Header
    # Magic (glTF), Version (2), Total Length
    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    header = struct.pack('<4sII', b'glTF', 2, total_length)
    
    # JSON Chunk
    json_chunk = struct.pack('<I4s', len(json_bytes), b'JSON') + json_bytes
    
    # BIN Chunk
    bin_chunk = struct.pack('<I4s', len(bin_bytes), b'BIN\x00') + bin_bytes
    
    glb = header + json_chunk + bin_chunk
    print(base64.b64encode(glb).decode())

if __name__ == "__main__":
    create_valid_minimal_glb()
