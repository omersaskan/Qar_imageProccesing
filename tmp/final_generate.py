import struct
import base64
import json

def generate():
    # JSON Content MUST be 4-byte aligned
    json_dict = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}] ,
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,1], "min": [0,0,0]}],
        "bufferViews": [{"buffer": 0, "byteLength": 36}],
        "buffers": [{"byteLength": 36}]
    }
    json_bytes = json.dumps(json_dict, separators=(',', ':')).encode('ascii')
    # Pad JSON with spaces to be multiple of 4
    padding = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b' ' * padding
    
    # Binary buffer (3 floating point vertices)
    # Correct positions for a triangle
    bin_bytes = struct.pack('<9f', 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    
    # Pack GLB
    # Header: Magic(0x46546C67), Version(2), Total Length
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    header = struct.pack('<4sII', b'glTF', 2, total_len)
    
    # JSON Chunk: Length, Type(0x4E4F534A), Data
    json_chunk_header = struct.pack('<I4s', len(json_bytes), b'JSON')
    
    # BIN Chunk: Length, Type(0x004E4942), Data
    bin_chunk_header = struct.pack('<I4s', len(bin_bytes), b'BIN\x00')
    
    glb = header + json_chunk_header + json_bytes + bin_chunk_header + bin_bytes
    
    # Verify magic and version in the result
    if glb[:4] != b'glTF': raise ValueError("Magic mismatch")
    if struct.unpack('<I', glb[4:8])[0] != 2: raise ValueError("Version mismatch")
    if struct.unpack('<I', glb[8:12])[0] != total_len: raise ValueError("Length mismatch")
    
    encoded = base64.b64encode(glb).decode()
    print(encoded)

if __name__ == "__main__":
    generate()
