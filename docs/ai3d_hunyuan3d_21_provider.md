# AI 3D Generation: Tencent Hunyuan3D-2.1 Local Provider

Hunyuan3D-2.1 is integrated as a premium local/server-side AI 3D generation provider. It operates via an isolated subprocess architecture to ensure that its heavy dependencies do not contaminate the main project environment.

## 1. Security & Isolation Architecture
- **Subprocess Runner**: Inference is executed using `scripts/run_hunyuan3d_21.py`. This script is called with a specific Python interpreter (via `HUNYUAN3D_21_PYTHON`).
- **Memory Separation**: Hunyuan runs in its own process. Memory is released immediately after the subprocess terminates.
- **Local-Only**: This is a `external_provider=false` implementation. Data is processed on the local machine or a private server configured by the operator. No data is sent to Tencent or other third-party APIs by this adapter.

## 2. Configuration (Environment Variables)
The provider is **disabled by default**. Five gates must be cleared:
1. `HUNYUAN3D_21_ENABLED=true`: Master switch.
2. `HUNYUAN3D_21_LEGAL_ACK=true`: Explicit acknowledgement of Tencent's license.
3. `HUNYUAN3D_21_REPO_PATH`: Path to the cloned Hunyuan3D-2.1 repository.
4. `HUNYUAN3D_21_PYTHON`: Path to the Python executable in the Hunyuan-specific virtual environment.
5. **Selection**: The provider must be explicitly selected via the UI or `AI_3D_PROVIDER` override.

### Full Setting List
| Variable | Default | Description |
|----------|---------|-------------|
| `HUNYUAN3D_21_ENABLED` | `false` | Enable the provider. |
| `HUNYUAN3D_21_LEGAL_ACK` | `false` | Consent to licensing terms. |
| `HUNYUAN3D_21_REPO_PATH` | (empty) | Absolute path to Hunyuan repo. |
| `HUNYUAN3D_21_PYTHON` | (empty) | Path to venv python. |
| `HUNYUAN3D_21_MODE` | `shape_only` | `shape_only` or `shape_and_texture`. |
| `HUNYUAN3D_21_LOW_VRAM_MODE`| `false` | Enable for cards with < 32GB VRAM. |
| `HUNYUAN3D_21_TIMEOUT_SEC` | `1800` | Max duration for generation. |
| `HUNYUAN3D_21_DEVICE` | `cuda` | Target compute device. |
| `HUNYUAN3D_21_MOCK_RUNNER` | `false` | Test-only: bypasses real inference. |

## 3. Usage in UI
- Select "Hunyuan3D-2.1 Premium Local/Server" from the provider dropdown in the AI 3D Studio.
- Note: Shape+Texture mode is significantly heavier (~29 GB VRAM recommended).

## 4. Development & Testing
To run the mock integration suite:
```bash
py -m pytest tests/test_hunyuan3d_21_provider.py
```

To test the runner script directly in mock mode:
```bash
python scripts/run_hunyuan3d_21.py --input-image test.png --output-dir out --manifest-out manifest.json --mock-runner
```

## 5. Licensing
Tencent Hunyuan3D-2.1 is subject to its own community license. Ensure compliance with region-specific usage and commercial limitations before deployment.
