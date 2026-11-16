"""Download the two pinned portable ONNX exports into project-local storage."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.pose_runtime import MODELS, ensure_export

if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "data/model_weights/portable"
    for key in MODELS:
        path, fingerprint = ensure_export(directory, key)
        print(key, path, fingerprint, flush=True)
