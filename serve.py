"""Start the local Fall Review application: python -X utf8 serve.py."""
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local fall detection review workspace")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pose-engine", choices=("mmpose",), default="mmpose")
    args = parser.parse_args()
    os.environ["FALL_DEVICE"] = args.device
    os.environ["FALL_POSE_ENGINE"] = args.pose_engine
    import uvicorn
    uvicorn.run("service.app:create_app", factory=True, host="127.0.0.1", port=args.port)
