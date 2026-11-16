# Original baseline on this Windows PC

The application runs locally in Ubuntu/WSL and reads this same project on `D:`. The browser opens on Windows at http://127.0.0.1:8000. No server copy or SSH connection is involved. Windows Python is independent of this runtime.

## Launch

From the project folder in PowerShell:

```powershell
.\start-baseline.cmd
```

Closing VS Code can stop a server launched through its terminal. The installed runtime stays available; run the launcher again. You can also launch it from a separate Windows PowerShell window. Stop with Ctrl+C. If another copy of the app is already running, stop it first. To use another port:

```powershell
.\start-baseline.cmd 8001
```

Reload the browser with Ctrl+F5 after an update. The environment panel must report the original baseline pose pipeline as ready.

## First installation

Ubuntu/WSL must be installed. Run:

```powershell
.\setup-baseline.cmd
```

This installs Ubuntu compiler/system libraries and creates an isolated runtime at `~/.local/share/fall-review-baseline` inside Ubuntu. It downloads a checksum-pinned Python 3.11.18 build, installs CPU PyTorch 2.9.1 and torchvision 0.24.1, and builds MMCV 2.1.0 with CPU operations against that PyTorch. MMPose 1.3.2, MMDetection 3.3.0 and MMEngine 0.10.7 match the recorded research runtime. NumPy and the research dependencies remain pinned by `requirements.txt`.

Compilation can take several minutes. An older precompiled MMCV wheel built for another PyTorch must not replace this build. `mmcv-lite` lacks the needed operations.

The final installation check loads the copied detector/pose weights, runs the actual MMPose inferencer on a bundled portrait, checks COCO-17 output, and verifies the registered ERM checkpoint. It records the successful runtime in Git-ignored `.baseline-runtime.json`.

To rerun the check inside Ubuntu from the project folder:

```bash
~/.local/share/fall-review-baseline/venv/bin/python scripts/check_baseline_runtime.py --initialize
```

## Baseline consistency

New videos use the frozen `src.pose.extract_one` path with the original RTMDet-m COCO detector and RTMPose-m AIC/COCO checkpoint. Before enabling videos, the app checks pinned package versions, an actual compiled CPU NMS invocation, MMPose imports, and SHA-256 hashes of both original `.pth` files in `data/model_weights/`.

Classification uses the verified `results/runs/urfd_2026/best_erm.pt`, its saved configuration and threshold 0.20, 15 fps, COCO-17, 64-frame windows at stride 32, and the mean of the top two window logits followed by sigmoid. Raw finite SimCC response scores are preserved, including scores above 1. No training or benchmark source file is changed.

There is no automatic extractor fallback. The standard app rejects ONNX video requests and portable caches. Imported research caches must carry the matching original `pose_config_hash`; this records the extraction configuration, not independent proof of how an arbitrary uploaded file was made. Old portable results remain in history with their existing warnings.

This is the original pipeline on **CPU**. The training machine used CUDA 12.8 and an RTX PRO 4000 Blackwell. CPU and GPU floating-point results need not be bit-for-bit identical, and CPU extraction is slower. The launcher uses four CPU threads, matching the recorded thread count. Matching the pipeline does not make arbitrary recordings part of the saved benchmark or establish accuracy on new videos.

The supplied legacy `swad.pt` remains disabled because it fails the repaired averaging checks. The ERM model is the verified baseline used by this app.

## Real-video verification

From the project folder inside Ubuntu:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ~/.local/share/fall-review-baseline/venv/bin/python scripts/verify_baseline_app.py 01.mp4 "How to fake fall.mp4"
```

This uploads each video into temporary storage, runs the real original models and ERM classifier, replays the actual model outputs through the unwrapped frozen extractor, and compares every extraction array. It also compares the classifier score with `src.data.load_video` and `src.train.video_logit`, and checks detector mean/std against the checkpoint. Results are saved after each completed video in `.runtime-cache/baseline-verification.json`. User history is independent of this check.

The old RTMDet checkpoint includes `data_preprocessor.mean` and `.std` buffers that MMEngine reports as unexpected because they are nonpersistent in the current model. The verification checks that the configured values match these checkpoint buffers exactly.
