# Fall Review local app

The app is implemented in `service/` (FastAPI + one inference worker) and `web/` (buildless HTML/CSS/JavaScript). No Node installation or frontend build is needed. It uses the frozen research preprocessing, COCO-17 model and top-two-logit aggregation. All recording storage and analysis history are local.

## Launch

Run `start-baseline.cmd` from the project folder, then open **http://127.0.0.1:8000**. Use `start-baseline.cmd 8001` for another port. Stop with Ctrl+C. See [local baseline installation and verification](BASELINE_LOCAL.md) for `setup-baseline.cmd` and runtime details.

The app uses the isolated local Ubuntu/WSL Python 3.11 runtime. Starting `serve.py` with Windows Python will still open the UI, but video analysis remains disabled when the original runtime checks fail. New video jobs always use the original MMPose pipeline. There is no automatic ONNX fallback. Restart the backend and use Ctrl+F5 after updating.

## Imported checkpoints

The user-supplied files were relocated from the repository root to:

```text
results/runs/urfd_2026/best_erm.pt
results/runs/urfd_2026/swad.pt
```

Their reported source is `projects/fall_detection_project/research/results/runs/urfd_2026`. Import hashes are recorded in `results/runs/urfd_2026/imported_models.json`; this is an import record, not fabricated training provenance.

- **ERM:** SHA-256 `069313d00e0c9007a1381eb047dd04da9aceb0ae83e2ac8bcaf3d846f70994f8`. Matches the bundled benchmark's URFD/2026 run, legacy implementation compatibility checks, saved configuration and frozen split. The app uses its saved **0.20** decision threshold.
- **SWAD:** SHA-256 `fd4755587f0ec25ad04ac93db2ba0304a8479d89ab00a86323d9d41987f74daa`. Retained but disabled: it has no averaging-version-2 certification. Its declared state count also differs from the inclusive step range. Repair needs the original run's traces/checkpoints/data; the two imported files alone are insufficient.

The weights stay Git-ignored. The app reads only registered local checkpoints with restricted `weights_only=True` loading; video upload does not accept arbitrary model files. The application does not require or manufacture all twelve complete training-run directories.

## What works

- **Analyze:** video upload through the original baseline pose engine; original skeleton-cache upload works directly with the classifier. Persistent background jobs, stage/counter progress, cancellation, whole-recording result, saved threshold, pose coverage, warnings and processing time.
- **Playback:** native video controls with a synchronized skeleton overlay. Skeleton-only caches have their own play/seek controls. Browser video decoding depends on the source codec; use H.264 MP4 if the source AVI/MOV codec is unsupported.
- **Inspect:** clickable window-score bars, top contributing windows and model/source provenance. Window scores do not establish exact fall boundaries and are not individually supervised event predictions.
- **Export:** result/provenance JSON and window-score CSV.
- **History:** searchable persistent analyses, reopen and explicit delete. Interrupted jobs become failed/retryable on restart instead of remaining running.
- **Benchmark:** checksum-verified saved ERM metrics, per-dataset/seed filters, macro scores, confusion counts and false-positive/false-negative review. Full CSV exports are available. Benchmark videos are not bundled.
- **Environment:** model verification, pose dependency status and runtime details. A missing pose runtime does not prevent history/benchmark access or skeleton-cache analysis.

## Original pose pipeline

New videos use the original **RTMDet-m COCO detector** and **RTMPose-m AIC/COCO** checkpoint copied into `data/model_weights/`. Readiness checks validate package versions, compiled CPU NMS, MMPose imports and SHA-256 weight hashes. Results record runtime versions, processing device and both pose-weight hashes. See [baseline setup](BASELINE_LOCAL.md).

The frozen research extraction and preprocessing path is reused. CPU and the original CUDA GPU may differ numerically; baseline consistency means the same pipeline, settings and weights, not bit-for-bit GPU reproduction. The existing portable ONNX implementation is retained for diagnostic tests and reading old results. Standard app launches reject portable video jobs and portable caches. Historical portable results retain their warning that benchmark accuracy does not apply to that extractor.

## Skeleton-cache input

Select **Skeleton cache** on Analyze and choose an existing project `.npz`. A cache must contain:

| Array | Shape / meaning |
|---|---|
| `xy` | `[T,17,2]`, pixel coordinates |
| `score` | `[T,17]`, finite raw joint response scores; SimCC scores can exceed 1 and are preserved without clipping |
| `valid_joint` | `[T,17]`, boolean joint validity |
| `timestamp` | `[T]`, strictly increasing at 15 fps |
| `image_size` | `[2]`, positive integer `[width,height]` |
| `valid_frame` | Optional `[T]`; must match `valid_joint.any(axis=1)` |
| `pose_config_hash` | Required scalar; must match the original registered extraction config |

Caches without the extraction-config hash, or marked as portable, are rejected. A matching config hash is metadata and does not independently authenticate an arbitrary uploaded cache. Empty usable poses produce **Unable to analyze**, not Non-Fall. The app enforces 32 MB compressed / 256 MB expanded cache limits; videos are limited to 512 MB and ten minutes. Padding and ambiguous-person diagnostics are displayed. Only one person track is classified.

## Storage and verification

Jobs, uploads, derived pose data and SQLite history live in Git-ignored `local_app/`. Set the environment variable `FALL_STORAGE` before starting to select another local directory. The app never runs the research CLI's manifest-rewriting `inspect` command. Benchmark data and frozen training splits remain independent of uploads.

JSON files are now explicitly LF in `.gitattributes`. The working copies of `data/splits.json` and `benchmarks/erm/environment.json` were restored to their original checksum-verified bytes. CSV bytes and recorded hashes were preserved. No hash-covered research source was modified.

Automated tests cover application/research inference parity (including the supplied ERM checkpoint), preservation of raw SimCC response scores, thresholds, padding, empty poses, cache/video adapters, cancellation, restart recovery, exports, media ranges and the baseline-only policy. `node scripts/check_ui.mjs` verifies file-selection rendering and that a missing original runtime never enables a fallback. The benchmark verifier checks bundled artifact hashes and metrics.

The runtime installer performs actual original detector/pose initialization and COCO-17 inference. `scripts/verify_baseline_app.py` can additionally verify real video uploads and compare their classifier scores with the frozen research loader. Neither execution test establishes accuracy on new videos.

Browser visual QA was previously unavailable. The user has manually verified that the interface works. The responsive interface uses solid colors and no gradients.

## Original-runtime validation on 2026-11-17

The local Ubuntu/WSL runtime passed compiled NMS, original detector/pose initialization and real COCO-17 inference. The completed runtime regression suite passed **38 tests**, with the portable integration test skipped because ONNX dependencies are intentionally absent in this environment. The UI state check and packaged benchmark verifier also passed.

Both supplied videos completed original-pipeline upload jobs in temporary storage. Every extracted array matched the frozen extractor when replaying the actual model outputs, and both classifier scores matched the frozen research loader/inference path exactly on this CPU runtime:

| Recording | Sampled frames | Model score | Saved threshold | CPU processing |
|---|---:|---:|---:|---:|
| `01.mp4` | 103 | 0.9999989271 | 0.20 | 128.1 s |
| `How to fake fall.mp4` | 273 | 0.9302158952 | 0.20 | 319.9 s |

Both had usable poses in all sampled frames. These are execution/parity checks, not a labeled accuracy evaluation. The verification report is saved in `.runtime-cache/baseline-verification.json`; the first completed result was recovered from the verifier console output after VS Code closed, and the second was written by the resumed run. Existing user history was left separate from these temporary test jobs.

The Windows launcher and Windows-to-WSL localhost access were verified. `/api/readiness` reports `video_ready=true`, `engine=mmpose` and threshold 0.20. Restart with `start-baseline.cmd` if closing the IDE stops the app.
