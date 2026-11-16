# Original baseline asset reference

The app now targets the original pipeline locally in Ubuntu/WSL. Use [the local baseline guide](BASELINE_LOCAL.md) for installation and launch. The paths below document where the imported research assets came from; deploying the app to the training server is unnecessary.

The existing `best_erm.pt` is sufficient for the trained fall-classifier stage. It matches the bundled URFD/2026 ERM checkpoint and includes its architecture/configuration and the source-selected threshold of 0.20. No additional classifier checkpoint or training dataset is required to analyze a new video with ERM.

The complete pipeline also includes a frozen person detector and pose estimator. The portable option was introduced for CPU video analysis without the missing MMCV runtime; it is a separate extraction configuration, not a reproduction of the benchmark preprocessing.

| Stage | Original benchmark | Portable option |
|---|---|---|
| Person detector | RTMDet-m, COCO checkpoint `229f527c` | YOLOX-tiny, HumanArt export `6f3252f9` |
| Pose estimator | RTMPose-m, AIC/COCO checkpoint `63eb25f7` | RTMPose-m, body7 export `e48f03d0` |
| Runtime/decoding | Original MMPose/MMDetection pipeline | RTMLib + ONNX Runtime; different decoding implementation |
| Fall classifier | User's URFD/2026 `best_erm.pt` | The same checkpoint |
| Normalization/windowing/aggregation | Image normalization, 15 fps, 64/32 windows, top-two logits | Same downstream rules, applied to different extracted poses |

Different detections, joints and response scores can change the classifier input and prediction. The published benchmark applies to the original extraction path. Keeping the same skeleton layout or classifier architecture does not establish equivalent performance.

## Research assets and their source paths

The original workspace was reported as `projects/fall_detection_project/research/`. Paths below are relative to the working directory/config used for extraction. Locate the actual configured data directory if it differs.

1. **`data/model_weights/`**, especially these original files and their `.sha256` sidecars:

   ```text
   rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth
   rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth.sha256
   rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
   rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth.sha256
   ```

   Place them in the local project's `data/model_weights/`, directly, not inside `portable/`. The original extractor can download the pinned public weights automatically, so copying them is optional for online operation. Copying the working server files and sidecars preserves a known byte identity and avoids another download.

2. **`results/runs/urfd_2026/config.yaml`, `environment.json` and `provenance.json`** for the exact run settings and traceability. The classifier already embeds its configuration; these files help verify/reconstruct the runtime. They do not create missing compiled dependencies by themselves.

3. **An export of the actual working Python environment.** Activate the environment used for pose extraction before recording it. Capture `python -m pip freeze` and, if it is Conda-managed, `conda env export --no-builds`. Include any custom package patches or installation notes. The packaged environment records Python 3.11.18, PyTorch 2.9.1, NumPy 1.26.4, MMPose 1.3.2, MMDetection 3.3.0, MMEngine 0.10.7, MMCV 2.1.0 and CUDA runtime 12.8. See `benchmarks/erm/environment.json` for the recorded environment.

4. **Optional verification sample:** one original `.npz` pose cache and its corresponding known ERM prediction/video ID. A matching recording can verify that copied weights, preprocessing and inference reproduce its saved result before testing new clips.

Do not copy a Linux `site-packages` directory or virtual environment into Windows and expect its compiled MMCV/CUDA extensions to work. Reconstruct a compatible environment on the target platform, or keep inference in the already-working server environment. The raw training datasets, optimizer state and all twelve classifier runs are unnecessary for ordinary ERM inference.

The imported `swad.pt` is a legacy average. To use SWAD, repair it in the original complete workspace and retrieve the corrected checkpoint with its updated provenance/repair record. The two imported `.pt` files alone do not contain the original SWAD replay history. ERM does not need this repair.

## Imported weights and server runtime check

Both original `.pth` weights and their sidecars were subsequently supplied in `copy_assets/model_weights/`, moved into `data/model_weights/`, and verified before and after moving. The RTMDet-m SHA-256 is `229f527ca88498e8894a778a62a878a322b4a3ea2cae09ea537d34b7e907792b`; the RTMPose-m SHA-256 is `5e55be2a03f6e5dcd14d088afc4ae5afe94a4f9de93c22e5deb725ad0eee899d`. The original weight files are now present locally. MMPose/MMCV and the other original pose packages remain absent from the local Windows Python environment.

The server shell showed both `.venv-openai-study` and Conda environment markers. That raises an interpreter-selection question but does not prove whether MMCV is installed in `datpt`. An import statement that imports MMCV before printing `sys.executable` cannot resolve that question when the import fails. Run the explicit interpreter first:

```bash
/opt/miniconda3/envs/datpt/bin/python -c "import sys; print(sys.executable, flush=True); import mmcv, mmpose; from mmcv.ops import nms; print('Original pose runtime imports and MMCV ops: OK')"
```

If this still reports a missing module, it is missing from that interpreter's environment too. Check `datpt_mtl` by using `/opt/miniconda3/envs/datpt_mtl/bin/python` in the same command. A successful import of `mmcv.ops.nms` checks compiled-op loading in addition to the package name. It does not replace a real detector/pose inference check. Use the confirmed working interpreter for `-m pip freeze` rather than relying on whichever `python` the stacked shell resolves.

## Run locally with the original pose engine

Launch `start-baseline.cmd` from the local project folder. It uses the isolated local Ubuntu/WSL runtime and the project on `D:`. See [local setup and verification](BASELINE_LOCAL.md). No app copy on the training server or SSH setup is needed.

## Response-score validation fix

The app formerly rejected pose scores above 1. That was an application validation error: these SimCC keypoint scores are response peaks, not bounded probabilities. The original `src/data.py:load_video` preserves the score channel and applies the invalid-joint mask; the app now does the same and rejects non-finite coordinates/scores separately. It does not clip or renormalize responses.

The saved failed caches for `01.mp4` and `How to fake fall.mp4` contained finite maximum responses of 1.016702 and 1.129709 respectively. Both are accepted by the corrected validation. This bug is independent of the question of whether the portable extractor reproduces the benchmark. RTMLib's [RTMPose decoder](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/pose_estimation/rtmpose.py) and MMPose's [SimCC postprocessing](https://github.com/open-mmlab/mmpose/blob/v1.3.2/mmpose/codecs/utils/post_processing.py) document the response-peak decoding path.
