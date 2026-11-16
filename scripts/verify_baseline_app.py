"""Exercise real original-pipeline uploads without altering the user's history."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from service.app import create_app
from service.inference import ModelRegistry
from service.pose_runtime import make_inferencer
from src.data import load_video
from src.pose import extract_one
from src.train import video_logit


class RecordingInferencer:
    def __init__(self, inferencer):
        self.inferencer, self.outputs = inferencer, []
        self.model_seconds = 0.
        self.provenance = inferencer.provenance

    def __call__(self, *args, **kwargs):
        iterator = iter(self.inferencer(*args, **kwargs))
        while True:
            started = time.monotonic()
            try:
                prediction = next(iterator)
            except StopIteration:
                return
            self.model_seconds += time.monotonic()-started
            self.outputs.append(copy.deepcopy(prediction))
            yield prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('videos', nargs='+', type=Path)
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    torch.set_num_threads(4)
    registry = ModelRegistry()
    cfg = copy.deepcopy(registry.get('urfd_2026_erm')['config'])
    cfg['paths']['data'] = str(ROOT/'data')
    inferencer = RecordingInferencer(make_inferencer(cfg, 'cpu', 'mmpose'))
    # Old RTMDet checkpoints include normalization buffers that newer MMEngine
    # treats as nonpersistent. Check that configured values are identical.
    detector = inferencer.inferencer.inferencer.inferencer.detector.model
    checkpoint = torch.load(ROOT/'data/model_weights'/Path(cfg['pose']['detector_checkpoint']).name,
                            map_location='cpu', weights_only=False)
    for name in ('mean', 'std'):
        torch.testing.assert_close(getattr(detector.data_preprocessor, name),
                                   checkpoint['state_dict']['data_preprocessor.'+name], rtol=0, atol=0)
    del checkpoint
    output = ROOT/'.runtime-cache/baseline-verification.json'
    output.parent.mkdir(exist_ok=True)
    reports = json.loads(output.read_text(encoding='utf-8')) if output.exists() else []
    with tempfile.TemporaryDirectory(prefix='fall-baseline-verify-') as temp:
        app = create_app(storage=Path(temp)/'jobs', registry=registry)
        app.state.manager.inferencers[('urfd_2026_erm', 'mmpose')] = inferencer
        with TestClient(app) as client:
            ready = client.get('/api/readiness').json()
            assert ready['video_ready'] and ready['pose']['engine'] == 'mmpose', ready
            for video in args.videos:
                inferencer.outputs.clear()
                inferencer.model_seconds = 0.
                with video.open('rb') as handle:
                    response = client.post('/api/analyses', files={'file': (video.name, handle, 'video/mp4')})
                assert response.status_code == 202, response.text
                ident = response.json()['id']
                deadline, last_status = time.monotonic()+args.timeout, None
                while True:
                    job = client.get('/api/analyses/'+ident).json()
                    status = (job['status'], job.get('progress'))
                    if status != last_status:
                        print(video.name, status, flush=True)
                        last_status = status
                    if job['status'] in ('completed', 'failed', 'cancelled'):
                        break
                    if time.monotonic() > deadline:
                        client.post('/api/analyses/'+ident+'/cancel')
                        raise TimeoutError(video.name)
                    time.sleep(2)
                assert job['status'] == 'completed', job
                folder = app.state.manager.folder(ident)
                with np.load(folder/'pose.npz', allow_pickle=False) as data:
                    arrays = {k: data[k] for k in data.files}
                source = folder/job['source_file']
                cap = cv2.VideoCapture(str(source))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                # Replay actual model outputs through the unwrapped frozen
                # extractor to check that app limits/progress did not change data.
                outputs = iter(inferencer.outputs)
                def replay(frames, **kwargs):
                    for _ in frames:
                        yield next(outputs)
                reference = extract_one(cfg, {'source_path': str(source), 'original_fps': fps}, replay)
                for name, expected in reference.items():
                    np.testing.assert_array_equal(arrays[name], expected, err_msg=name)
                assert next(outputs, None) is None
                data_root = Path(temp)/'reference'
                target = data_root/'skeletons'/'fixture'
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target/'video.npz', **arrays)
                ref_cfg = copy.deepcopy(cfg)
                ref_cfg['paths']['data'] = str(data_root)
                windows, _ = load_video(ref_cfg, {'dataset': 'fixture', 'video_id': 'video'})
                with torch.inference_mode():
                    score = float(torch.sigmoid(video_logit(registry.model('urfd_2026_erm'), windows, 'cpu', cfg['train']['window_chunk'])))
                result = job['result']
                assert abs(result['score']-score) <= 1e-7, (result['score'], score)
                assert result['pose_backend'] == 'mmpose'
                assert result['threshold'] == .2
                report = {k: result[k] for k in ('decision', 'score', 'threshold', 'observed_frames',
                          'pose_coverage', 'processing_seconds', 'source_sha256', 'extractor_provenance')}
                report.update(video=video.name, extraction_arrays_equal=True,
                              research_score=score, score_difference=abs(result['score']-score),
                              detector_normalization_matches_checkpoint=True,
                              pose_inference_seconds=round(inferencer.model_seconds, 3))
                reports = [r for r in reports if r['video'] != report['video']]
                reports.append(report)
                temporary = output.with_suffix('.tmp')
                temporary.write_text(json.dumps(reports, indent=2)+'\n', encoding='utf-8')
                temporary.replace(output)
                print(json.dumps(report, indent=2), flush=True)
    print('Verified:', output, flush=True)


if __name__ == '__main__':
    main()
