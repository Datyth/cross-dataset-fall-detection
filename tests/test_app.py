"""Application tests use explicitly synthetic pose/video fixtures, never benchmark outputs."""
import copy
import io
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from service.app import JobStore, create_app
from service.inference import ROOT, ModelRegistry, classify_pose, validate_pose
from src.data import load_video
from src.model import CTRGCN
from src.train import video_logit
from service.pose_runtime import PORTABLE_ID, PortableInferencer, portable_config, readiness


def pose_data(frames=75, empty=False):
    t = np.arange(frames, dtype=np.float64)/15
    xy = np.zeros((frames, 17, 2), np.float32)
    xy[..., 0] = 160 + np.sin(t[:, None]*3)*25 + np.arange(17)[None, :]*2
    xy[..., 1] = 80 + np.arange(17)[None, :]*7
    return dict(xy=xy, score=np.ones((frames, 17), np.float32),
                valid_joint=np.full((frames, 17), not empty, bool),
                valid_frame=np.full(frames, not empty, bool),
                timestamp=t, image_size=np.array([320, 240]))


class FixtureRegistry:
    device = "cpu"
    def __init__(self):
        self.cfg = yaml.safe_load((ROOT/'config.yaml').read_text())
        self.cfg['model']['base_channels'] = 8
        torch.manual_seed(42)
        self.net = CTRGCN(**self.cfg['model']).eval()
        self.checkpoint = {'config': self.cfg, 'threshold': .2}
        self.entries = {'urfd_2026_erm':dict(id='urfd_2026_erm', available=True,
            name='SYNTHETIC TEST FIXTURE', threshold=.2, method='erm', fold='urfd', seed=2026)}
    def get(self, ident):
        if ident not in self.entries:
            raise ValueError('Unknown model.')
        return self.checkpoint
    def model(self, ident):
        self.get(ident)
        return self.net
    def public(self):
        return list(self.entries.values())


@pytest.fixture(scope='module')
def registry():
    torch.set_num_threads(2)
    return FixtureRegistry()


def wait_job(client, ident):
    deadline = time.monotonic()+30
    while time.monotonic()<deadline:
        job = client.get('/api/analyses/'+ident).json()
        if job['status'] in ('completed','failed','cancelled'):
            return job
        time.sleep(.03)
    raise AssertionError('Analysis did not finish within 30 seconds.')


def npz_bytes(arrays):
    out=io.BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


@pytest.mark.parametrize('length',[19,64,101])
def test_app_inference_matches_research_windows_and_saved_threshold(tmp_path, registry, length):
    arrays=pose_data(length)
    target=tmp_path/'skeletons'/'fixture'
    target.mkdir(parents=True)
    np.savez_compressed(target/'example.npz', **arrays)
    cfg=copy.deepcopy(registry.cfg)
    cfg['paths']['data']=str(tmp_path)
    windows,_=load_video(cfg, {'dataset':'fixture','video_id':'example'})
    with torch.no_grad():
        reference=float(torch.sigmoid(video_logit(registry.net,windows,'cpu',32)))
    result,_=classify_pose(registry,'urfd_2026_erm',arrays)
    assert result['score']==pytest.approx(reference,abs=1e-7)
    assert result['threshold']==.2
    assert result['decision']==('Fall' if reference>=.2 else 'Non-Fall')
    assert sum(w['contributes'] for w in result['windows'])==min(2,len(windows))
    assert max(w['end'] for w in result['windows'])<=arrays['timestamp'][-1]


def test_empty_pose_abstains_and_bad_timing_is_rejected(registry):
    result,_=classify_pose(registry,'urfd_2026_erm',pose_data(empty=True))
    assert result['score'] is None and result['decision']=='Unable to analyze'
    arrays=pose_data();arrays['timestamp']*=2
    with pytest.raises(ValueError,match='15 fps'):
        validate_pose(arrays,registry.cfg)
    arrays=pose_data();arrays['xy'][1,0,0]=np.nan
    with pytest.raises(ValueError,match='Invalid coordinates'):
        validate_pose(arrays,registry.cfg)


def test_raw_simcc_scores_above_one_are_preserved_and_match_research(tmp_path, registry):
    arrays=pose_data(64)
    arrays['score'][0,:]=1.1297085
    arrays['score'][1,0]=-.05
    arrays['valid_joint'][1,0]=False
    folder=tmp_path/'skeletons'/'fixture';folder.mkdir(parents=True)
    np.savez_compressed(folder/'responses.npz',**arrays)
    cfg=copy.deepcopy(registry.cfg);cfg['paths']['data']=str(tmp_path)
    windows,_=load_video(cfg,{'dataset':'fixture','video_id':'responses'})
    assert windows[0,2,0,0,0]==arrays['score'][0,0]
    assert windows[0,2,1,0,0]==0
    checked=validate_pose(arrays,registry.cfg)
    np.testing.assert_array_equal(checked['score'],arrays['score'])
    with torch.no_grad():
        expected=float(torch.sigmoid(video_logit(registry.net,windows,'cpu',32)))
    result,_=classify_pose(registry,'urfd_2026_erm',arrays)
    assert result['score']==pytest.approx(expected,abs=1e-7)
    arrays['score'][0,0]=np.inf
    with pytest.raises(ValueError,match='joint response scores'):
        validate_pose(arrays,registry.cfg)


def test_portable_readiness_and_explicit_original_selection(monkeypatch):
    from service import pose_runtime, baseline_runtime
    monkeypatch.setenv('FALL_POSE_ENGINE','auto')
    monkeypatch.setattr(pose_runtime.importlib.util,'find_spec',lambda n: object() if n in ('rtmlib','onnxruntime') else None)
    monkeypatch.setattr(baseline_runtime, 'runtime_status', lambda: {'available': False, 'missing': ['mmcv']})
    status=readiness()
    assert not status['available'] and status['engine']=='mmpose'
    assert readiness('onnx')['available']
    original=readiness('mmpose')
    assert not original['available']
    assert 'mmcv' in original['missing']
    assert not readiness('unknown')['available']


def test_portable_no_person_does_not_infer_a_full_image_pose():
    infer=PortableInferencer.__new__(PortableInferencer)
    infer.detector=lambda image:np.empty((0,4))
    def wrong(*args,**kwargs):
        raise AssertionError('Pose model must not run when no person is detected.')
    infer.pose=wrong
    assert list(infer([np.zeros((20,20,3),np.uint8)]))==[{'predictions':[[]]}]


def test_default_app_requires_baseline_and_verified_cache(tmp_path, registry, monkeypatch):
    import hashlib
    monkeypatch.setenv('FALL_POSE_ENGINE', 'onnx')
    with TestClient(create_app(storage=tmp_path, registry=registry)) as client:
        ready = client.get('/api/readiness').json()
        assert ready['pose']['engine'] == 'mmpose'
        assert [o['id'] for o in ready['pose']['options']] == ['mmpose']
        rejected = client.post('/api/analyses', data={'pose_engine': 'onnx'},
                              files={'file': ('video.mp4', b'fixture')})
        assert rejected.status_code == 409
        for backend in (None, PORTABLE_ID):
            arrays = pose_data()
            if backend:
                arrays['pose_backend'] = np.array(backend)
                arrays['pose_config_hash'] = np.array('portable fixture')
            response = client.post('/api/analyses', data={'kind': 'cache'},
                                   files={'file': ('cache.npz', npz_bytes(arrays))})
            assert wait_job(client, response.json()['id'])['status'] == 'failed'
        arrays = pose_data()
        arrays['pose_config_hash'] = np.array(hashlib.sha256(json.dumps(registry.cfg['pose'], sort_keys=True).encode()).hexdigest())
        response = client.post('/api/analyses', data={'kind': 'cache'},
                               files={'file': ('original.npz', npz_bytes(arrays))})
        job = wait_job(client, response.json()['id'])
        assert job['status'] == 'completed', job
        assert job['result']['pose_backend'] == 'mmpose'


def test_baseline_weights_are_checked_against_expected_bytes(tmp_path, monkeypatch):
    import hashlib
    from service import baseline_runtime
    monkeypatch.setattr(baseline_runtime, 'WEIGHTS', {'fixture.pth': hashlib.sha256(b'original').hexdigest()})
    assert not baseline_runtime.weight_status(tmp_path)['available']
    path = tmp_path/'fixture.pth'
    path.write_bytes(b'original')
    assert baseline_runtime.weight_status(tmp_path)['available']
    path.write_bytes(b'changed')
    assert not baseline_runtime.weight_status(tmp_path)['available']


def test_portable_cache_uses_distinct_hash_and_result_warning(registry):
    import hashlib
    arrays=pose_data(64)
    arrays['pose_backend']=np.array(PORTABLE_ID)
    arrays['pose_config_hash']=np.array(hashlib.sha256(json.dumps(registry.cfg['pose'],sort_keys=True).encode()).hexdigest())
    with pytest.raises(ValueError,match='different extraction configuration'):
        validate_pose(arrays,registry.cfg)
    arrays['pose_config_hash']=np.array(hashlib.sha256(json.dumps(portable_config(registry.cfg),sort_keys=True).encode()).hexdigest())
    result,_=classify_pose(registry,'urfd_2026_erm',arrays)
    assert result['pose_backend']==PORTABLE_ID
    assert any('differs from the benchmark' in w for w in result['warnings'])


def test_cache_upload_history_exports_delete_and_errors(tmp_path, registry):
    app=create_app(storage=tmp_path,registry=registry,baseline_only=False)
    with TestClient(app) as client:
        assert client.get('/').status_code==200
        assert client.get('/api/readiness').json()['cache_ready']
        invalid=client.post('/api/analyses',data={'kind':'cache'},files={'file':('bad.npz',b'bad')})
        failed=wait_job(client,invalid.json()['id'])
        assert failed['status']=='failed'
        response=client.post('/api/analyses',data={'kind':'cache'},files={'file':('../../example.npz',npz_bytes(pose_data()))})
        assert response.status_code==202
        ident=response.json()['id']
        job=wait_job(client,ident)
        assert job['status']=='completed', job
        assert job['filename']=='example.npz'
        assert client.get(f'/api/analyses/{ident}/skeleton').json()['image_size']==[320,240]
        assert client.get(f'/api/analyses/{ident}/export').json()['result']['threshold']==.2
        assert 'contributes' in client.get(f'/api/analyses/{ident}/export?format=csv').text
        assert len(client.get('/api/analyses').json())==2
        assert client.get(f'/api/analyses/{ident}/media').status_code==404
        assert client.get('/api/analyses/not-a-uuid').status_code==404
        assert client.post('/api/analyses',data={'kind':'cache'},files={'file':('model.pt',b'test')}).status_code==400
        assert client.post('/api/analyses',data={'kind':'cache'},files={'file':('empty.npz',b'')}).status_code==400
        assert client.delete(f'/api/analyses/{ident}').status_code==204
        assert client.get(f'/api/analyses/{ident}').status_code==404
        assert not (tmp_path/ident).exists()


def test_video_adapter_uses_frozen_pose_path(tmp_path, registry):
    video=tmp_path/'fixture.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),15,(320,240))
    for _ in range(65):
        writer.write(np.full((240,320,3),80,np.uint8))
    writer.release()
    def pose_factory(cfg,device):
        def infer(frames,**kwargs):
            for frame in frames:
                yield {'predictions':[[{'bbox':[50,20,250,230],'bbox_score':.99,
                    'keypoints':pose_data(1)['xy'][0].tolist(),'keypoint_scores':[.95]*17}]]}
        return infer
    app=create_app(storage=tmp_path/'jobs',registry=registry,pose_factory=pose_factory)
    with TestClient(app) as client:
        response=client.post('/api/analyses',files={'file':('fixture.avi',video.read_bytes())})
        assert response.status_code==202
        ident=response.json()['id']
        job=wait_job(client,ident)
        assert job['status']=='completed', job
        assert job['result']['pose_coverage']==1
        assert not any('unverified' in w for w in job['result']['warnings'])
        assert client.get(f'/api/analyses/{ident}/media',headers={'Range':'bytes=0-99'}).status_code==206


def test_cancel_and_restart_recovery(tmp_path,registry):
    entered,release=threading.Event(),threading.Event()
    def blocked_factory(cfg,device):
        entered.set()
        release.wait(10)
        def infer(*args,**kwargs):
            return iter(())
        return infer
    video=tmp_path/'fixture.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),15,(320,240))
    writer.write(np.zeros((240,320,3),np.uint8));writer.release()
    app=create_app(storage=tmp_path/'jobs',registry=registry,pose_factory=blocked_factory)
    with TestClient(app) as client:
        ident=client.post('/api/analyses',files={'file':('fixture.avi',video.read_bytes())}).json()['id']
        assert entered.wait(5)
        assert client.delete(f'/api/analyses/{ident}').status_code==409
        assert client.post(f'/api/analyses/{ident}/cancel').status_code==200
        release.set()
        assert wait_job(client,ident)['status']=='cancelled'
    store=JobStore(tmp_path/'restart.sqlite3')
    store.update('fixture-job',status='classifying')
    assert JobStore(tmp_path/'restart.sqlite3').get('fixture-job')['status']=='failed'


@pytest.mark.skipif(not (ROOT/'results/runs/urfd_2026/best_erm.pt').exists(),reason='User checkpoint is not distributed in Git')
def test_actual_checkpoint_identity_and_research_inference_parity(tmp_path):
    registry=ModelRegistry()
    assert registry.entries['urfd_2026_erm']['available']
    assert registry.entries['urfd_2026_erm']['threshold']==.2
    assert not registry.entries['urfd_2026_swad']['available']
    arrays=pose_data(64)
    folder=tmp_path/'skeletons'/'fixture';folder.mkdir(parents=True)
    np.savez_compressed(folder/'example.npz',**arrays)
    cfg=copy.deepcopy(registry.get('urfd_2026_erm')['config'])
    cfg['paths']['data']=str(tmp_path)
    windows,_=load_video(cfg,{'dataset':'fixture','video_id':'example'})
    with torch.no_grad():
        reference=float(torch.sigmoid(video_logit(registry.model('urfd_2026_erm'),windows,'cpu',32)))
    result,_=classify_pose(registry,'urfd_2026_erm',arrays)
    assert result['score']==pytest.approx(reference,abs=1e-7)


@pytest.mark.skipif(not (ROOT/'results/runs/urfd_2026/best_erm.pt').exists()
                    or not (ROOT/'data/model_weights/portable/pose/model.onnx').exists(),
                    reason='Local classifier and portable exports are not distributed in Git')
def test_real_portable_video_upload_with_supplied_classifier(tmp_path,monkeypatch):
    """A repeated bundled portrait tests execution, not real fall accuracy."""
    pytest.importorskip('rtmlib');pytest.importorskip('onnxruntime')
    import matplotlib
    monkeypatch.setenv('FALL_POSE_ENGINE','onnx')
    image=cv2.imread(str(Path(matplotlib.get_data_path())/'sample_data/grace_hopper.jpg'))
    assert image is not None
    height,width=image.shape[:2]
    video=tmp_path/'portrait-smoke.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),15,(width,height))
    for _ in range(20):
        writer.write(image)
    writer.release()
    app=create_app(storage=tmp_path/'jobs',baseline_only=False)
    with TestClient(app) as client:
        ready=client.get('/api/readiness').json()
        assert ready['video_ready']
        assert client.get('/favicon.svg').status_code==200
        # The original engine is still unavailable here and must fail clearly.
        if not next(o for o in ready['pose']['options'] if o['id']=='mmpose')['available']:
            original=client.post('/api/analyses',data={'pose_engine':'mmpose'},
                files={'file':('portrait-smoke.avi',video.read_bytes())})
            assert original.status_code==409
        response=client.post('/api/analyses',data={'pose_engine':'onnx'},
            files={'file':('portrait-smoke.avi',video.read_bytes())})
        assert response.status_code==202
        job=wait_job(client,response.json()['id'])
        assert job['status']=='completed',job
        assert job['result']['pose_coverage']>0
        assert job['result']['pose_backend']==PORTABLE_ID
        assert job['result']['extractor_provenance']['benchmark_equivalent'] is False
        assert job['result']['threshold']==.2
        assert job['result']['score'] is not None
