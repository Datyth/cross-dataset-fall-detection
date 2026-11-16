// A small render-state test, not a replacement for visual browser QA.
// In particular, do not provide a global `ready`: file selection must use state.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const elements = {
  main: {innerHTML: ''},
  '#breadcrumb': {},
  '#runtime-summary': {},
  '#runtime-dot': {},
  '#runtime-dialog': {addEventListener() {}},
};
const readiness = {
  classifier_ready: true, cache_ready: true, video_ready: true,
  device: 'cpu', models: [{available: true}],
  pose: {engine: 'mmpose', available: true, options: [
    {id: 'mmpose', name: 'Original MMPose pipeline', available: true, missing: []},
  ]},
};
const context = {
  assert,
  document: {
    querySelector: selector => elements[selector] || null,
    querySelectorAll: () => [],
    addEventListener() {},
  },
  window: {addEventListener() {}},
  location: {hash: '#analyze'},
  fetch: async () => ({ok: true, status: 200, json: async () => readiness}),
  cancelAnimationFrame() {}, setTimeout, clearTimeout,
};
const source = fs.readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
const checks = `
  assert.equal(state.poseEngine, 'mmpose');
  assert.ok(main.innerHTML.includes('Original MMPose pipeline'));
  assert.ok(!main.innerHTML.includes('Portable CPU (ONNX)'));
  assert.ok(main.innerHTML.includes('data-action="upload" disabled'));
  chooseFile({name: 'recording.mp4', size: 1024});
  assert.ok(!main.innerHTML.includes('data-action="upload" disabled'));
  assert.ok(main.innerHTML.includes('Ready to analyze'));
  state.readiness.pose.options[0].available = false;
  await refreshReadiness(); renderAnalyze();
  assert.equal(state.poseEngine, 'mmpose');
  assert.ok(main.innerHTML.includes('data-action="upload" disabled'));
  assert.ok(main.innerHTML.includes('Video analysis is unavailable'));
  state.mode = 'cache'; state.file = {name: 'cache.npz', size: 1024}; renderAnalyze();
  assert.ok(!main.innerHTML.includes('data-action="upload" disabled'));
`;
await vm.runInNewContext(`(async () => {${source}\n${checks}})()`, context);
console.log('UI state checks passed: selecting a video enables original analysis; missing original runtime never falls back; cache analysis is independent.');
