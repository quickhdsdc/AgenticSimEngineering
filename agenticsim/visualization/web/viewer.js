import { createScene } from './scene.js';
import { snapshot } from './state.js';

const $ = id => document.getElementById(id);
const names = { queued: '排队', processing: '加工', completed: '完成', not_arrived: '未到达' };
const sources = { synthetic_fixture: '合成输入', reference_simulation: '参考仿真', measurement: '真实测量', learned_prediction: '学习模型预测' };
let replay, run, scene, time = 0, playing = false, selected, lastFrame;

function evidence(label, value) {
  const term = document.createElement('dt'), detail = document.createElement('dd');
  term.textContent = label;
  detail.textContent = typeof value === 'string' ? value : JSON.stringify(value);
  $('evidence').append(term, detail);
}

function showSelection(id) {
  selected = id;
  const job = run?.jobs.find(j => j.id === id);
  $('selection').textContent = job ? `${job.id} · 到达 ${job.arrival.toFixed(3)} s · 开始 ${job.start.toFixed(3)} s · 离开 ${job.departure.toFixed(3)} s` : '选择一个作业。';
  render();
}

function render() {
  if (!run || run.status !== 'succeeded') return;
  const { states, counts } = snapshot(run, time);
  scene.update(states, selected);
  for (const id of ['queued', 'processing', 'completed']) $(id).textContent = counts[id];
  $('time').value = time;
  $('clock').textContent = `${time.toFixed(3)} s`;
  $('jobs').replaceChildren();
  states.forEach(job => {
    const row = document.createElement('tr');
    row.tabIndex = 0;
    row.dataset.job = job.id;
    if (job.id === selected) row.classList.add('selected');
    for (const value of [job.id, names[job.state], job.departure.toFixed(3)]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    row.addEventListener('click', () => showSelection(job.id));
    row.addEventListener('keydown', e => { if (e.key === 'Enter') showSelection(job.id); });
    $('jobs').append(row);
  });
}

function pause() { playing = false; $('play').textContent = '播放'; }

function choose() {
  pause(); time = 0; selected = undefined;
  run = replay.runs.find(r => r.run_id === $('candidate').value);
  $('evidence').replaceChildren();
  evidence('Study', replay.study_id);
  evidence('候选 / 运行', `${run.candidate_id} / ${run.run_id}`);
  evidence('Study 内容 SHA-256', replay.study_sha256);
  evidence('场景来源', '示意几何；无真实工厂位置或运输轨迹');
  const succeeded = run.status === 'succeeded';
  for (const id of ['play', 'time', 'reset']) $(id).disabled = !succeeded;
  $('selection').textContent = '选择一个作业。';
  $('jobs').replaceChildren();
  $('clock').textContent = '0.000 s';
  $('time').value = 0;
  if (!succeeded) {
    scene.clear();
    $('assessment').textContent = run.status === 'timed_out' ? '执行超时' : '执行未成功';
    $('assessment').className = 'badge rejected';
    for (const id of ['duration', 'queued', 'processing', 'completed']) $(id).textContent = '—';
    $('message').textContent = '该候选没有可回放的成功结果。';
    $('message').hidden = false;
    evidence('执行错误', run.error);
    return;
  }
  $('message').hidden = true;
  $('duration').textContent = `${run.duration.toFixed(3)} s`;
  const accepted = run.assessment.status === 'accepted';
  $('assessment').textContent = accepted ? '模型约束通过' : '模型约束未通过';
  $('assessment').className = `badge ${accepted ? 'accepted' : 'rejected'}`;
  $('time').max = run.duration;
  scene.setJobs(run.jobs);
  evidence('结果来源', sources[run.source] ?? run.source);
  evidence('后端证据', run.provenance);
  evidence('约束判断', run.assessment);
  render();
}

async function verifiedBytes(name, manifest) {
  const response = await fetch(name, { cache: 'no-store' });
  if (!response.ok) throw new Error(`无法读取 ${name}`);
  const bytes = await response.arrayBuffer();
  const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(v => v.toString(16).padStart(2, '0')).join('');
  if (hash !== manifest.files[name]) throw new Error(`文件摘要不匹配：${name}`);
  return bytes;
}

async function verifiedJson(name, manifest) {
  return JSON.parse(new TextDecoder().decode(await verifiedBytes(name, manifest)));
}

async function start() {
  const response = await fetch('bundle.json', { cache: 'no-store' });
  if (!response.ok) throw new Error('无法读取回放清单');
  const manifest = await response.json();
  replay = await verifiedJson('replay.json', manifest);
  await verifiedJson('study.json', manifest);
  if (replay.mapping === 'newton-robot-playback/1') {
    const { startRobot } = await import('./robot-viewer.js');
    await startRobot(replay, manifest, verifiedBytes);
    return;
  }
  if (replay.mapping !== 'fifo-single-server-playback/1' || replay.time_unit !== 's') throw new Error('不支持的回放模型或单位');
  scene = createScene($('viewport'), showSelection);
  $('task-title').textContent = replay.task_id;
  $('description').textContent = replay.description;
  $('sources').replaceChildren();
  const tag = document.createElement('span'); tag.textContent = sources[replay.input_source]; $('sources').append(tag);
  $('decision').textContent = replay.decision.status === 'candidate_selected'
    ? `给定候选中选中：${replay.decision.best_observed_candidate}` : '当前决策：证据不足以选定候选。';
  replay.runs.forEach(item => {
    const option = document.createElement('option'); option.value = item.run_id; option.textContent = item.candidate_id; $('candidate').append(option);
  });
  const initial = replay.runs.find(r => r.candidate_id === replay.decision.best_observed_candidate)
    ?? replay.runs.find(r => r.status === 'succeeded') ?? replay.runs[0];
  $('candidate').value = initial.run_id;
  $('candidate').disabled = false;
  $('candidate').addEventListener('change', choose);
  $('time').addEventListener('input', () => { pause(); time = Number($('time').value); render(); });
  $('reset').addEventListener('click', () => { pause(); time = 0; render(); });
  $('play').addEventListener('click', () => {
    if (playing) { pause(); return; }
    if (time >= run.duration) time = 0;
    playing = true; $('play').textContent = '暂停'; lastFrame = undefined;
  });
  choose();
  requestAnimationFrame(tick);
}

function tick(now) {
  if (playing) {
    if (lastFrame !== undefined) time = Math.min(run.duration, time + (now - lastFrame) / 1000 * Number($('speed').value));
    render();
    if (time >= run.duration) pause();
  }
  lastFrame = now;
  requestAnimationFrame(tick);
}

start().catch(error => {
  scene?.clear(); pause();
  $('message').hidden = false;
  $('message').textContent = `无法展示回放：${error.message}`;
  $('assessment').textContent = '结果不可用';
  for (const id of ['candidate', 'play', 'time', 'reset']) $(id).disabled = true;
  console.error(error);
});
