import { createRobotScene } from './robot-scene.js';

const $ = id => document.getElementById(id);
export async function startRobot(replay, manifest, verifiedBytes) {
  document.body.classList.add('robot-mode');
  const modelName = replay.model.id === 'unitree-g1' ? 'Unitree G1 · 29 自由度' : 'KUKA LBR iiwa 14 · 7 自由度';
  document.querySelector('h1').textContent = modelName;
  document.title = `${modelName} · 仿真回放`;
  document.querySelector('.scene-caption').textContent = '原始机器人网格 · 拖动旋转 · 滚轮缩放 · 点击连杆';
  document.querySelector('.legend').innerHTML = '<label><input id="trace" type="checkbox" checked> 连杆原点轨迹</label><label><input id="loop" type="checkbox" checked> 循环回放</label><button id="camera-reset">重置视角</button><span>姿态来自 Newton 求解；按真实采样帧显示</span>';
  const labels = document.querySelectorAll('.metrics > div > span');
  ['仿真时长', '关节自由度', '当前接触数', '采样时间'].forEach((v, i) => labels[i].textContent = v);
  document.querySelector('aside section:nth-child(2) h2').textContent = '关节状态';
  document.querySelector('.hint').textContent = '角度单位 rad；点击关节或机器人连杆查看位置。';
  document.querySelector('thead').innerHTML = '<tr><th>关节</th><th>当前 / rad</th><th>目标 / rad</th></tr>';
  document.querySelector('footer').textContent = '真实型号公开网格 · 合成控制工况 · Newton 参考仿真；没有真机校准，场景布光与网格地面仅用于展示。';
  $('task-title').textContent = replay.task_id; $('description').textContent = replay.description;
  $('sources').innerHTML = '<span>原始机器人网格</span><span>合成工况</span><span>Newton 仿真</span>';
  $('decision').textContent = replay.decision.status === 'candidate_selected'
    ? `模型约束下选中：${replay.decision.best_observed_candidate}` : '当前决策：证据不足以选定候选。';
  let run, time = 0, playing = false, previous, sample = 0, selected, generation = 0, rows = [];
  const scene = createRobotScene($('viewport'), body => { selected = body; render(); });
  function pause() { playing = false; $('play').textContent = '播放'; }
  function evidence(label, value) {
    const term = document.createElement('dt'), detail = document.createElement('dd');
    term.textContent = label; detail.textContent = typeof value === 'string' ? value : JSON.stringify(value);
    $('evidence').append(term, detail);
  }
  function render() {
    if (!run || run.status !== 'succeeded' || !rows.length) return;
    let lo = 0, hi = run.times.length - 1;
    while (lo < hi) { const mid = Math.ceil((lo + hi) / 2); if (run.times[mid] <= time + 1e-10) lo = mid; else hi = mid - 1; }
    sample = lo; scene.update(run.body_poses[sample], selected);
    $('time').value = time; $('clock').textContent = `${time.toFixed(3)} s`;
    $('processing').textContent = run.contact_counts[sample] ?? '—';
    $('completed').textContent = `${run.times[sample].toFixed(3)} s`;
    rows.forEach((row, i) => { row.children[1].textContent = run.joint_values[sample][i].toFixed(4); row.classList.toggle('selected', selected === run.scene.joint_bodies[i]); });
    if (selected !== undefined) {
      const p = run.body_poses[sample][selected];
      $('selection').textContent = `${run.scene.bodies[selected]} · 世界坐标 / m：${p.slice(0, 3).map(v => v.toFixed(3)).join(', ')}`;
    }
  }
  async function choose() {
    const ticket = ++generation;
    pause(); time = 0; selected = undefined; rows = []; scene.clear();
    run = replay.runs.find(r => r.run_id === $('candidate').value);
    $('jobs').replaceChildren(); $('evidence').replaceChildren();
    $('selection').textContent = '选择一个关节或连杆。'; $('clock').textContent = '0.000 s'; $('time').value = 0;
    for (const id of ['play', 'time', 'reset']) $(id).disabled = true;
    evidence('Study / 运行', `${replay.study_id} / ${run.run_id}`);
    evidence('Study 内容摘要', replay.study_sha256);
    if (run.status !== 'succeeded') {
      $('message').hidden = false; $('message').textContent = '该候选没有可回放的成功结果。';
      $('assessment').textContent = '执行未成功'; $('assessment').className = 'badge rejected';
      for (const id of ['duration', 'queued', 'processing', 'completed']) $(id).textContent = '—';
      evidence('错误', run.error); return;
    }
    $('candidate').disabled = true;
    $('message').hidden = false; $('message').textContent = '正在载入机器人网格…';
    try {
      await scene.load(run, name => verifiedBytes(name, manifest));
      if (ticket !== generation) return;
      scene.setTrace($('trace').checked);
      $('duration').textContent = `${run.duration.toFixed(3)} s`; $('queued').textContent = run.target_q.length;
      const accepted = run.assessment.status === 'accepted';
      $('assessment').textContent = accepted ? '模型约束通过' : '模型约束未通过'; $('assessment').className = `badge ${accepted ? 'accepted' : 'rejected'}`;
      $('time').max = run.duration;
      run.scene.joint_bodies.forEach((body, i) => {
        const row = document.createElement('tr'); row.dataset.joint = run.provenance.joint_order[i]; row.tabIndex = 0;
        [run.provenance.joint_order[i], '', run.target_q[i].toFixed(4)].forEach(text => { const cell = document.createElement('td'); cell.textContent = text; row.append(cell); });
        const select = () => { selected = body; render(); };
        row.addEventListener('click', select); row.addEventListener('keydown', e => { if (e.key === 'Enter') select(); });
        rows.push(row); $('jobs').append(row);
      });
      evidence('求解器', `${run.provenance.engine} · ${run.provenance.numerical_backend}`);
      evidence('网格来源提交', run.provenance.asset_commit);
      evidence('坐标与姿态', '米 / 右手 Z-up / xyzw 四元数；Newton 连杆世界姿态');
      evidence('约束判断', run.assessment); evidence('导入检查', run.provenance.import_audit);
      const link = document.createElement('a'); link.href = run.license; link.textContent = '机器人原始资产许可'; $('evidence').append(link);
      $('message').hidden = true; for (const id of ['play', 'time', 'reset']) $(id).disabled = false;
      render();
    } catch (error) {
      scene.clear(); rows = []; $('jobs').replaceChildren(); $('message').hidden = false;
      $('message').textContent = `无法展示机器人：${error.message}`; $('assessment').textContent = '结果不可用';
      for (const id of ['duration', 'queued', 'processing', 'completed']) $(id).textContent = '—';
    } finally { $('candidate').disabled = false; }
  }
  replay.runs.forEach(item => { const option = document.createElement('option'); option.value = item.run_id; option.textContent = item.candidate_id; $('candidate').append(option); });
  const initial = replay.runs.find(r => r.candidate_id === replay.decision.best_observed_candidate) ?? replay.runs.find(r => r.status === 'succeeded') ?? replay.runs[0];
  $('candidate').value = initial.run_id;
  $('candidate').addEventListener('change', choose);
  $('time').addEventListener('input', () => { pause(); time = Number($('time').value); render(); });
  $('reset').addEventListener('click', () => { pause(); time = 0; render(); });
  $('play').addEventListener('click', () => { if (playing) return pause(); if (time >= run.duration) time = 0; playing = true; previous = undefined; $('play').textContent = '暂停'; });
  $('trace').addEventListener('change', () => scene.setTrace($('trace').checked));
  $('camera-reset').addEventListener('click', scene.resetCamera);
  await choose();
  function tick(now) {
    if (playing && previous !== undefined) {
      time = Math.min(run.duration, time + (now - previous) / 1000 * Number($('speed').value)); render();
      if (time >= run.duration) { if ($('loop').checked) time = 0; else pause(); }
    }
    previous = now; requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
