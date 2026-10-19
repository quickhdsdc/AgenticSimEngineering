/* Transport for the opsworld console.
   Talks to the real server at 127.0.0.1:8000; if the probe fails it serves the
   identical payload shapes out of sample.js and says so. A 4xx from a live
   server is NOT a fallback — it is an answer, and it surfaces as an error. */

import { sample, SCHEMA, EN, ZH, resolveParams } from './sample.js';

const TIMEOUT = 4000;

async function call(url, options = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT);
  try {
    const res = await fetch(url, { ...options, signal: ctrl.signal, headers: { 'content-type': 'application/json' } });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) { const e = new Error(body.detail || `${res.status} ${res.statusText}`); e.http = res.status; throw e; }
    return body;
  } finally { clearTimeout(timer); }
}

/* Known registry ids receive the selected Chinese/English overlay. Anything
   added by a domain later keeps the server-provided label as a safe fallback. */
export function applyLocale(schema, locale = 'zh') {
  const copy = locale === 'en' ? EN : ZH;
  const over = (list, table) => (list || []).map((x) => (table[x.id] ? { ...x, label: table[x.id][0], description: table[x.id][1] } : x));
  const params = (ps) => (ps || []).map((p) => ({ ...p, label: copy.param[p.name] || p.label, help: copy.help[p.name] || p.help }));
  return {
    ...schema,
    scenarios: over(schema.scenarios, copy.scenario).map((s) => ({ ...s, params: params(s.params) })),
    worlds: over(schema.worlds, copy.world),
    solvers: over(schema.solvers, copy.solver).map((s) => ({ ...s, params: params(s.params) })),
    /* The selected locale wins for capabilities the console knows; anything the server
       adds later still shows up in the server's own words. */
    capabilities: { ...(schema.capabilities || {}), ...copy.capability },
  };
}

export function createClient({ base = 'http://127.0.0.1:8000', onMode = () => {} } = {}) {
  const state = { mode: 'probing', base, lastError: null, latency: {} };
  const jobs = new Map();          // sample-mode job table
  let counter = 0;

  const setMode = (m, why) => { if (state.mode !== m) { state.mode = m; state.lastError = why || null; onMode(m, why); } };

  async function live(path, body, method = 'POST') {
    const t = performance.now();
    const out = await call(state.base + path, method === 'GET' ? { method } : { method, body: JSON.stringify(body || {}) });
    state.latency[path] = Math.round(performance.now() - t);
    return out;
  }

  /* One request. Network-level failure demotes to sample data; an HTTP status
     from a server that answered is propagated, because it means something. */
  async function ask(path, body, fallback, method = 'POST') {
    if (state.mode === 'live') {
      try { return await live(path, body, method); }
      catch (e) { if (e.http) throw e; setMode('sample', `${state.base} stopped answering (${e.message})`); }
    }
    const t = performance.now();
    const out = fallback();
    state.latency[path] = Math.max(1, Math.round(performance.now() - t));
    return out;
  }

  /* -- sample-mode job runner: one timer per job, cancel and timeout real -- */
  function submitSampleJob(body) {
    const id = (++counter).toString(16).padStart(6, '0');
    const timeout = body.timeout || SCHEMA.limits.default_timeout_seconds;
    const job = { id, kind: `solve:${body.world}:${body.solver}`, status: 'queued', elapsed_seconds: 0, queued_seconds: 0, timeout_seconds: timeout, result: null, error: null, _t0: performance.now(), _started: null };
    jobs.set(id, job);
    const spawn = 600 + Math.random() * 400;         // the process-spawn cost, honestly slow
    const work = 1400 + Math.random() * 2600;
    setTimeout(() => { if (job.status !== 'queued') return; job.status = 'running'; job._started = performance.now(); job.queued_seconds = +((job._started - job._t0) / 1000).toFixed(3); }, spawn);
    setTimeout(() => {
      if (job.status !== 'running') return;
      if (job.elapsed_seconds > timeout) { job.status = 'failed'; job.error = `timed out after ${timeout}s`; return; }
      job.status = 'done';
      try { job.result = sample.solve(body); } catch (e) { job.status = 'failed'; job.error = String(e.message || e); }
    }, spawn + work);
    return job;
  }
  const tick = (j) => { if (j.status === 'running') j.elapsed_seconds = +((performance.now() - j._started) / 1000).toFixed(3); if (j.status === 'running' && j.elapsed_seconds > j.timeout_seconds) { j.status = 'failed'; j.error = `timed out after ${j.timeout_seconds}s — the worker process was terminated`; } return j; };
  const strip = (j) => { const { _t0, _started, ...rest } = tick(j); return rest; };

  return {
    state,
    async probe() {
      try { const h = await live('/api/health', null, 'GET'); setMode('live'); return h; }
      catch (e) { setMode('sample', `no server at ${state.base} — ${e.name === 'AbortError' ? 'timed out' : e.message}`); return sample.health(); }
    },
    async schema(locale = 'zh') { return applyLocale(await ask('/api/schema', null, () => sample.schema(), 'GET'), locale); },
    scenario: (b) => ask('/api/scenario', b, () => sample.scenario(b)),
    initial: (b) => ask('/api/initial', b, () => sample.initial(b)),
    evaluate: (b) => ask('/api/evaluate', b, () => sample.evaluate(b)),
    gradient: (b) => ask('/api/gradient', b, () => sample.gradient(b)),
    tau: (b) => ask('/api/relaxation/tau', b, () => sample.tau(b)),
    solve: (b) => ask('/api/solve', b, () => {
      const world = SCHEMA.worlds.find((w) => w.id === b.world);
      const inline = !b.force_job && b.budget > 0 && b.budget <= world.inline_max_budget;
      if (inline) return { mode: 'inline', status: 'done', kind: `solve:${b.world}:${b.solver}`, result: sample.solve(b) };
      const job = submitSampleJob(b);
      return { ...strip(job), mode: 'job', reason: b.force_job ? 'forced by the client' : `budget ${b.budget} exceeds ${b.world}'s inline limit of ${world.inline_max_budget}` };
    }),
    jobs: () => ask('/api/jobs', null, () => ({ jobs: [...jobs.values()].map(strip) }), 'GET'),
    job: (id) => ask(`/api/jobs/${id}`, null, () => strip(jobs.get(id)), 'GET'),
    cancel: (id) => ask(`/api/jobs/${id}`, null, () => { const j = jobs.get(id); if (j && !['done', 'failed'].includes(j.status)) { j.status = 'cancelled'; j.error = 'cancelled by the client'; } return strip(j); }, 'DELETE'),
    resolveParams,
  };
}
