// Intervals are right-continuous: at a departure instant the job is completed.
export function stateAt(job, time) {
  if (time < job.arrival) return 'not_arrived';
  if (time < job.start) return 'queued';
  if (time < job.departure) return 'processing';
  return 'completed';
}

export function snapshot(run, time) {
  const states = run.jobs.map(job => ({ ...job, state: stateAt(job, time) }));
  const counts = { queued: 0, processing: 0, completed: 0, not_arrived: 0 };
  states.forEach(job => counts[job.state]++);
  return { states, counts };
}
