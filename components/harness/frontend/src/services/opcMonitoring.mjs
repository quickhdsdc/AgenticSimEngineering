// One projection for live OPC and the loopback simulator. Only read evidence
// contributes values; a simulator command is never treated as an observation.
export function projectReport(report, source) {
  const reportedSource = report.source || report.target?.source || 'live'
  if (reportedSource !== source) return null
  const sample = report.state?.latest
  return {
    running: !!report.running,
    sample: sample ? {...sample, failed: sample.failed || [],
      success: !report.stale && report.state.communication === 'GOOD'} : null,
    error: report.running && (report.stale || report.state?.communication !== 'GOOD')
      ? 'OPC evidence unavailable; current state unknown.' : '',
  }
}

export function appendPressure(history, sample) {
  const value = sample?.values?.find(row => row.path === 'Press.PressureBar')?.value
  const at = sample?.observed_at
  if (!Number.isFinite(value) || !Number.isFinite(at)) return history
  if (history.at(-1)?.at >= at) return history
  return [...history.filter(row => row.at >= at - 60), {at, value}]
}

export function signalRows(sample, bindings = []) {
  const observed = sample?.values || []
  const byPath = new Map(observed.map(row => [row.path, row]))
  return bindings.length ? bindings.map(({value, ...binding}) => ({...binding, ...byPath.get(binding.path)})) : observed
}
