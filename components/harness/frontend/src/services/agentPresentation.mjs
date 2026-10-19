// Presentation only: the complete original request still goes to the Agent.
// Terminal packets carry user-facing content even when finish/ask_user were tools.
export function boundaryFeedback(data, messages) {
  const text = data.type === 'awaiting_input' ? data.content
    : data.type === 'complete' ? data.result : ''
  if (typeof text !== 'string' || !text.trim()) return ''
  // Ignore tool/spinner cards, but do not deduplicate across a new user turn.
  const previous = [...messages].reverse().find(m => ['assistant', 'user', 'error'].includes(m.type))
  return previous && previous.content?.trim() === text.trim() ? '' : text
}

export function operationLabel(value) {
  const labels = {
    BLOCKED_BEFORE_WRITE: 'Not started', NO_WRITES_SENT: 'Not started',
    finished: 'Completed', running: 'In progress', needs_review: 'Review required',
    interrupted: 'Interrupted', 'not sent': 'Not sent',
  }
  return labels[value] || String(value || '—').replaceAll('_', ' ')
}

export function automatedRequest(message) {
  const simulated = message.startsWith('SIMULATED OPC TEST event ')
  const live = message.startsWith('LIVE OPC event ') || message.startsWith('Maintenance investigation:')
  if (simulated || live) return {
    content: simulated ? 'Test fault detected · Analysis requested' : 'Equipment fault detected · Analysis requested',
    details: message,
    automated: true,
  }
  return {content: 'Analysis requested', details: message, automated: true}
}
