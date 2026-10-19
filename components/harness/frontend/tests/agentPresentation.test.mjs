import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
import {automatedRequest, operationLabel, boundaryFeedback} from '../src/services/agentPresentation.mjs'

test('Finish summaries and approval questions are visible without opening tools', () => {
  const history = [{type:'user', content:'Investigate'}, {type:'tool_call', tool_name:'finish'}]
  assert.equal(boundaryFeedback({type:'complete', result:'Inspection proposed.'}, history), 'Inspection proposed.')
  assert.equal(boundaryFeedback({type:'awaiting_input', content:'Approve inspection?'}, history), 'Approve inspection?')
  assert.equal(boundaryFeedback({type:'complete', result:''}, history), '')
  assert.equal(boundaryFeedback({type:'tool_call', result:'internal'}, history), '')
})

test('Boundary feedback deduplicates exact prose but preserves new conclusions', () => {
  const history = [{type:'assistant',content:'Done'}, {type:'tool_call',tool_name:'finish'}]
  assert.equal(boundaryFeedback({type:'complete',result:'Done'},history), '')
  assert.equal(boundaryFeedback({type:'complete',result:'Done. Schedule blocked.'},history), 'Done. Schedule blocked.')
  assert.equal(boundaryFeedback({type:'complete',result:'Done'}, [...history,{type:'user',content:'Again'}]), 'Done')
  assert.equal(boundaryFeedback({type:'complete',result:'Failed'}, [{type:'error',content:'Failed'}]), '')
})

test('Technical start outcomes are displayed without claiming physical success', () => {
  assert.equal(operationLabel('BLOCKED_BEFORE_WRITE'), 'Not started')
  assert.equal(operationLabel('NO_WRITES_SENT'), 'Not started')
  assert.equal(operationLabel('needs_review'), 'Review required')
  assert.equal(operationLabel('finished'), 'Completed')
})

test('Automated fault summaries keep full evidence out of the default transcript', () => {
  for (const prefix of ['Maintenance investigation: ', 'SIMULATED OPC TEST event ', 'LIVE OPC event ']) {
    const prompt = prefix + '123. Internal instructions. Evidence: {"values":[false,true]}'
    const card = automatedRequest(prompt)
    assert.ok(card.automated)
    assert.equal(card.details, prompt)
    assert.ok(card.content.length < 80)
    assert.ok(!card.content.includes('Evidence'))
    assert.equal(card.content.startsWith('Test fault'), prefix.startsWith('SIMULATED'))
  }
})

test('Presentation does not replace the prompt sent to the Agent', () => {
  const source = readFileSync(new URL('../src/components/ChatWindow.vue', import.meta.url), 'utf8')
  assert.match(source, /\.\.\.automatedRequest\(message\)/)
  assert.match(source, /buildChatMessage\(message, fileContext, wizardContext, shopFloorContext\)/)
  assert.match(source, /<details v-if="message.details">/)
  assert.match(source, /content: userMsg/)
})
