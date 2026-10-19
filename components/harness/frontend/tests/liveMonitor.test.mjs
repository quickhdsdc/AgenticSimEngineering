import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
import {parse, compileScript, compileTemplate} from '@vue/compiler-sfc'

const read = path => readFileSync(new URL('../src/' + path, import.meta.url), 'utf8')

test('Changed components compile with their template bindings', () => {
  for (const filename of ['App.vue', ...['RecoveryWorkspace', 'ChatWindow', 'LiveFaultMonitor', 'ShopFloor', 'InputStage', 'ExtractionStage', 'MatchingStage', 'GenerationStage', 'MachineConsole', 'HelpTip'].map(name => `components/${name}.vue`)]) {
    const {descriptor, errors} = parse(read(filename), {filename})
    assert.deepEqual(errors, [])
    const script = compileScript(descriptor, {id:filename})
    const template = compileTemplate({source:descriptor.template.content, filename, id:filename,
      compilerOptions:{bindingMetadata:script.bindings}})
    assert.deepEqual(template.errors, [])
  }
})

test('Fault handoff is fresh, scoped, idle-only, and reserved before queuing', () => {
  const monitor = read('components/LiveFaultMonitor.vue')
  for (const guard of ["e.kind === 'FAULT_RAISED'", "!e.claimed", "e.episode_id === data.state.episode_id",
    "Date.now()/1000 - e.at < 300", "data.running", "!data.stale", "data.state.communication === 'GOOD'",
    "!props.agentLocked", "connectionStatus.value === 'connected'"]) assert.ok(monitor.includes(guard), guard)
  assert.ok(monitor.indexOf('/api/festo/live-monitor/claim') < monitor.indexOf("store.updateData('pendingAgentQuestion'"))
  assert.match(monitor, /asset_id:props.assetId, endpoint:props.endpoint/)
  assert.match(monitor, /role="status" aria-live="polite"/)
  assert.doesNotMatch(monitor, /\/api\/opc\/write|\/api\/festo\/start/)
  assert.doesNotMatch(monitor, /v-model="automatic"|automatic.value/)
  assert.match(monitor, /store.updateData\('pendingAgentQuestion', claimed.request\)/)
  assert.doesNotMatch(monitor, /no tools are needed|Do not query|SIMULATED OPC TEST/)
  assert.match(monitor, /emit\('sample', \{\.\.\.data, source:requestedSource\}\)/)
})

test('Local host forwards chat locks and keeps physical Start separate', () => {
  const app = read('App.vue'), recovery = read('components/RecoveryWorkspace.vue')
  assert.match(app, /@interaction-state="operationsAgentLocked = \$event"/)
  assert.match(app, /:agent-locked="operationsAgentLocked"/)
  assert.match(recovery, /:agent-locked="agentLocked" @sample="applyLiveMonitor"/)
  assert.doesNotMatch(recovery, /\/api\/opc\/read|\/api\/demo\/festo\/monitor/)
  assert.match(recovery, /Confirm physical Start/)
  assert.match(recovery, /<details v-if="startRun" class="start-history">/)
  assert.match(read('components/ChatWindow.vue'), /isSending.value \|\| awaitingAnswer.value/)
})
