<template>
  <section class="recovery-workspace">
    <aside class="rail">
      <div class="rail-label">WORKSPACE</div>
      <button v-for="item in [{id:'monitor',name:'Cell monitor',icon:'◉'},{id:'orders',name:'Production',icon:'▤'},{id:'incident',name:'Incidents',icon:'△'},{id:'knowledge',name:'Knowledge',icon:'▧'},{id:'audit',name:'Activity',icon:'≡'}]" :key="item.id" :class="{active: tab === item.id}" @click="tab = item.id"><span>{{ item.icon }}</span>{{ item.name }}<b v-if="item.id === 'incident' && activeIssue">1</b></button>
      <div class="rail-footer"><span class="dot" :class="{red: connectionStatus !== 'connected'}"></span>Agent {{ connectionStatus }}</div>
    </aside>
    <main class="workspace-body">
      <header class="command-bar">
        <div><div class="breadcrumb">OPERATIONS / FESTO CP LAB</div><h1>Phone housing production <span class="asset-code">{{ assetCode }}</span></h1></div>
        <div class="toolbar"><a v-if="state" :href="state.mes_url" target="_blank" rel="noopener noreferrer">MES ↗</a><a :href="aasUrl" target="_blank" rel="noopener noreferrer">AAS ↗</a><button @click="refresh" :disabled="loading || busy">↻ {{ loading ? 'Reading' : 'Refresh' }}</button></div>
      </header>
      <LiveFaultMonitor v-if="liveAssetId" :key="liveAssetId + assetEndpoint" :asset-id="liveAssetId" :endpoint="assetEndpoint" :agent-locked="agentLocked" @sample="applyLiveMonitor" @source-change="changeMonitoringSource" @bindings="signalBindings = $event" />
      <div v-if="error" class="alert error" role="alert">{{ error }}<button @click="error=''">×</button></div>
      <div v-if="stale" class="alert warning">MES / AAS snapshot stale · last successful read {{ clock(state?.observed_at) }}</div>
      <div v-if="!currentAsset && tab !== 'orders'" class="empty"><h2>{{ loading ? 'Connecting to operations…' : 'Operations unavailable' }}</h2><p>Check AAS, MES and demo provisioning.</p><button @click="refresh">Retry connection</button></div>
      <ProductionWorkspace v-if="!currentAsset && tab === 'orders'" view="orders" />
      <template v-if="currentAsset">
        <div v-if="state" class="context-strip"><span><i class="dot" :class="{red: stale}"></i>AAS + MES {{ stale ? 'STALE' : 'CONNECTED' }}</span><span class="right">Read {{ clock(state.observed_at) }}</span></div>
        <div v-if="tab === 'monitor'" class="view">
          <div class="view-toolbar"><span class="badge" :class="source==='live'?'green':'amber'">{{ source === 'live' ? 'Live OPC' : 'Simulation · OPC' }}</span><span class="muted">Last read {{ liveOpcAt || '—' }}</span><span class="right badge" :class="cellStatus==='FAULT'?'red':'blue'">{{ cellStatus }}</span></div>
          <div v-if="liveOpcError" class="alert error">{{ liveOpcError }}</div>
          <article v-if="source==='live'" class="panel inspector">
            <div class="panel-title">Machine control<HelpTip text="Starts Press and Turning after onsite confirmation. This operates physical equipment." /></div>
            <div class="toolbar"><button class="danger-outline" @click="previewStart" :disabled="startBusy || startRun?.status==='running' || startRun?.status==='needs_review' || startRun?.status==='interrupted'">Start both stations…</button><button @click="loadStart">Refresh run</button><span v-if="startRun && ['running','needs_review','interrupted'].includes(startRun.status)" class="badge amber">{{ operationLabel(startRun.status) }}</span></div>
            <p v-if="startError" class="alert error">{{ startError }}</p>
            <details v-if="startRun" class="start-history"><summary>Last start attempt · {{ operationLabel(startRun.pair_outcome==='NO_WRITES_SENT' ? startRun.pair_outcome : startRun.status) }} <HelpTip text="Historical result, not current machine status. Expand to review the command outcome and recorded signals." /></summary>

              <p v-for="reason in startRun.blockers" :key="reason" class="alert warning">{{ reason }} · No Start sent</p>
              <p v-if="startRun.error" class="alert error">{{ startRun.error }}</p>

              <table><thead><tr><th>Station</th><th>Command</th><th>Observed result</th></tr></thead><tbody><tr v-for="station in ['Press','Turning']" :key="station"><td>{{ station }}</td><td>{{ operationLabel(startRun.commands[station] || 'not sent') }}</td><td>{{ operationLabel(startRun.results[station] || 'Pending observation') }}</td></tr></tbody></table>
              <div class="toolbar"><button @click="askStartAgent" :disabled="connectionStatus!=='connected' || startRun.status==='running'">Analyze attempt</button></div>
              <details><summary>Recorded signals</summary><div class="scroll-table"><table><thead><tr><th>t / s</th><th>Station</th><th>Event</th><th>Evidence</th></tr></thead><tbody><tr v-for="(event,index) in startRun.events" :key="index"><td>{{ event.elapsed }}</td><td>{{ event.station || 'Both' }}</td><td>{{ event.kind }}</td><td class="mono">{{ JSON.stringify(event.values || event.reasons || event.note || event.error || '') }}</td></tr></tbody></table></div></details>
              <template v-if="['needs_review','interrupted'].includes(startRun.status)"><label class="approval-check"><input type="checkbox" v-model="startReviewed"/> I inspected both stations locally and reviewed uncertain command outcomes.</label><button @click="reviewStart" :disabled="!startReviewed || startBusy">Acknowledge review · no PLC command</button></template>
            </details>
            <details><summary>Safety</summary><p>Existing faults block dispatch. No automatic reset, retry or stop. Ending observation does not stop equipment; use the local HMI and safety controls.</p></details>
          </article>
          <div class="kpi-grid">
            <article><small>{{ source==='simulation'?'AIR PRESSURE · SIM':'PRESSURE 1 · RAW OPC' }}</small><strong>{{ display(liveValue(source==='simulation'?'Press.PressureBar':'Press.Pressure1')) }}<em>{{ source==='simulation'?'bar':'counts' }}</em></strong></article>
            <article><small>PRESS READY</small><strong class="compact">{{ pressState }}</strong></article>
            <article><small>TURNING-OVER READY</small><strong class="compact">{{ turnState }}</strong></article>
            <article><small>PRESS FAULT · OPC</small><strong>{{ display(liveValue('Press.Fault')) }}</strong></article>
          </div>
          <article v-if="source==='live'" class="panel">
            <div class="panel-title">PLC / HMI CROSS-CHECK <span>READ ONLY · {{ liveOpcAt || 'NOT READ' }}</span></div>
            <table><thead><tr><th>PLC signal</th><th>Press · 4042</th><th>Turning · 4041</th></tr></thead><tbody>
              <tr v-for="signal in hmiSignals" :key="signal.id"><td>{{ signal.label }}<HelpTip :text="signal.tag" /></td><td class="mono">{{ display(liveValue('Press.'+signal.id)) }}</td><td class="mono">{{ display(liveValue('Turning.'+signal.id)) }}</td></tr>
            </tbody></table>
            <div class="cell-caption">Fault bit meanings and pressure scaling require the HMI / PLC mapping. Ready is not permission to start.</div>
            <button @click="tab='incident'">Inspect live evidence →</button>
          </article>
          <div class="monitor-grid">
            <article class="panel"><div class="panel-title">CELL OVERVIEW <span>{{ source==='simulation'?'SIMULATED STATE':'OPC SNAPSHOT' }}</span></div>
              <div class="cell-flow"><div class="station" :class="{fault: cellStatus==='FAULT'}"><svg viewBox="0 0 120 90"><path d="M20 80V12H100V80M10 80H110M38 12V34H82V12M55 34V57H65V34M32 70H88" /><rect x="40" y="60" width="40" height="8"/></svg><b>Stamp / press</b><span>{{ pressState }}</span></div><div class="flow-arrow">→</div><div class="station"><svg viewBox="0 0 120 90"><path d="M25 76H95M60 76V56M32 28A30 30 0 1 1 30 60M25 18L32 28L44 22"/><rect x="46" y="34" width="28" height="18"/></svg><b>Turn over 180°</b><span>{{ turnState }}</span></div></div>
              <div class="cell-caption"><span>CP LAB / INSTANCE {{ assetCode }}</span><span>{{ source==='simulation'?'No physical actuation':'Readiness and station fault are independent' }}</span></div>
            </article>
            <article class="panel"><div class="panel-title">{{ source==='simulation'?'PRESSURE / LAST 60 s':'OPC ACQUISITION' }}<span>{{ source==='simulation'?'OPC READ-BACK · bar':'OPC READ-BACK' }}</span></div>
              <template v-if="source==='simulation'"><svg v-if="pressureHistory.length" class="trend" viewBox="0 0 600 130" role="img" aria-label="Simulated OPC pressure read-back trend"><polyline :points="pressurePoints" fill="none" stroke="#38bdf8" stroke-width="2"/></svg><p v-else class="empty">Start monitoring above to read the simulated nodes.</p></template>
              <div v-else class="acquisition"><strong>{{ liveOpc?.values?.length ?? '—' }}</strong><span>signals read</span><p>{{ liveOpc?.failed?.length || 0 }} endpoint errors</p><HelpTip text="Pressure is shown as raw sensor counts; bar calibration is not configured." /></div>
            </article>
          </div>
          <article class="panel"><div class="panel-title">OPC SIGNALS <input v-model="signalFilter" placeholder="Filter signal or OPC tag" aria-label="Filter signals"/></div><div v-for="failure in liveOpc?.failed" :key="failure.endpoint" class="alert error">{{ failure.endpoint }} · {{ failure.error }}</div><div class="scroll-table"><table><thead><tr><th>AAS path</th><th>OPC node / tag</th><th>Value</th><th>Read status</th></tr></thead><tbody><tr v-for="value in liveSignals" :key="value.path"><td>{{ value.path }}</td><td class="mono muted">{{ value.node || value.tag }}</td><td class="mono">{{ display(value.value) }}</td><td>{{ value.status || (value.value === undefined ? 'Not read' : 'Observed') }}</td></tr></tbody></table><p v-if="!liveSignals.length" class="empty">No OPC readings. Start monitoring above.</p></div></article>
          <article class="panel"><div class="panel-title">PRODUCTION QUEUE <button @click="tab='orders'">View orders →</button></div><table><thead><tr><th>Order</th><th>Status</th><th>Resource</th><th>Qty</th><th>Scheduled start</th></tr></thead><tbody><tr v-for="(item,index) in orders" :key="item.order.id" class="clickable" @click="selectedOrder=index===0?'order_a':'order_b'; tab='orders'"><td>{{ item.order.order_no }}</td><td><span class="badge" :class="item.order.status==='BLOCKED'?'red':'blue'">{{ item.order.status }}</span></td><td>{{ lineName(item.order.line_id) }}</td><td>{{ item.order.planned_qty }}</td><td>{{ date(item.order.planned_start_at) }}</td></tr></tbody></table></article>

        </div>
        <ProductionWorkspace v-else-if="['orders','incident','audit'].includes(tab)" :view="tab" />
        <div v-else-if="tab==='knowledge'" class="view">
          <div class="view-heading"><h2>Knowledge library</h2></div>
          <div class="document-grid"><a v-for="doc in documents" :key="doc.id" :href="pdfUrl(doc.id)" target="_blank" rel="noopener noreferrer" class="document-card"><span class="pdf-icon">PDF</span><strong>{{ doc.filename }}</strong><small>{{ doc.pages }} pages</small></a></div>
          <form class="search-bar" @submit.prevent="retrieveKnowledge"><input v-model="knowledgeQuery" placeholder="Search maintenance and technical manuals…" aria-label="Search PDF knowledge"/><button class="primary" :disabled="searchBusy">{{ searchBusy?'Retrieving…':'Search' }}</button></form>
          <div class="result-heading">{{ chunks.length }} retrieved passages <button class="right" @click="askAgent" :disabled="connectionStatus!=='connected'">Analyze case with agent →</button></div>
          <article class="passage" v-for="chunk in chunks" :key="chunk.id"><div><span class="badge" :class="['DEMO_FICTION','ENGINEERING_SCENARIO'].includes(chunk.origin)?'amber':'green'">{{ chunk.origin }}</span><a class="right" :href="pdfUrl(chunk.document_id,chunk.page)" target="_blank" rel="noopener noreferrer">{{ chunk.filename }} · p. {{ chunk.page }} ↗</a></div><h3>{{ chunk.title }}</h3><p>{{ chunk.text }}</p><HelpTip :text="`Reference: ${chunk.id}; SHA256: ${chunk.sha256}; relevance: ${chunk.score}`" /></article>
        </div>
      </template>
      <footer class="workspace-footer"><span>FESTO / {{ assetCode }}</span><span>{{ source==='simulation'?'SIMULATION':'LIVE OPC' }}</span><span>Physical Start requires separate confirmation</span><span class="right">A4S OPERATIONS</span></footer>
    </main>
    <div v-if="plan" class="approval-overlay" @click.self="!busy && (plan=null)">
      <section class="approval-dialog" role="dialog" aria-modal="true" aria-labelledby="approval-title"><div class="panel-title">CHANGE REQUEST <button @click="plan=null; approved=false" :disabled="busy">×</button></div><h2 id="approval-title">{{ actions.find(a=>a.id===plan.action)?.title }}</h2><span class="badge amber">MES BUSINESS WRITE</span><ul><li v-for="effect in plan.preview.action_contract.effects" :key="effect">{{ effect }}</li></ul><details><summary>Command</summary><pre>{{ plan.command }}</pre></details><details><summary>Request & preflight evidence</summary><pre>{{ JSON.stringify(plan.preview,null,2) }}</pre></details><label class="approval-check"><input type="checkbox" v-model="approved"/> Approve this single MES change. No equipment operation.</label><div class="toolbar"><button @click="plan=null; approved=false">Cancel</button><button class="primary" @click="commit" :disabled="busy || !approved">{{ busy?'Executing…':'Approve & execute' }}</button></div><small class="muted">Expires after 5 minutes · Changed state requires a fresh plan</small></section>
    </div>
    <div v-if="startPlan" class="approval-overlay"><section class="approval-dialog" role="dialog" aria-modal="true" aria-labelledby="start-title"><h2 id="start-title">Start both physical stations?</h2><p class="alert warning">{{ startPlan.warning }}</p><pre>{{ JSON.stringify(startPlan.targets,null,2) }}\n{{ startPlan.node }} = true</pre><p>Two recorded baseline reads must pass before dispatch. Requests are concurrent, not atomic. One station can start even if the other request fails. No MES order is started automatically.</p><label class="approval-check"><input type="checkbox" v-model="startSafe"/> I am the authorized onsite operator. Both stations are clear, local operating conditions are checked, and local stop / emergency controls are available.</label><label class="approval-check"><input type="checkbox" v-model="startApproved"/> Approve this one physical Start attempt for Press and Turning.</label><div class="toolbar"><button @click="startPlan=null" :disabled="startBusy">Cancel</button><button class="danger-outline" @click="executeStart" :disabled="startBusy || !startSafe || !startApproved">Confirm physical Start</button></div></section></div>
  </section>
</template>

<script setup lang="ts">
import { operationLabel } from '../services/agentPresentation.mjs'
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useSharedWebSocket } from '../services/websocketService'
import { api } from '../services/api'
import { useWizardStore } from '../stores/wizardStore'
import { useEnvStore } from '../stores/environmentStore'
import LiveFaultMonitor from './LiveFaultMonitor.vue'
import ProductionWorkspace from './ProductionWorkspace.vue'
import {projectReport, appendPressure, signalRows} from '../services/opcMonitoring.mjs'

defineProps<{agentLocked?:boolean}>()
const currentAsset = ref<any>(null)
const assetEndpoint = ref('')
const liveAssetId = computed(() => currentAsset.value?.id || '')
const assetCode = computed(() => currentAsset.value?.idShort?.replace(/^FestoWorkCenterStamping_/, '') || 'Unavailable')
let discoveryLoading = false
async function discoverAsset() {
  if (discoveryLoading || disposed) return
  discoveryLoading = true
  try {
    const {data} = await api.get<any>('/api/demo/festo/asset')
    if (disposed) return
    const changed = liveAssetId.value !== data.asset.id || assetEndpoint.value !== data.endpoint
    if (changed) clearAssetState()
    currentAsset.value = {...data.asset, operational_id: data.operational_id}
    assetEndpoint.value = data.endpoint
    if (changed) void refresh()
  } catch (e:any) {
    clearAssetState()
    currentAsset.value = null
    error.value = e.message || String(e)
  } finally { discoveryLoading = false }
}
function clearAssetState() {
  state.value = null; liveOpc.value = null; liveOpcAt.value = ''
  backendMonitoring.value = false; source.value = 'live'; pressureHistory.value = []; signalBindings.value = []
  plan.value = null; startPlan.value = null; startRun.value = null
  approved.value = false; startApproved.value = false; startSafe.value = false
  stale.value = true
}

const { connectionStatus } = useSharedWebSocket()
const tab = ref('monitor')
const source = ref('live')
const startPlan = ref<any>(null)
const startRun = ref<any>(null)
const startBusy = ref(false)
const startSafe = ref(false)
const startApproved = ref(false)
const startReviewed = ref(false)
const startError = ref('')
const selectedOrder = ref('order_a')
const documents = ref<any[]>([])
const searchBusy = ref(false)
const stale = ref(false)
const signalFilter = ref('')
const store = useWizardStore()
const env = useEnvStore()
const state = ref<any>(null)
const plan = ref<any>(null)
const lastResult = ref<any>(null)
const audit = ref<any[]>([])
const loading = ref(false)
const busy = ref(false)
const approved = ref(false)
const scenario = ref(false)
const error = ref('')
const liveOpc = ref<any>(null)
const backendMonitoring = ref(false)
const pressureHistory = ref<any[]>([])
const signalBindings = ref<any[]>([])
function changeMonitoringSource(selected:string) {
  source.value = selected === 'simulator' ? 'simulation' : 'live'
  backendMonitoring.value = false; liveOpc.value = null; liveOpcAt.value = ''; liveOpcError.value = ''
  pressureHistory.value = []; signalBindings.value = []
}
function applyLiveMonitor(report:any) {
  const projection = projectReport(report, source.value === 'simulation' ? 'simulator' : 'live')
  if (!projection) return
  backendMonitoring.value = projection.running
  liveOpc.value = projection.sample
  liveOpcError.value = projection.error
  liveOpcAt.value = projection.sample ? new Date(projection.sample.observed_at * 1000).toLocaleTimeString() : ''
  if (source.value === 'simulation' && projection.sample?.success) {
    pressureHistory.value = appendPressure(pressureHistory.value, projection.sample)
    scenario.value = !!liveValue('Press.Fault') || !!liveValue('Turning.Fault')
  }
}
const liveOpcAt = ref('')
const liveOpcError = ref('')
const knowledgeQuery = ref('pressure leakage maintenance OPC turning')
const chunks = ref<any[]>([])
let timer: ReturnType<typeof setTimeout> | undefined
let disposed = false
const aasUrl = computed(() => {
  if (!currentAsset.value) return env.aasUiUrl || 'http://localhost:3000'
  const id = btoa(liveAssetId.value).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  return `${env.aasUiUrl || 'http://localhost:3000'}/?aas=${encodeURIComponent(`${assetEndpoint.value}/shells/${id}`)}`
})
const pressurePoints = computed(() => pressureHistory.value.map((sample, index, rows) => `${index * 600 / Math.max(1, rows.length - 1)},${16 + (6 - sample.value) * 40}`).join(' '))
const orders = computed(() => state.value ? [{ label: 'A / AFFECTED ORDER', order: state.value.order_a }, { label: 'B / UPCOMING ORDER', order: state.value.order_b }] : [])
const activeIssue = computed(() => state.value?.issues.find((issue: any) => issue.status !== 'CLOSED'))
const moved = computed(() => state.value?.order_b.line_id === Number(state.value?.backup.id))
const text = (element: any) => Array.isArray(element?.value) ? element.value.find((v: any) => v.language === 'en')?.text || '' : element?.value || ''
const maintenanceSteps = computed(() => {
  const review = state.value?.maintenance.submodelElements.find((el: any) => el.idShort === 'ConditionTriggeredReview')
  const steps = review?.value.find((el: any) => el.idShort === 'ListMaintenanceSteps')?.value || []
  return steps.map((step: any) => ({name: text(step.value.find((el: any) => el.idShort === 'MaintenanceStepName')), instruction: text(step.value.find((el: any) => el.idShort === 'InstructionMaintenanceStep'))}))
})
const lineName = (id: number) => id === state.value.line.id ? state.value.line.name : id === state.value.backup.id ? state.value.backup.name : String(id)

async function refresh() {
  if (loading.value) return
  loading.value = true
  error.value = ''
  try {
    await discoverAsset()
    if (!liveAssetId.value) return
    const requestedAsset = liveAssetId.value
    const result = (await api.get<any>('/api/demo/festo', { timeout: 60000 })).data
    if (disposed || requestedAsset !== liveAssetId.value || result.asset.id !== liveAssetId.value) return
    state.value = result
    stale.value = false
    if (state.value.issues.some((issue: any) => issue.status !== 'CLOSED')) scenario.value = true
    audit.value = (await api.get<any>('/api/demo/festo/audit')).data.events
  } catch (e: any) { stale.value = true; error.value = e.message || String(e) }
  finally { loading.value = false }
}
async function pollMonitor() {
  if (disposed) return
  await discoverAsset()
  if (source.value === 'live') void loadStart()
  if (!disposed) timer = setTimeout(pollMonitor, 3000)
}
async function retrieveKnowledge() {
  searchBusy.value = true
  try { chunks.value = (await api.get<any>('/api/demo/festo/knowledge', {params: {query: knowledgeQuery.value}})).data.chunks }
  catch (e: any) { error.value = e.message || String(e) }
  finally { searchBusy.value = false }
}
async function loadDocuments() {
  try { documents.value = (await api.get<any>('/api/demo/festo/knowledge/documents')).data.documents }
  catch (e: any) { error.value = e.message || String(e) }
}
function pdfUrl(id: string, page = 1) {
  return env.backendUrl.replace(/\/$/, '') + '/api/demo/festo/knowledge/documents/' + encodeURIComponent(id) + '#page=' + page
}
let startLoading = false
async function loadStart() {
  if (startLoading || disposed) return
  startLoading = true
  try {
    startRun.value = (await api.get<any>('/api/festo/start/latest')).data.run
    if (startError.value.startsWith('Start evidence unavailable.')) startError.value = ''
  }
  catch (e: any) { startError.value = 'Start evidence unavailable. Do not repeat a submitted Start. ' + (e.message || String(e)) }
  finally { startLoading = false }
}
async function previewStart() {
  startBusy.value = true; startError.value = ''; startSafe.value = false; startApproved.value = false
  try { startPlan.value = (await api.post<any>('/api/festo/start/preview', {}, {timeout:60000})).data }
  catch (e: any) { startError.value = e.message || String(e); await loadStart() }
  finally { startBusy.value = false }
}
async function executeStart() {
  if (!startPlan.value || !startSafe.value || !startApproved.value) return
  const id = startPlan.value.id
  startPlan.value = null; startBusy.value = true; startError.value = ''
  try { await api.post('/api/festo/start/execute', {id, approved:true, onsite_safe:true}); await loadStart() }
  catch (e: any) { startError.value = (e.message || String(e)) + ' Outcome may be unknown: inspect the latest run. Do not replay.'; await loadStart() }
  finally { startBusy.value = false }
}
async function reviewStart() {
  startBusy.value = true
  try { await api.post('/api/festo/start/review', {id:startRun.value.id, onsite_reviewed:startReviewed.value}); startReviewed.value=false; startError.value=''; await loadStart() }
  catch (e: any) { startError.value = e.message || String(e) }
  finally { startBusy.value = false }
}
function askStartAgent() {
  store.updateData('pendingAgentQuestion', `Analyze this recorded REAL two-station Start attempt for ${liveAssetId.value}: ${JSON.stringify(startRun.value)}. Distinguish OPC write acknowledgement from motion output evidence and physical movement. Explain baseline blockers, gaps, partial outcomes and fault transitions. Do not infer pneumatic leakage from xSF01. Retrieve cited PDF evidence if useful, distinguish fiction. Read only: do not issue Start, reset, acknowledge, stop, or MES writes.`)
}
async function preview(action: string) {
  busy.value = true; error.value = ''; plan.value = null; approved.value = false
  try { plan.value = (await api.post<any>('/api/demo/festo/preview', { action, scenario: 'simulated-maintenance' }, { timeout: 60000 })).data }
  catch (e: any) { error.value = e.message || String(e) }
  finally { busy.value = false }
}
async function commit() {
  if (!approved.value || !plan.value) return
  busy.value = true; error.value = ''
  const id = plan.value.id
  plan.value = null; approved.value = false
  try {
    lastResult.value = (await api.post<any>('/api/demo/festo/execute', { plan_id: id, approved: true }, { timeout: 120000 })).data
    await refresh()
  } catch (e: any) { error.value = (e.message || String(e)) + ' Do not replay: refresh authoritative MES state before preparing another action.' }
  finally { busy.value = false }
}
function askAgent(alarm: string | Event = '') {
  const observation = typeof alarm === 'string' ? alarm : ''
  store.updateData('pendingAgentQuestion', [
    `Assess the phone-housing stamping and turning-over maintenance case for the INSTANCE ${liveAssetId.value}.`,
    observation,
    'Read its MaintenanceInstructions and ProductionContext submodels and use mes to read the linked orders and issues.',
    "Retrieve maintenance evidence with python -m a4s.machines.festo_cp_lab.knowledge --query 'pressure leakage maintenance OPC turning'.",
    'Cite PDF filename, page, chunk ID and source hash; distinguish REFERENCE from DEMO_FICTION.',
    source.value === 'live' ? `LIVE OPC investigation, not a simulated leak. Last read: ${liveOpcAt.value || 'none'}. Read error: ${liveOpcError.value || 'none'}. Evidence: ${JSON.stringify(liveOpc.value)}. Re-read OPC before conclusions. Do not infer leakage or a failed start from static fault bits. Do not apply the simulated MES workflow to this fault.` : `SIMULATED OPC read-back from the loopback server. Last read: ${liveOpcAt.value || 'none'}. Evidence: ${JSON.stringify(liveOpc.value)}. No physical fault is asserted.`,
    'Explain candidate actions and missing evidence. No PLC writes, no reset/start, no invented measured RUL or physical capacity.',
    'Read existing issues to avoid duplicates. Use mes dry-run for any proposal and ask through the approval gate before a business write. Do not enumerate blueprints.',
  ].filter(Boolean).join('\n'))
}
const currentOrder = computed(() => state.value?.[selectedOrder.value])
const hmiSignals = [
  {id:'Ready', label:'Module ready', tag:'ApplicationModule_DB.xReady'},
  {id:'Fault', label:'Collective fault', tag:'dbVar.Err.xSF'},
  {id:'Fault01', label:'Individual fault 01 · meaning unconfirmed', tag:'dbVar.Err.xSF01'},
  {id:'Error', label:'Module error · not collective fault', tag:'ApplicationModule_DB.xError'},
  {id:'ConveyorForward', label:'Conveyor forward output', tag:'xQA1_A1'},
  {id:'ReturnCode', label:'Module return code · decoding unconfirmed', tag:'ApplicationModule_DB.iRetCode'},
]
const liveSignals = computed(() => signalRows(liveOpc.value, signalBindings.value).filter((v: any) => (v.path + ' ' + v.tag).toLowerCase().includes(signalFilter.value.toLowerCase())))
const liveValue = (path: string) => liveOpc.value?.values.find((v: any) => v.path === path)?.value
const display = (value: any) => value === undefined || value === null ? '—' : typeof value === 'boolean' ? (value ? 'TRUE' : 'FALSE') : value
const clock = (value: any) => value ? new Date(value * 1000).toLocaleTimeString() : '—'
const date = (value: any) => value ? new Date(value).toLocaleString() : 'Unscheduled'
const cellStatus = computed(() => !backendMonitoring.value ? 'STOPPED' : liveOpcError.value ? 'STALE' : !liveOpc.value ? 'CONNECTING' : !liveOpc.value.success ? 'PARTIAL' : liveValue('Press.Fault') || liveValue('Turning.Fault') ? 'FAULT' : 'NORMAL')
const pressState = computed(() => display(liveValue('Press.Ready')))
const turnState = computed(() => display(liveValue('Turning.Ready')))
const stage = computed(() => activeIssue.value ? activeIssue.value.status : cellStatus.value === 'FAULT' ? 'DETECTED' : 'NO INCIDENT')
const actions = computed(() => [
  {id: 'create_issue', title: 'Hold order A', sub: 'Create blocking equipment issue', disabled: !scenario.value || !!activeIssue.value},
  {id: 'move_upcoming', title: 'Reassign order B', sub: 'Check and move to backup line', disabled: !activeIssue.value || moved.value},
  {id: 'acknowledge_issue', title: 'Acknowledge', sub: 'Take ownership of the incident', disabled: activeIssue.value?.status !== 'OPEN'},
  {id: 'resolve_issue', title: 'Resolve issue', sub: 'Record end of simulated incident', disabled: activeIssue.value?.status !== 'ACKNOWLEDGED'},
  {id: 'close_issue', title: 'Close issue', sub: 'Complete the business review', disabled: activeIssue.value?.status !== 'RESOLVED'},
  {id: 'return_upcoming', title: 'Return B to main', sub: 'Restore planned resource after closure', disabled: !!activeIssue.value || !moved.value},
  {id: 'accept_affected', title: 'Accept A for planning', sub: 'Accept the pending order; no machine start', disabled: !!activeIssue.value || state.value?.order_a.status !== 'PENDING'},
])
defineExpose({refresh})
onMounted(() => { void refresh(); loadDocuments(); pollMonitor() })
onUnmounted(() => { disposed = true; if (timer) clearTimeout(timer) })
</script>

<style scoped>
.recovery-workspace{--bg:#0a0c11;--panel:#0f131a;--raised:#151b24;--line:#283240;--muted:#9aa6b7;--fg:#e4eaf2;--cyan:#38bdf8;display:flex;height:calc(100vh - 44px);background:var(--bg);color:var(--fg);font:12px 'Segoe UI',sans-serif;min-width:650px;box-sizing:border-box}
*{box-sizing:border-box}button,input{font:inherit}button,a,input{outline-offset:3px}button{border:1px solid var(--line);background:var(--raised);color:var(--fg);border-radius:3px;padding:7px 10px;cursor:pointer;white-space:nowrap}button:hover:not(:disabled){border-color:var(--cyan);color:white}button:disabled{opacity:.38;cursor:not-allowed}button.primary{background:#13779d;border-color:#209dc7;color:#fff}button.danger-outline{color:#f1a199;border-color:#783e3e}a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}input{background:#171e29;color:var(--fg);border:1px solid var(--line);padding:8px;border-radius:3px;min-width:0}h1{font-size:18px;margin:5px 0 0;font-weight:600}h2{font-size:16px;margin:0}h3{font-size:14px;margin:12px 0}p{line-height:1.6}small{font-size:10px}.mono{font-family:Consolas,monospace}.muted{color:var(--muted)}.right{margin-left:auto}.dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#49d5a0;margin-right:7px}.dot.red{background:#f17676}.rail{width:145px;flex:none;background:var(--panel);border-right:1px solid var(--line);padding:18px 8px;display:flex;flex-direction:column}.rail-label{font-size:9px;letter-spacing:1px;color:var(--muted);padding:0 8px 12px}.rail button{display:flex;align-items:center;gap:8px;border:0;background:none;text-align:left;padding:10px 8px;margin-bottom:4px;font-size:11px}.rail button.active{background:#19232f;color:var(--cyan);box-shadow:inset 2px 0 var(--cyan)}.rail button b{margin-left:auto;color:#ff9c92}.rail .spaced{margin-top:28px}.rail-footer{margin-top:auto;color:var(--muted);font-size:10px;padding:8px}.rail-footer small{display:block;margin-top:8px}.workspace-body{flex:1;min-width:0;display:flex;flex-direction:column;position:relative}.command-bar{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:16px 20px;border-bottom:1px solid var(--line)}.breadcrumb{font:9px Consolas,monospace;color:var(--muted);letter-spacing:1px}.asset-code{color:var(--muted);font:11px Consolas,monospace;margin-left:6px}.toolbar{display:flex;align-items:center;gap:10px}.context-strip{display:flex;align-items:center;flex-wrap:wrap;gap:16px;padding:9px 20px;border-bottom:1px solid var(--line);font-size:10px;color:var(--muted)}.view{overflow-y:auto;flex:1;min-height:0;padding:16px 20px}.view-toolbar,.run-controls,.view-heading{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px}.segments{display:flex}.segments button{border-radius:0}.segments .selected{color:var(--cyan);border-bottom:2px solid var(--cyan)}.badge{display:inline-block;padding:3px 6px;border-radius:2px;font:9px Consolas,monospace;white-space:nowrap;background:#1c2a3e;color:#a7c9ef}.amber{background:#32281a;color:#e9b865}.green{background:#122e29;color:#73d7b3}.red{background:#392022;color:#ff9696}.blue{background:#142c3d;color:#78c8ed}.toggle{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:10px}.toggle input,.approval-check input{accent-color:var(--cyan)}.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--panel);margin-bottom:14px}.kpi-grid article{padding:16px 13px;border-right:1px solid var(--line)}.kpi-grid small{display:block;color:var(--muted);font-size:9px}.kpi-grid strong{display:block;font:28px Consolas,monospace;margin-top:14px}.kpi-grid .compact{font-size:17px;padding-top:8px}.kpi-grid em{font:10px Consolas,monospace;color:var(--muted);margin-left:6px}.monitor-grid{display:grid;grid-template-columns:1.1fr 1fr;gap:14px}.panel{background:var(--panel);border:1px solid var(--line);margin-bottom:14px;min-width:0}.panel-title{display:flex;align-items:center;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding:10px 12px;font-size:10px;letter-spacing:.6px}.panel-title span{font:9px Consolas,monospace;color:var(--muted)}.panel-title button{font-size:10px;padding:4px 7px}.cell-flow{display:flex;align-items:center;justify-content:space-around;padding:18px 10px}.station{text-align:center;width:40%}.station svg{height:75px;width:100%;stroke:#64adc6;stroke-width:3;fill:none}.station b{display:block;font-size:11px;margin:5px 0}.station span{font:9px Consolas,monospace;color:var(--muted)}.station.fault svg{stroke:#f1847d}.flow-arrow{color:var(--cyan);font-size:22px}.cell-caption{display:flex;justify-content:space-between;gap:10px;padding:10px 12px;border-top:1px solid var(--line);color:var(--muted);font:8px Consolas,monospace}.trend{width:100%;height:176px;display:block;padding:15px}.trend text{fill:#c2a776;font:10px Consolas,monospace}.acquisition{text-align:center;padding:20px}.acquisition strong{font:40px Consolas,monospace;color:var(--cyan);margin-right:12px}.incident-banner{display:flex;align-items:center;flex-wrap:wrap;gap:12px;width:100%;padding:13px;background:#261b20;border-color:#65373c;margin-bottom:14px;font-size:11px}.incident-banner b{margin-left:auto;color:#ffb3a8}table{width:100%;border-collapse:collapse;font-size:11px}th,td{text-align:left;padding:11px 12px;border-bottom:1px solid #202937;vertical-align:top;overflow-wrap:anywhere}th{color:var(--muted);font-size:9px;font-weight:500;text-transform:uppercase}.clickable{cursor:pointer}.clickable:hover{background:#1a2430}.scroll-table{max-height:300px;overflow:auto}.scroll-table td{max-width:270px;font-size:10px}.status-note{display:flex;flex-wrap:wrap;gap:8px;font-size:10px;color:var(--muted);margin-top:12px}.workspace-footer{display:flex;gap:16px;flex-wrap:wrap;padding:6px 12px;border-top:1px solid var(--line);font:9px Consolas,monospace;color:var(--muted)}.alert{display:flex;gap:12px;align-items:center;margin:10px 16px;padding:10px;font-size:11px;border:1px solid var(--line)}.alert button{margin-left:auto}.alert.error{background:#351b22;color:#ffb1b1;border-color:#6e3948}.alert.warning{background:#30291c;color:#f0cb89}.alert.success{background:#16332c;color:#94dfbc}.empty{padding:40px;text-align:center;color:var(--muted)}.order-layout{display:grid;grid-template-columns:1fr 1.1fr;gap:18px}.order-card{display:block;text-align:left;width:100%;padding:18px;margin-bottom:12px}.order-card.selected{border-color:var(--cyan);background:#132735}.order-card small{display:block;color:var(--muted)}.order-card strong{display:block;margin:12px 0;font:13px Consolas,monospace;overflow-wrap:anywhere}.inspector{padding:18px}.inspector .panel-title{padding:0 0 12px;margin-bottom:18px}.inspector a{margin-left:12px}dl{display:grid;grid-template-columns:90px 1fr;gap:13px;margin:24px 0}dt{color:var(--muted)}dd{margin:0}.case-summary{border:1px solid var(--line);border-left:3px solid #ca8b4d;padding:18px;background:var(--panel);margin-bottom:18px}.case-summary h2{margin-top:12px}.action-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:18px}.action-cards button{text-align:left;padding:16px;white-space:normal}.action-cards strong,.action-cards span{display:block}.action-cards span{color:var(--muted);font-size:10px;margin-top:10px}.step-row{display:flex;align-items:center;gap:12px;padding:12px;border-bottom:1px solid var(--line)}.step-index{color:var(--cyan);font-family:Consolas,monospace}.document-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.document-card{display:flex;flex-direction:column;gap:12px;border:1px solid var(--line);padding:16px;color:var(--fg);background:var(--panel);overflow-wrap:anywhere}.document-card:hover{border-color:var(--cyan);text-decoration:none}.document-card strong{font-size:12px}.pdf-icon{font:10px Consolas,monospace;color:#f4a6a6;border:1px solid #755057;padding:8px;width:max-content}.search-bar{display:flex;gap:10px;margin:24px 0 16px}.search-bar input{flex:1}.result-heading{display:flex;align-items:center;gap:12px;font-size:11px;margin:14px 0}.passage{padding:18px;background:var(--panel);border:1px solid var(--line);margin-bottom:12px}.passage>div{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.passage a{font-size:10px}.passage p{font-size:12px;color:#c0ccda;line-height:1.7}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:250px;overflow:auto;background:#0b1017;padding:12px;font:10px/1.6 Consolas,monospace}summary{cursor:pointer;color:var(--cyan)}.approval-overlay{position:fixed;inset:0;z-index:2000;background:#000b;display:flex;align-items:center;justify-content:center;padding:20px}.approval-dialog{background:#101720;border:1px solid #406178;box-shadow:0 20px 70px #0008;width:580px;max-height:90vh;overflow:auto;padding:22px}.approval-dialog h2{margin:20px 0 12px}.approval-dialog li{line-height:1.7}.approval-dialog .toolbar{justify-content:flex-end;margin:20px 0 12px}.approval-check{display:flex;gap:10px;margin-top:20px;font-size:12px;align-items:flex-start}
@media(max-width:1250px){.rail{width:110px}.rail button{font-size:10px;padding:10px 4px}.command-bar{padding:12px}.view{padding:12px}.context-strip{gap:8px;padding:8px}.kpi-grid strong{font-size:23px}.kpi-grid .compact{font-size:13px}.monitor-grid{gap:10px}.run-controls button{font-size:10px}}
</style>
<style scoped>
.recovery-workspace{font-size:15px}
.rail{width:165px}.rail button,.rail-footer{font-size:14px}.rail-label{font-size:12px}
h1{font-size:23px}h2{font-size:20px}h3{font-size:17px}
button,input{font-size:14px}.context-strip,.panel-title,.panel-title button,.station b,.alert,.incident-banner,table,.scroll-table td,.result-heading{font-size:14px}
.breadcrumb,.asset-code,.badge,.toggle,.kpi-grid small,.kpi-grid em,.panel-title span,th,.station span,.status-note,small{font-size:12px}
.kpi-grid strong{font-size:32px}.kpi-grid .compact{font-size:21px}
.action-cards span,.document-card strong,.passage p,.approval-check{font-size:15px}
.order-card strong{font-size:16px}pre{font-size:13px}
.cell-caption,.workspace-footer{display:none}
@media(max-width:1250px){.rail{width:140px}.rail button,.run-controls button{font-size:14px}.kpi-grid .compact{font-size:18px}}
</style>
