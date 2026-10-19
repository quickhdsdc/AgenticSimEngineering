<template>
  <section class="live-events" :class="{'sim-active': source==='simulator'}">
    <div class="controls"><strong>Monitoring</strong><HelpTip text="Checks OPC signals and requests Agent analysis when a new fault is confirmed. Stop ends monitoring, not equipment operation." />
      <select aria-label="Monitoring source" v-model="source" :disabled="busy || report?.running" @change="changeSource">
        <option value="live">Live OPC</option><option value="simulator">Simulation · OPC</option>
      </select>
      <button @click="control('start')" :disabled="busy || report?.running">Start monitoring</button>
      <button @click="control('stop')" :disabled="busy || !report?.running">Stop</button>
      <span>{{ monitorStatus }}</span>
    </div>
    <p v-if="controlError || error" role="alert">{{ controlError || error }}</p>
    <p v-if="!report?.running && !controlError && !error">Click Start monitoring to connect to the selected OPC source.</p>
    <div v-if="source==='simulator'" class="simulator-panel">
      <div class="controls">
        <strong>Simulated state</strong><HelpTip text="Changes the loopback test server only. Values below come from OPC read-back." />
        <button @click="simControl('normal')" :disabled="busy || !simulator?.running">Normal</button>
        <button @click="simControl('fault')" :disabled="busy || !simulator?.running">Fault / leak</button>
        <button @click="simControl('estop')" :disabled="busy || !simulator?.running">Emergency stop signal</button>
        <span>{{ simulator?.running ? `Server running · ${simulator.mode}` : 'Server starts with monitoring' }}</span>
      </div>
      <p v-if="simulator?.error" role="alert">{{ simulator.error }}</p>
    </div>
    <p v-if="report?.running && !report.stale && report.state.communication === 'GOOD' && report.state.stable === true">{{ source==='simulator' ? 'Test fault' : 'Station fault' }} · cause unverified</p>
    
    
    <div v-if="toast" class="toast" role="status" aria-live="polite">{{ toast }} <button @click="toast=''">Dismiss</button></div>
    <details v-if="report?.events.length"><summary>Events · {{ report.events.length }}</summary>
      <article v-for="event in [...(report?.events || [])].reverse()" :key="event.id">
        <strong>{{ label(event.kind) }}</strong> · {{ new Date(event.at * 1000).toLocaleString() }}
        <small>{{ event.claimed ? 'Analysis requested' : event.kind === 'FAULT_RAISED' ? 'Not handed off' : 'Recorded' }}</small>
        <details><summary>Details</summary><pre>{{ JSON.stringify(event, null, 2) }}</pre></details>
      </article>
    </details>
  </section>
</template>
<script setup lang="ts">
import {computed, onMounted, onUnmounted, ref} from 'vue'
import {api} from '../services/api'
import {useWizardStore} from '../stores/wizardStore'
import {useSharedWebSocket} from '../services/websocketService'
const props = defineProps<{assetId:string; endpoint:string; agentLocked?:boolean}>()
const emit = defineEmits<{'sample':[sample:any]; 'sourceChange':[source:string]; 'bindings':[nodes:any[]]}>()
const store = useWizardStore()
const {connectionStatus} = useSharedWebSocket()
const report = ref<any>(null), simulator = ref<any>(null), source = ref('live'), busy = ref(false), error = ref(''), controlError = ref(''), toast = ref('')
let disposed = false, timer: ReturnType<typeof setTimeout> | undefined, cursor: number | null = null, revision = 0
const monitorStatus = computed(() => !report.value?.running ? 'Stopped' : report.value.stale || report.value.state.communication === 'UNKNOWN' ? 'Connection lost' : report.value.state.stable == null ? 'Connecting…' : report.value.state.stable ? 'Fault' : 'Normal')
const selection = () => ({asset_id:props.assetId, endpoint:props.endpoint, source:source.value})
function changeSource() {
  revision++
  report.value = null; cursor = null; toast.value = ''; error.value = ''; controlError.value = ''
  emit('sourceChange', source.value)
  emit('bindings', [])
  if (source.value === 'simulator') refreshSimulator()
}
async function refreshSimulator() {
  try {
    const data = (await api.get<any>('/api/festo/live-monitor/simulator')).data
    if (disposed || source.value !== 'simulator') return
    simulator.value = data
    emit('bindings', data.nodes.map(({value, ...binding}:any) => binding))
  }
  catch (e) { controlError.value = String(e) }
}
async function simControl(action:string) {
  busy.value = true; controlError.value = ''
  try {
    const path = ['start','stop'].includes(action) ? action : 'state'
    simulator.value = (await api.post<any>('/api/festo/live-monitor/simulator/' + path,
      path === 'state' ? {mode:action} : {})).data
  } catch (e) { controlError.value = String(e) }
  finally { busy.value = false }
}
const labels: Record<string,string> = {BASELINE_FAULT:'Fault already present at baseline', FAULT_RAISED:'New station fault detected',
  FAULT_OBSERVED_AFTER_GAP:'Fault observed after a monitoring gap; onset unknown', RECOVERY_OBSERVED:'Fault bits cleared; physical repair not verified',
  COMMUNICATION_LOST:'Communication / quality unavailable', COMMUNICATION_RESTORED:'Valid communication restored', BINDING_CHANGED:'AAS binding changed; new baseline required'}
const label = (kind:string) => labels[kind] || kind
async function control(action:string) {
  revision++
  busy.value = true; controlError.value = ''
  try {
    if (source.value === 'simulator' && action === 'start' && !simulator.value?.running) {
      simulator.value = (await api.post<any>('/api/festo/live-monitor/simulator/start', {})).data
    }
    report.value = (await api.post<any>('/api/festo/live-monitor/' + action, selection())).data
    emit('sample', {...report.value, source:source.value})
  }
  catch (e) { controlError.value = String(e) }
  finally { busy.value = false }
}
async function poll() {
  if (disposed) return
  const requestedSource = source.value
  const requestedRevision = revision
  try {
    const data = (await api.get<any>('/api/festo/live-monitor', {params:selection()})).data
    if (disposed || requestedSource !== source.value || requestedRevision !== revision) return
    report.value = data; error.value = ''
    const newest = data.events.at(-1)
    if (cursor !== null && newest && newest.seq > cursor) toast.value = label(newest.kind)
    cursor = newest?.seq || 0
    emit('sample', {...data, source:requestedSource})
    if (source.value === 'simulator') await refreshSimulator()
    if (disposed || requestedSource !== source.value || requestedRevision !== revision) return
    const candidate = [...data.events].reverse().find((e:any) => e.kind === 'FAULT_RAISED' && !e.claimed &&
      e.episode_id === data.state.episode_id && Date.now()/1000 - e.at < 300)
    if (candidate && data.running && !data.stale && data.state.communication === 'GOOD' &&
        connectionStatus.value === 'connected' && !props.agentLocked && !store.wizardData.pendingAgentQuestion) {
      const {data:claimed} = await api.post<any>('/api/festo/live-monitor/claim', {...selection(), event_id:candidate.id})
      if (disposed || requestedSource !== source.value) return // Never replay an uncertain handoff.
      if (store.wizardData.pendingAgentQuestion) {
        toast.value = 'Fault handoff reserved but another request is queued. Inspect this event manually.'
        return
      }
      store.updateData('pendingAgentQuestion', claimed.request)
      toast.value = 'Fault detected · Analysis requested'
    }
  } catch (e) {
    if (!disposed && requestedSource === source.value && requestedRevision === revision) {
      error.value = 'OPC monitoring state unavailable: ' + String(e)
      emit('sample', {source:requestedSource, running:report.value?.running, stale:true,
        state:{...report.value?.state, communication:'UNKNOWN'}})
    }
  }
  finally { if (!disposed) timer = setTimeout(poll, 3000) }
}
onMounted(() => { emit('sourceChange', source.value); poll() })
onUnmounted(() => { disposed = true; if (timer) clearTimeout(timer) })
</script>
<style scoped>
.live-events{margin:10px 16px;padding:14px;border:1px solid #35516a;color:#c5d7e6;font-size:14px}
select{background:#172632;color:#d4ebff;border:1px solid #456176;padding:7px;font-size:14px}.test-banner{font-weight:600;color:#ffce79}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px}button{background:#172632;color:#d4ebff;border:1px solid #456176;padding:6px;cursor:pointer}button:disabled{opacity:.4}
p{color:#efc087}small{display:block;color:#98afc1;margin-top:6px}article{padding:8px;border-bottom:1px solid #304352}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:180px;overflow:auto}
.toast{position:fixed;right:24px;top:80px;z-index:2500;max-width:460px;padding:15px;background:#203d53;border:1px solid #6bbbea;box-shadow:0 4px 20px #0008}
.simulator-panel{margin-top:10px;padding-top:10px;border-top:1px solid #35516a}
</style>
