<template>
  <section class="production-workspace">
    <div class="heading"><h2>{{ view==='orders'?'Production plan':view==='incident'?'Incident response':'Activity' }}</h2>
      <a v-if="data?.mes_url" :href="data.mes_url + '/admin/schedule'" target="_blank" rel="noopener noreferrer">Open MES planner ↗</a>
      <button v-if="view==='orders'" class="reset-trigger" @click="previewReset" :disabled="resetBusy">Reset demo MES…</button>
      <button @click="refresh" :disabled="resetBusy">Refresh</button>
    </div>
    <div class="filters"><select aria-label="Time window" v-model.number="hours" @change="refresh"><option :value="1">Last hour</option><option :value="4">Last 4 hours</option><option :value="24">Last 24 hours</option><option :value="0">All time</option></select>
      <label><input type="checkbox" v-model="currentRun" @change="refresh"/> Current run</label><small v-if="data?.started_at">Plan opened {{ date(data.started_at) }}</small></div>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="resetError" class="error" role="alert">{{ resetError }}</p>
    <div v-if="resetResult" class="reset-success" role="status">
      New production plan created: {{ resetResult.orders.length }} orders. Previous MES data was backed up.
      <ul><li v-for="order in resetResult.orders" :key="order.order_no">{{ order.order_no }}: {{ date(order.planned_start_at) }} – {{ date(order.planned_end_at) }}</li></ul>
      <details><summary>Reset details</summary><p>Run {{ resetResult.run_id }}</p><p>Backup: {{ resetResult.backup }}</p></details>
    </div>
    <template v-if="data">
      <template v-if="view==='orders'">
        <div class="summary"><article><small>Planned production</small><strong>{{ quantity }} housings</strong></article><article><small>Order changes</small><strong>{{ data.changes.length }}</strong></article><article><small>Maintenance</small><strong>{{ data.maintenance.length ? 'Planned' : 'None planned' }}</strong></article></div>
        <p class="scope">Planned production times and delivery due dates are separate. Rescheduling updates existing orders.</p><div class="table-wrap"><table><thead><tr><th>Lot</th><th>Status / quantity</th><th>Initial plan</th><th>Current plan</th><th>Due date</th><th>Impact</th></tr></thead><tbody><tr v-for="(row,index) in data.orders" :key="row.current.id"><td><b>Lot {{ index===0?'A':'B' }}</b><small>{{ row.current.order_no }}</small></td><td>{{ row.current.status }}<small>{{ row.current.planned_qty }} pcs</small></td><td>{{ line(row.initial?.line_id) }}<small>{{ slot(row.initial) }}</small></td><td>{{ line(row.current.line_id) }}<small>{{ slot(row.current) }}</small></td><td>{{ row.due_day || '?' }}<small v-if="row.planned_after_due" class="error">Planned after due date</small></td><td :class="{changed:changed(row)}">{{ impact(row) }}</td></tr></tbody></table></div>
        <p class="empty" v-if="!data.changes.length">Initial plan ready. Schedule changes will appear here after replanning.</p>
        <article v-for="m in data.maintenance" :key="m.id" class="card"><b>Maintenance #{{ m.id }} · {{ m.status }}</b><p>{{ m.title }}</p><span>{{ date(m.scheduled_at) }} · {{ line(m.line_id) }} · {{ m.scheduled_end_at ? 'Until '+date(m.scheduled_end_at) : '60-minute reservation' }}</span></article>
      </template>
      <template v-else-if="view==='incident'">
        <div class="steps"><span :class="{done:data.faults.length}">1 · Fault detected</span><span :class="{done:data.faults.some(e=>e.claimed)}">2 · Agent investigating</span><span :class="{done:data.maintenance.length}">3 · Maintenance planned</span><span :class="{done:replanned}">4 · {{ replanned ? 'Production replanned' : 'Production replanning' }}</span></div>
        <p v-if="!data.faults.length" class="empty">No fault recorded in this time window.</p>
        <article v-for="e in [...data.faults].reverse()" :key="e.id" class="card"><b>{{ e.kind==='FAULT_RAISED'?'New station fault':'Observed station fault' }}</b><small>{{ date(e.at*1000) }}</small><div class="signals"><span v-for="signal in transitions(e)" :key="signal.path">{{ signal.path }} <b>{{ signal.before }} → {{ signal.after }}</b></span></div><p>{{ e.claimed ? 'Agent investigation requested. Assessment and approval are in chat.' : 'Evidence recorded. Request an investigation in chat.' }}</p><details><summary>Evidence</summary><pre>{{ JSON.stringify(e,null,2) }}</pre></details></article>
        <article v-for="m in data.maintenance" :key="m.id" class="card"><b>Maintenance #{{ m.id }} · {{ m.status }}</b><h3>{{ m.title }}</h3><p>{{ date(m.scheduled_at) }} · {{ line(m.line_id) }}</p><details><summary>Inspection plan</summary><p>{{ m.description }}</p></details></article>
        <article v-if="data.maintenance.length && !data.changes.length" class="card warning">Maintenance is recorded. Production replanning is awaiting investigation or approval in chat.</article>
        <article v-for="c in data.changes" :key="c.id" class="card"><b>{{ c.order_no }}</b><p>{{ line(c.before.line_id) }} · {{ slot(c.before) }} → {{ line(c.after.line_id) }} · {{ slot(c.after) }}</p><small>{{ date(c.created_at) }} · {{ c.user }}</small></article>
      </template>
      <template v-else><p class="scope">{{ data.scope }}</p><p v-for="warning in data.warnings || []" :key="warning" class="warning">{{ warning }}</p><p v-if="data.ledger_truncated" class="warning">Showing the latest 500 ledger entries. Narrow the time window for older records.</p><p v-if="!data.activity.length" class="empty">No activity in this time window.</p><article v-for="(event,index) in data.activity" :key="event.id || index" class="activity"><time>{{ date(event.at*1000) }}</time><div><b>{{ event.title }}</b><p>{{ event.detail }}</p><details><summary>Details · {{ event.type }}</summary><pre>{{ JSON.stringify(event.evidence,null,2) }}</pre></details></div></article></template>
    </template>
    <div v-if="resetPreview" class="reset-overlay" @click.self="!resetBusy && closeReset()">
      <section class="reset-dialog" role="dialog" aria-modal="true" aria-labelledby="reset-title">
        <h3 id="reset-title">Reset local demo MES?</h3>
        <p>Target: {{ resetPreview.endpoint }}. This replaces the current demo plan with two new accepted orders of 90 housings each.</p>
        <p>{{ resetPreview.orders.length }} existing A4S orders and {{ resetPreview.maintenance_count }} maintenance events will be archived.</p>
        <p>New schedule, calculated from the current clock:</p>
        <ul><li v-for="slot in resetPreview.schedule" :key="slot.lot">Lot {{ slot.lot }}: {{ date(slot.planned_start_at) }} – {{ date(slot.planned_end_at) }} · due {{ slot.due_date }}</li></ul>
        <details><summary>Orders included</summary><ul><li v-for="order in resetPreview.orders" :key="order.id">{{ order.order_no }} · {{ order.status }}</li></ul></details>
        <ul><li v-for="effect in resetPreview.effects" :key="effect">{{ effect }}</li></ul>
        <label class="reset-confirm"><input type="checkbox" v-model="resetApproved" :disabled="resetBusy"/> I reviewed this MES reset and its backup. Stop any Agent task and monitoring before continuing.</label>
        <div class="reset-actions"><button @click="closeReset" :disabled="resetBusy">Cancel</button><button class="reset-execute" @click="executeReset" :disabled="resetBusy || !resetApproved">{{ resetBusy ? 'Resetting MES…' : 'Reset MES and create new plan' }}</button></div>
        <small>These exact slots are used when confirmed; if they become too close to the current time, preview again. AAS and OPC are not changed.</small>
      </section>
    </div>
  </section>
</template>
<script setup lang="ts">
import {computed,onMounted,onUnmounted,ref,watch} from 'vue'
import {api} from '../services/api'
const props = defineProps<{view:string}>()
const data=ref<any>(null), error=ref(''), hours=ref(1), currentRun=ref(true)
const resetPreview=ref<any>(null), resetApproved=ref(false), resetBusy=ref(false), resetError=ref(''), resetResult=ref<any>(null)
let disposed=false, timer:ReturnType<typeof setTimeout>|undefined
const date=(value:any)=>value?new Date(value).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'—'
const line=(id:any)=>data.value?.lines.find((l:any)=>l.id===id)?.name || '—'
const slot=(order:any)=>order?.planned_start_at?`${date(order.planned_start_at)} – ${date(order.planned_end_at)}`:'Unscheduled'
const changed=(row:any)=>row.initial && (row.initial.line_id!==row.current.line_id || row.initial.planned_start_at!==row.current.planned_start_at)
const impact=(row:any)=>!changed(row)?'On original plan':row.initial.line_id!==row.current.line_id?'Transferred to available line':`Delayed ${Math.round((Date.parse(row.current.planned_start_at)-Date.parse(row.initial.planned_start_at))/60000)} min`
const quantity=computed(()=>data.value?.orders.reduce((sum:number,r:any)=>sum+Number(r.current.planned_qty),0)||0)
const replanned=computed(()=>data.value?.orders.length===2 && data.value.orders.every((row:any)=>changed(row)))
function transitions(e:any){const before=new Map((e.before?.values||[]).map((v:any)=>[v.path,v.value]));return (e.after?.values||[]).filter((v:any)=>before.get(v.path)!==v.value).map((v:any)=>({path:v.path,before:String(before.get(v.path)??'—'),after:String(v.value)}))}
async function refresh(){if(resetBusy.value)return;const requestedView=props.view;try{const result=await api.get<any>(props.view==='audit'?'/api/demo/festo/activity':'/api/demo/festo/workflow',{params:{hours:hours.value,current_run:currentRun.value},timeout:60000});if(!disposed && requestedView===props.view){data.value=result.data;error.value=''}}catch(e:any){if(!disposed)error.value=e.message||String(e)}}
async function previewReset(){
  if(resetBusy.value)return
  resetBusy.value=true;resetError.value='';resetResult.value=null
  try{resetPreview.value=(await api.post<any>('/api/demo/festo/reset/preview',{})).data;resetApproved.value=false}
  catch(e:any){resetError.value=e.message||String(e)}
  finally{resetBusy.value=false}
}
function closeReset(){if(resetBusy.value)return;resetPreview.value=null;resetApproved.value=false}
async function executeReset(){
  if(resetBusy.value||!resetApproved.value||!resetPreview.value)return
  const id=resetPreview.value.id
  resetBusy.value=true;resetError.value=''
  try{
    const result=await api.post<any>('/api/demo/festo/reset/execute',{id,confirmed:true},{timeout:12*60*1000})
    resetResult.value=result.data;currentRun.value=true;hours.value=1
  }catch(e:any){resetError.value=e.message||String(e)}
  finally{resetBusy.value=false;resetPreview.value=null;resetApproved.value=false;await refresh()}
}
async function poll(){await refresh();if(!disposed)timer=setTimeout(poll,10000)}
watch(()=>props.view,()=>{data.value=null;void refresh()})
onMounted(poll)
onUnmounted(()=>{disposed=true;if(timer)clearTimeout(timer)})
</script>
<style scoped>
.production-workspace{padding:20px;overflow:auto;min-height:0;font-size:15px}.heading,.filters,.steps{display:flex;gap:16px;align-items:center;flex-wrap:wrap}.heading h2{margin-right:auto}.heading a{color:#38bdf8}button,select{padding:8px 12px;background:#172632;color:#dce9f1;border:1px solid #35516a}h2{font-size:22px}.filters{margin:16px 0;color:#adbdce}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}.summary article,.card{padding:18px;border:1px solid #304352;background:#0f1720}.summary strong{display:block;font-size:23px;margin-top:10px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left}td,th{padding:16px 12px;border-bottom:1px solid #304352}small{display:block;color:#9caec0;font-size:13px;margin-top:6px;overflow-wrap:anywhere}.changed,.done{color:#68dfb2}.card{margin:14px 0}.steps span{border-bottom:3px solid #344352;padding:10px}.steps .done{border-color:#68dfb2}.signals{display:flex;gap:20px;flex-wrap:wrap;margin:16px 0}.signals b{display:block}.activity{display:flex;gap:24px;padding:18px 0;border-bottom:1px solid #304352}.activity time{min-width:130px;color:#a7bace}.empty{padding:24px;color:#a7bace}.warning{border-left:3px solid #dcb567}.error{color:#ffb3a7}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:240px;overflow:auto;font-size:12px}summary{cursor:pointer;color:#8cbed8}
.reset-trigger{border-color:#bd805a;color:#ffcbab}.reset-trigger:disabled{opacity:.5}.reset-success{margin-top:12px;padding:12px;border:1px solid #3e8169;color:#a8edcf;background:#10281f}.reset-overlay{position:fixed;inset:0;background:#000a;display:grid;place-items:center;z-index:3000}.reset-dialog{background:#101b26;color:#e1edf5;border:1px solid #4d6679;padding:24px;width:min(580px,calc(100vw - 32px));max-height:85vh;overflow:auto;box-shadow:0 16px 40px #0008}.reset-dialog h3{margin-top:0;font-size:20px}.reset-dialog li{margin:6px 0}.reset-confirm{display:flex;align-items:flex-start;gap:10px;margin:18px 0}.reset-actions{display:flex;justify-content:flex-end;gap:10px}.reset-execute{background:#9b3d32;border-color:#c75949;color:white}.reset-execute:disabled{opacity:.5}
</style>
