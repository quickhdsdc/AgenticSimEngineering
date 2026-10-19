<template>
  <div class="console-overlay" @click.self="$emit('close')">
    <div class="console">
      <header class="console-head">
        <div>
          <h3>{{ asset.label }}</h3>
          <HelpTip :text="asset.aasId || 'No AAS instance'" />
        </div>
        <div class="console-head-right">
          <span class="poll-state" :class="{ 'poll-live': polling }">
            {{ polling ? `live · every ${intervalSeconds}s` : 'paused' }}
          </span>
          <button class="btn btn-secondary btn-sm" @click="polling = !polling">
            {{ polling ? 'Pause' : 'Resume' }}
          </button>
          <button class="close-btn" @click="$emit('close')">×</button>
        </div>
      </header>

      <!-- A server that cannot be reached is said so, per server. One subnet
           down must not blank the whole cell. -->
      <div v-if="aasMissing" class="console-warn">
        <strong>This asset has no AAS in the repository.</strong>
        <div class="warn-line">{{ loadError }}</div>
      </div>

      <div v-if="unreachable.length" class="console-warn">
        <strong>{{ unreachable.length }} of {{ endpointCount }} servers unreachable.</strong>
        <div v-for="f in unreachable" :key="f.endpoint" class="warn-line">
          {{ f.endpoint }} — {{ f.error }}
        </div>
      </div>

      <div class="console-body">
        <!-- ---------------------------------------------------------- -->
        <section class="panel">
          <h4>Monitoring</h4>
          <p v-if="loadError" class="panel-empty">{{ loadError }}</p>
          <p v-else-if="!groups.length" class="panel-empty">
            {{ loading ? 'Reading…' : 'This AAS names no OPC UA value sources.' }}
          </p>

          <div v-for="group in groups" :key="group.name" class="group">
            <h5>{{ group.name || 'Values' }}</h5>
            <table class="readings">
              <tbody>
                <tr v-for="v in group.values" :key="v.path">
                  <!-- Both names. The semantic one says what it means; the PLC
                       tag says what to search for in TIA Portal, and an
                       operator holding one screen against the other needs to
                       see that they are the same signal. -->
                  <td class="reading-name" :title="v.description">
                    {{ v.idShort }}
                    <span class="tag-name">{{ v.tag }}</span>
                  </td>
                  <td class="reading-value">
                    <span v-if="v.dataType === 'Boolean'"
                          class="bit" :class="bitClass(v)">
                      {{ v.value ? 'TRUE' : 'FALSE' }}
                    </span>
                    <span v-else class="number">{{ v.value }}</span>
                  </td>
                  <td class="reading-meta">{{ v.dataType }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <HelpTip v-if="rawCounts" text="Raw sensor counts; calibration is not available in the AAS." />
        </section>

        <!-- ---------------------------------------------------------- -->
        <section class="panel">
          <h4>Control</h4>
          <p v-if="!operations.length" class="panel-empty">
            {{ loading ? 'Reading…' : 'This AAS declares no OPC UA operations.' }}
          </p>

          <div v-for="op in operations" :key="op.path" class="action">
            <div class="action-head">
              <strong>{{ op.idShort }}</strong>
              <span class="tag" :class="approvalClass(op)">{{ approvalLabel(op) }}</span>
            </div>
            <HelpTip :text="`${op.description} · ${op.tag} · ${op.endpoint}`" />
            <button
              class="btn btn-danger btn-sm"
              :disabled="running === op.path"
              @click="confirming = op"
            >
              {{ running === op.path ? 'Sending…' : 'Run' }}
            </button>
          </div>

          <div v-if="lastResult" class="action-result">
            <div>Wrote <strong>{{ lastResult.wrote }}</strong> to {{ lastResult.node }}</div>
            <div>Read back: <strong>{{ String(lastResult.readBack) }}</strong></div>
            <div v-if="!lastResult.standing" class="action-note">
              The value did not stand. That is normal for a trigger bit the PLC
              program clears in the same scan — it is not a failed write.
            </div>
          </div>
        </section>

        <!-- ---------------------------------------------------------- -->
        <section class="panel panel-agent">
          <h4>Ask the agent</h4>
          <div class="agent-suggestions">
            <button
              v-for="q in suggestions"
              :key="q"
              class="btn btn-secondary btn-sm suggestion"
              @click="askAgent(q)"
            >
              {{ q }}
            </button>
          </div>
          <textarea
            v-model="question"
            class="agent-input"
            rows="3"
            placeholder="Ask about this machine…"
          ></textarea>
          <button class="btn btn-primary btn-sm" :disabled="!question.trim()"
                  @click="askAgent(question)">
            Send to agent
          </button>
          <p v-if="sentNote" class="panel-note sent">{{ sentNote }}</p>
        </section>
      </div>

      <!-- A start is confirmed on its own, naming the node. The button says
           "Run"; this says what running means. -->
      <div v-if="confirming" class="confirm-overlay" @click.self="confirming = null">
        <div class="confirm">
          <h4>Start a physical machine?</h4>
          <p>
            <strong>{{ confirming.idShort }}</strong> writes
            <code>true</code> to
          </p>
          <p class="action-tag">{{ confirming.tag }}</p>
          <p class="action-node">{{ confirming.endpoint }}<br />{{ confirming.node }}</p>
          <p class="confirm-warn">
            This acts on real equipment. Make sure the cell is clear.
          </p>
          <div class="confirm-actions">
            <button class="btn btn-secondary" @click="confirming = null">Cancel</button>
            <button class="btn btn-danger" @click="runOperation(confirming)">
              Start it
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import type { AssetInstance } from '@/types'
import { api } from '../services/api'
import { useWizardStore } from '../stores/wizardStore'

/**
 * A console for one machine, addressed entirely through its AAS.
 *
 * Nothing here knows an endpoint or a node id. It asks the server for the
 * submodels of this shell, and the server reads the `valueSource` and
 * `invocationDelegation` qualifiers to find out where the values come from and
 * what can be asked. A console with its own address list would drift from the
 * model the moment anything moved, and would then be showing an operator
 * something the AAS does not say.
 */

interface Props {
  asset: AssetInstance
}

const props = defineProps<Props>()
defineEmits<{ close: [] }>()

const store = useWizardStore()

interface Reading {
  path: string
  idShort: string
  tag: string
  group: string
  endpoint: string
  node: string
  value: unknown
  dataType: string
  description: string
}

interface Operation {
  path: string
  idShort: string
  tag: string
  endpoint: string
  node: string
  description: string
  requiresSafetyApproval: boolean | null
}

const readings = ref<Reading[]>([])
const operations = ref<Operation[]>([])
const unreachable = ref<Array<{ endpoint: string; error: string }>>([])
const loading = ref(true)
const loadError = ref('')
const polling = ref(true)
const running = ref('')
const confirming = ref<Operation | null>(null)
const lastResult = ref<any>(null)
const question = ref('')
const sentNote = ref('')
// True when the shell itself is gone, rather than unreadable for some other
// reason. The two need different things from the operator.
const aasMissing = ref(false)

// One second. The read is one OPC UA session per server per tick, and the
// servers answered a full ten-node read in well under that; below a second the
// browser would be asking faster than a PLC scan changes anything.
const intervalSeconds = 1
let timer: number | undefined

// Which submodels to ask about. Derived from the shell rather than named:
// an instance's submodel ids carry its own id, so hard-coding them would work
// for exactly one instantiation.
const submodels = ref<{ operational?: string; services?: string }>({})

const endpointCount = computed(() => {
  const seen = new Set(readings.value.map(r => r.endpoint))
  unreachable.value.forEach(f => seen.add(f.endpoint))
  return seen.size
})

const groups = computed(() => {
  const byGroup = new Map<string, Reading[]>()
  for (const r of readings.value) {
    const key = r.group || ''
    if (!byGroup.has(key)) byGroup.set(key, [])
    byGroup.get(key)!.push(r)
  }
  return [...byGroup.entries()].map(([name, values]) => ({ name, values }))
})

const rawCounts = computed(() => readings.value.some(r => r.dataType.startsWith('Int')))

const suggestions = computed(() => [
  `What is the status of ${props.asset.label}?`,
  'Are any of its modules reporting an error?',
  'Explain what the pressure and flow readings mean.',
])

/**
 * `Error` true is red; `Busy` true is working; `Ready` true is good.
 *
 * A bit is not good or bad on its own -- it depends which bit it is -- and
 * colouring every `true` green would show a fault as healthy.
 */
const bitClass = (v: Reading): string => {
  const on = v.value === true
  if (/error|fault|alarm/i.test(v.idShort)) return on ? 'bit-bad' : 'bit-idle'
  if (/busy|running/i.test(v.idShort)) return on ? 'bit-busy' : 'bit-idle'
  return on ? 'bit-good' : 'bit-idle'
}

const approvalLabel = (op: Operation): string => {
  if (op.requiresSafetyApproval === null) return 'undeclared'
  return op.requiresSafetyApproval ? 'needs approval' : 'no approval needed'
}

const approvalClass = (op: Operation): string => {
  if (op.requiresSafetyApproval === null) return 'tag-unknown'
  return op.requiresSafetyApproval ? 'tag-approval' : 'tag-free'
}

const discover = async (): Promise<void> => {
  const aasId = props.asset.aasId
  if (!aasId) {
    loadError.value = 'This asset has no AAS instance on the server yet.'
    loading.value = false
    return
  }
  try {
    // The shell with its submodels resolved. Which submodel holds the readings
    // is decided by its `idShort`, not by its id: an instance's ids carry the
    // instance name, so matching on them would work for one instantiation.
    const { data } = await api.get<any>(
      `/api/aas/shells/${encodeURIComponent(aasId)}/full`
    )
    const found: any[] = data.submodels || []
    const byIdShort = (name: string) =>
      found.find(sm => (sm.idShort || '') === name)?.id
    submodels.value = {
      operational: byIdShort('OperationalData'),
      services: byIdShort('services_delegated'),
    }
    if (!submodels.value.operational) {
      loadError.value = 'This AAS has no OperationalData submodel to read.'
    }
  } catch (error) {
    // A 404 here has one common cause and one useful answer: the AAS this
    // asset was instantiated as is no longer in the repository, because the
    // repository was restarted or the shell was deleted. The asset is still on
    // the shop floor because the browser remembers it, so the operator is
    // looking at a tile whose machine is gone. Saying "404 Element does not
    // exist" leaves them to work that out.
    const text = error instanceof Error ? error.message : String(error)
    const missing = text.includes('404') || text.toLowerCase().includes('does not exist')
    loadError.value = missing
      ? `This asset's AAS is no longer in the repository. It was probably ` +
        `removed, or the repository was restarted since the asset was placed. ` +
        `Remove the tile and drop a new one to instantiate it again.`
      : `Could not read the AAS: ${text}`
    aasMissing.value = missing
  }
}

const refresh = async (): Promise<void> => {
  if (!submodels.value.operational) return
  try {
    const { data } = await api.post<any>('/api/opc/read', {
      submodel: submodels.value.operational,
    }, { timeout: 20000 })
    readings.value = data.values || []
    unreachable.value = data.failed || []
    loadError.value = ''
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : String(error)
  } finally {
    loading.value = false
  }
}

const loadOperations = async (): Promise<void> => {
  if (!submodels.value.services) return
  try {
    const { data } = await api.get<any>(
      `/api/opc/operations/${encodeURIComponent(submodels.value.services)}`
    )
    operations.value = data.operations || []
  } catch {
    operations.value = []
  }
}

const runOperation = async (op: Operation): Promise<void> => {
  confirming.value = null
  running.value = op.path
  try {
    const { data } = await api.post<any>('/api/opc/invoke', {
      submodel: submodels.value.services,
      path: op.path,
      value: 'true',
    }, { timeout: 20000 })
    lastResult.value = data
    await refresh()
  } catch (error) {
    lastResult.value = {
      wrote: 'true', node: op.node, readBack: 'failed', standing: false,
      error: error instanceof Error ? error.message : String(error),
    }
  } finally {
    running.value = ''
  }
}

/**
 * Hand the question to the chat pane rather than opening a second agent.
 *
 * One session per browser, one ledger thread, one place the approval prompt
 * appears. A console with its own agent would be a second unwatched runner
 * against the same machine.
 */
const askAgent = (text: string): void => {
  const aasId = props.asset.aasId || props.asset.label
  store.updateData('pendingAgentQuestion',
    `${text}\n\n(About the AAS ${aasId}. Read it from the repository; ` +
    `do not start anything unless I ask.)`)
  sentNote.value = 'Sent to the chat pane. Close this console to read the reply.'
  question.value = ''
}

onMounted(async () => {
  await discover()
  await Promise.all([refresh(), loadOperations()])
  timer = window.setInterval(() => {
    if (polling.value && !aasMissing.value) refresh()
  }, intervalSeconds * 1000)
})

onBeforeUnmount(() => {
  if (timer) window.clearInterval(timer)
})
</script>

<style scoped>
.console-overlay {
  position: fixed;
  inset: 0;
  background: rgba(20, 24, 28, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 3000;
  padding: 24px;
}

.console {
  background: #fff;
  border-radius: 10px;
  width: min(1180px, 100%);
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
  overflow: hidden;
}

.console-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #e8e8e8;
  background: #f7f9fa;
}

.console-head h3 { margin: 0; font-size: 18px; color: #222; }
.console-sub { margin: 2px 0 0; font-size: 12px; color: #888; word-break: break-all; }
.console-head-right { display: flex; align-items: center; gap: 10px; }

.poll-state { font-size: 12px; color: #999; }
.poll-live { color: #179C7D; font-weight: 600; }

.console-warn {
  padding: 10px 20px;
  background: #fdf3f3;
  border-bottom: 1px solid #f3d6d6;
  font-size: 13px;
  color: #8a3b3b;
}
.warn-line { color: #a05a5a; font-size: 12px; margin-top: 3px; }

.console-body {
  display: grid;
  grid-template-columns: 1.25fr 1fr 1fr;
  gap: 0;
  overflow: auto;
}

.panel { padding: 18px 20px; border-right: 1px solid #eee; }
.panel:last-child { border-right: none; }
.panel h4 { margin: 0 0 12px; font-size: 15px; color: #333; }
.panel-empty { color: #999; font-size: 13px; }
.panel-note { color: #777; font-size: 12px; line-height: 1.55; margin-top: 12px; }
.panel-note.sent { color: #179C7D; }

.group { margin-bottom: 16px; }
.group h5 {
  margin: 0 0 6px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #999;
}

.readings { width: 100%; border-collapse: collapse; }
.readings td { padding: 5px 0; border-bottom: 1px solid #f2f2f2; font-size: 13px; }
.reading-name { color: #444; }
.tag-name {
  display: block;
  color: #b0b6bc;
  font-size: 10.5px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  margin-top: 1px;
}
.action-tag {
  margin: 4px 0 0;
  font-size: 11px;
  color: #8a929a;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  word-break: break-all;
}
.reading-value { text-align: right; padding-right: 10px !important; }
.reading-meta { color: #bbb; font-size: 11px; text-align: right; width: 56px; }

.bit {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.03em;
}
.bit-good { background: #e6f7f1; color: #12775e; }
.bit-busy { background: #fff4e0; color: #9a6410; }
.bit-bad  { background: #fdeaea; color: #b13333; }
.bit-idle { background: #f0f2f4; color: #8a929a; }

.number { font-variant-numeric: tabular-nums; font-weight: 600; color: #333; }

.action {
  border: 1px solid #eee;
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 10px;
}
.action-head { display: flex; justify-content: space-between; align-items: center; }
.action-desc { margin: 6px 0; font-size: 12px; color: #666; line-height: 1.5; }
.action-node { margin: 0 0 8px; font-size: 11px; color: #aaa; word-break: break-all; }

.tag { font-size: 10px; padding: 2px 7px; border-radius: 9px; font-weight: 700; }
.tag-approval { background: #fdeaea; color: #b13333; }
.tag-free { background: #eef1f4; color: #667; }
.tag-unknown { background: #fff4e0; color: #9a6410; }

.action-result {
  margin-top: 12px;
  padding: 10px;
  background: #f7f9fa;
  border-radius: 6px;
  font-size: 12px;
  color: #444;
  line-height: 1.6;
}
.action-note { color: #9a6410; margin-top: 4px; }

.panel-agent { background: #fbfcfd; }
.agent-suggestions { display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px; }
.suggestion { text-align: left; font-size: 12px; }
.agent-input {
  width: 100%;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 8px;
  font-size: 13px;
  font-family: inherit;
  resize: vertical;
  margin-bottom: 8px;
}

.confirm-overlay {
  position: absolute;
  inset: 0;
  background: rgba(20, 24, 28, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
}
.confirm {
  background: #fff;
  border-radius: 8px;
  padding: 22px;
  width: min(420px, 90%);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}
.confirm h4 { margin: 0 0 10px; color: #b13333; }
.confirm p { font-size: 13px; color: #444; margin: 6px 0; }
.confirm-warn { color: #b13333; font-weight: 600; }
.confirm-actions { display: flex; gap: 10px; justify-content: flex-end; margin-top: 16px; }

.close-btn {
  border: none;
  background: transparent;
  font-size: 24px;
  line-height: 1;
  cursor: pointer;
  color: #999;
}
</style>
