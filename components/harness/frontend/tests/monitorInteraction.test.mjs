import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFileSync, writeFileSync, mkdirSync, unlinkSync} from 'node:fs'
import {parse, compileScript} from '@vue/compiler-sfc'
import {transform} from 'esbuild'
import {createRenderer, h, ref, nextTick} from 'vue'

const renderer = createRenderer({
  createElement: tag => ({tag, tagName:tag.toUpperCase(), props:{}, children:[],
    addEventListener(){}, removeEventListener(){}, get options(){return this.children},
    getAttribute(key){return this.props[key]}, dispatchEvent(){}}),
  createText: text => ({text, children:[]}), createComment: text => ({comment:text, children:[]}),
  setText: (node,text) => { node.text = text }, setElementText: (node,text) => { node.text = text; node.children = [] },
  patchProp: (node,key,old,value) => { node.props[key] = value; if (key === 'value') node.value = value },
  insert(node,parent,anchor) {
    if (node.parent) node.parent.children.splice(node.parent.children.indexOf(node),1)
    const index = parent.children.indexOf(anchor)
    parent.children.splice(index < 0 ? parent.children.length : index, 0, node); node.parent = parent
  },
  remove(node) { const index = node.parent?.children.indexOf(node); if (index >= 0) node.parent.children.splice(index,1) },
  parentNode: node => node.parent, nextSibling: node => node.parent?.children[node.parent.children.indexOf(node)+1],
})
const text = node => (node.text || '') + node.children.map(text).join('')
const find = (node, predicate) => [ ...(predicate(node) ? [node] : []), ...node.children.flatMap(child => find(child,predicate)) ]
const settle = async () => { for (let i=0;i<12;i++) { await new Promise(resolve => setImmediate(resolve)); await nextTick() } }

async function component(name, mocks, cleanup) {
  const filename = 'src/components/' + name + '.vue'
  const {descriptor} = parse(readFileSync(new URL('../' + filename, import.meta.url),'utf8'), {filename})
  let code = compileScript(descriptor, {id:name, inlineTemplate:true, genDefaultAs:'__component'}).content
  for (const [pattern, replacement] of [
    [/import \{\s*api\s*\} from ['"]\.\.\/services\/api['"]/g, 'const {api} = globalThis.__monitorMocks'],
    [/import \{\s*useWizardStore\s*\} from ['"]\.\.\/stores\/wizardStore['"]/g, 'const {useWizardStore} = globalThis.__monitorMocks'],
    [/import \{\s*useEnvStore\s*\} from ['"]\.\.\/stores\/environmentStore['"]/g, 'const {useEnvStore} = globalThis.__monitorMocks'],
    [/import \{\s*useSharedWebSocket\s*\} from ['"]\.\.\/services\/websocketService['"]/g, 'const {useSharedWebSocket} = globalThis.__monitorMocks'],
    [/import LiveFaultMonitor from ['"]\.\/LiveFaultMonitor.vue['"]/g, 'const {LiveFaultMonitor} = globalThis.__monitorMocks'],
    [/import ProductionWorkspace from ['"]\.\/ProductionWorkspace.vue['"]/g, 'const ProductionWorkspace = {render: () => null}'],
  ]) code = code.replace(pattern, replacement)
  code = code.replace(/from ['"](\.[^'"]+)['"]/g,
    (_, specifier) => 'from ' + JSON.stringify(new URL('../src/components/' + specifier, import.meta.url).href))
  // Generated modules live outside the frontend; resolve Vue from this test's
  // installed dependency tree so both renderer and component share one runtime.
  code = code.replace(/from ['"]vue['"]/g, 'from ' + JSON.stringify(import.meta.resolve('vue')))
  const result = await transform(code + '\nexport default __component', {loader:'ts', format:'esm'})
  const directory = new URL('../../../../work/runs/festo/frontend-fixtures/', import.meta.url)
  mkdirSync(directory, {recursive:true})
  const file = new URL(name + '-' + Date.now() + '.mjs', directory)
  writeFileSync(file, result.code); cleanup.push(file)
    globalThis.__monitorMocks = mocks
  return (await import(file.href)).default
}

test('One source selector drives the dashboard and simulation uses the OPC monitor', async () => {
  globalThis.document = {activeElement:null}
  const cleanup = [], calls = []
  let running = false, fault = false, observedAt = 100
  const bindings = ['Press.Fault','Turning.Fault','Press.Ready','Turning.Ready','Press.Fault01','Press.PressureBar','Press.FlowLMin','Safety.EmergencyStop'].map(path => ({path, node:'ns=2;s='+path, tag:'SIMULATED.'+path, value:path === 'Press.PressureBar' ? 6 : false}))
  const state = {asset:{id:'urn:festo:demo',idShort:'FestoWorkCenterStamping_99'}, issues:[], maintenance:{submodelElements:[]}, context:{}, line:{id:1}, backup:{id:2}, order_a:{id:1}, order_b:{id:2}, mes_ready:true}
  const nodes = () => bindings.map(row => ({...row, value:row.path === 'Press.PressureBar' ? fault ? 3.8 : 6 : row.path === 'Press.Fault' ? fault : row.value}))
  const report = source => ({source, target:{source}, running, stale:!running, events:[], state:{communication:'GOOD', latest: running ? {observed_at:++observedAt, values:nodes().map(row => ({...row,status:'Good'}))} : null}})
  const mocks = {
    useWizardStore: () => ({wizardData:{},updateData:()=>{}}),
    useEnvStore: () => ({aasUiUrl:'http://aas',backendUrl:'http://backend'}),
    useSharedWebSocket: () => ({connectionStatus:ref('disconnected')}),
    api:{
      async get(url, options) {
        calls.push(url)
        if (url.endsWith('/asset')) return {data:{asset:state.asset,endpoint:'http://aas'}}
        if (url.endsWith('/simulator')) return {data:{running:true,mode:fault?'fault':'normal',nodes:nodes()}}
        if (url.endsWith('/live-monitor')) return {data:report(options.params.source)}
        if (url.endsWith('/audit')) return {data:{events:[]}}
        if (url.endsWith('/documents')) return {data:{documents:[]}}
        if (url.endsWith('/latest')) return {data:{run:null}}
        if (url === '/api/demo/festo') return {data:state}
        throw Error('Unexpected request ' + url)
      },
      async post(url, body) {
        calls.push(url)
        if (url.endsWith('/simulator/state')) {fault=body.mode === 'fault'; return {data:{running:true,mode:body.mode,nodes:nodes()}}}
        if (url.endsWith('/start')) {running=true; return {data:report(body.source)}}
        if (url.endsWith('/stop')) {running=false; return {data:report(body.source)}}
        throw Error('Unexpected write ' + url)
      },
    },
  }
  let app
  try {
    mocks.LiveFaultMonitor = await component('LiveFaultMonitor', mocks, cleanup)
    const workspace = await component('RecoveryWorkspace', mocks, cleanup)
    const host = {children:[]}
    app = renderer.createApp({render:()=>h(workspace, {assetId:state.asset.id,endpoint:'http://aas'})})
    app.component('HelpTip', {render:()=>null}); app.mount(host)
    await settle()
    const selectors = find(host, node => node.tag === 'select')
    assert.equal(selectors.length,1)
    const select = selectors[0]
    select.props['onUpdate:modelValue']('simulator'); select.props.onChange()
    await settle()
    assert.match(text(host), /AIR PRESSURE · SIM/)
    assert.equal(find(host, node => node.tag === 'tr' && text(node).includes('Not read')).length,8)
    assert.doesNotMatch(text(host), /Start server|Inject air leak|Restore simulated pressure|SIM · HTTP STREAM/)
    const start = find(host, node => node.tag === 'button' && text(node) === 'Start monitoring')[0]
    await start.props.onClick(); await settle()
    assert.equal(select.props.disabled, true)
    assert.match(text(host), /6bar/)
    const faultButton = find(host, node => node.tag === 'button' && text(node) === 'Fault / leak')[0]
    await faultButton.props.onClick(); await settle()
    assert.match(text(host), /6bar/, 'Commands must wait for OPC read-back')
    await new Promise(resolve => setTimeout(resolve,3100)); await settle()
    assert.match(text(host), /3.8bar/)
    const stop = find(host, node => node.tag === 'button' && text(node) === 'Stop')[0]
    await stop.props.onClick(); await settle()
    select.props['onUpdate:modelValue']('live'); select.props.onChange(); await settle()
    assert.match(text(host), /PRESSURE 1 · RAW OPC/)
    assert.doesNotMatch(text(host), /AIR PRESSURE · SIM|3.8bar|6bar/)
    assert.ok(calls.includes('/api/festo/live-monitor/start'))
    assert.ok(calls.every(url => !url.includes('/api/demo/festo/monitor') && url !== '/api/opc/read'))
  } finally {
    app?.unmount()
    cleanup.forEach(file => unlinkSync(file))
    delete globalThis.__monitorMocks
    delete globalThis.document
  }
})
