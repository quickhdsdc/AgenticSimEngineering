// Render the migrated Operations UI against browser-local synthetic API fixtures.
// No requests reach a PLC, AAS, MES, Docker, or an online model.
import {createServer} from '../../components/harness/frontend/node_modules/vite/dist/node/index.js'
import {chromium, expect} from '../../components/visualization/node_modules/@playwright/test/index.mjs'
import {mkdir, writeFile} from 'node:fs/promises'
import {fileURLToPath} from 'node:url'
import path from 'node:path'

const root = fileURLToPath(new URL('../../', import.meta.url))
const output = path.join(root, 'work/runs/festo/browser')
await mkdir(output, {recursive:true})
process.env.VITE_APP_BACKEND_URL = 'http://festo.fixture'
process.env.VITE_APP_AAS_SERVER_URL = 'http://festo.fixture/aas'
process.env.VITE_APP_AAS_UI_URL = 'http://festo.fixture/aas-ui'
process.env.VITE_APP_MES_UI_URL = ''
const frontend = path.join(root, 'components/harness/frontend')
const server = await createServer({root:frontend, configFile:path.join(frontend, 'vite.config.mts'),
  envDir:output, server:{host:'127.0.0.1', port:0, open:false}, logLevel:'error'})
let browser
const errors = [], calls = []
try {
  await server.listen()
  browser = await chromium.launch({headless:true})
  const page = await browser.newPage({viewport:{width:1600,height:1000}})
  page.on('pageerror', error => errors.push(error.message))
  let running = false, mode = 'normal', simulatorRunning = false
  const now = () => Date.now()/1000
  const asset = {id:'urn:agenticsim:fixture:festo',idShort:'FestoWorkCenterStamping_Fixture'}
  const order = id => ({id, order_no:`A4S-FIXTURE-${id}`, line_id:1, status:'ACCEPTED', planned_qty:90,
    planned_start_at:new Date(Date.now()+id*3600000).toISOString(), planned_end_at:new Date(Date.now()+id*3600000+2700000).toISOString()})
  const state = {asset, line:{id:1,name:'Fixture press cell'}, backup:{id:2,name:'Fixture backup cell'},
    order_a:order(1),order_b:order(2), issues:[], maintenance:{submodelElements:[]}, context:{},
    mes_url:'http://festo.fixture/mes',mes_ready:true,observed_at:now()}
  const nodes = () => Object.entries({'Press.Fault':mode==='fault','Turning.Fault':false,
    'Press.Ready':mode!=='fault','Turning.Ready':true,'Press.Fault01':mode==='fault',
    'Press.PressureBar':mode==='fault'?3.8:6,'Press.FlowLMin':mode==='fault'?48:12,
    'Safety.EmergencyStop':false}).map(([signal,value]) => ({path:signal,node:'ns=2;s='+signal,
      tag:'SIMULATED.'+signal,value,dataType:typeof value==='boolean'?'Boolean':'Double',status:'Good',
      source_timestamp:new Date().toISOString()}))
  const report = source => ({target:{asset_id:asset.id,endpoint:'http://festo.fixture/aas',source},
    source,running,stale:!running,events:[],state:{communication:running?'GOOD':'UNKNOWN',stable:mode==='fault',
      latest:running?{observed_at:now(),values:nodes(),failed:[]}:null}})
  await page.routeWebSocket('ws://festo.fixture/ws/chat', ws => {
    // Transport only; do not emulate a successful Agent investigation.
    ws.onMessage(() => {})
  })
  await page.route('http://festo.fixture/**', async route => {
    const request = route.request(), url = new URL(request.url())
    const pathname = url.pathname
    calls.push({method:request.method(),path:pathname})
    let body
    if (pathname==='/api/service-links') body={mes_url:null}
    else if (pathname==='/api/demo/festo/asset') body={asset,endpoint:'http://festo.fixture/aas'}
    else if (pathname==='/api/demo/festo') body={...state,observed_at:now()}
    else if (pathname.endsWith('/start/latest')) body={run:null}
    else if (pathname.endsWith('/documents')) body={documents:[]}
    else if (pathname.endsWith('/knowledge')) body={chunks:[]}
    else if (pathname.endsWith('/audit')) body={events:[]}
    else if (pathname.endsWith('/simulator/start')) {simulatorRunning=true;body={running:true,mode,nodes:nodes()}}
    else if (pathname.endsWith('/simulator/state')) {mode=request.postDataJSON().mode;body={running:simulatorRunning,mode,nodes:nodes()}}
    else if (pathname.endsWith('/simulator')) body={running:simulatorRunning,mode,nodes:nodes()}
    else if (pathname==='/api/festo/live-monitor/start') {running=true;body=report(request.postDataJSON().source)}
    else if (pathname==='/api/festo/live-monitor/stop') {running=false;body=report(request.postDataJSON().source)}
    else if (pathname==='/api/festo/live-monitor') body=report(url.searchParams.get('source'))
    else if (pathname.endsWith('/workflow')) body={orders:[{current:state.order_a},{current:state.order_b}],
      lines:[state.line,state.backup],maintenance:[],changes:[],faults:[],activity:[],mes_url:state.mes_url}
    else if (pathname.endsWith('/activity')) body={activity:[],warnings:[],scope:'Synthetic UI fixture'}
    else if (pathname.endsWith('/reset/preview')) body={id:'fixture-preview',endpoint:'http://localhost:8092',
      orders:[state.order_a,state.order_b],maintenance_count:0,effects:['Synthetic preview; no business writes'],
      schedule:[{lot:'A',planned_start_at:state.order_a.planned_start_at,planned_end_at:state.order_a.planned_end_at,due_date:'2026-10-03'},
                {lot:'B',planned_start_at:state.order_b.planned_start_at,planned_end_at:state.order_b.planned_end_at,due_date:'2026-10-03'}]}
    else {errors.push('Unexpected fixture request '+pathname);await route.fulfill({status:409,json:{detail:'Not available in fixture'}});return}
    await route.fulfill({status:200,json:body,headers:{'Access-Control-Allow-Origin':'*'}})
  })
  const address = server.httpServer.address()
  await page.goto(`http://127.0.0.1:${address.port}`)
  await expect(page.getByRole('heading',{name:/Phone housing production/})).toBeVisible()
  const selector = page.getByRole('combobox',{name:'Monitoring source'})
  await expect(selector).toHaveCount(1)
  await selector.selectOption('simulator')
  await page.getByRole('button',{name:'Start monitoring',exact:true}).click()
  const pressure = page.locator('.kpi-grid article').first()
  await expect(pressure).toContainText('6')
  await expect(selector).toBeDisabled()
  await expect(page.getByRole('button',{name:'Start both stations…'})).toHaveCount(0)
  await page.screenshot({path:path.join(output,'normal.png'),fullPage:true})
  await page.getByRole('button',{name:'Fault / leak',exact:true}).click()
  await expect(pressure).toContainText('3.8',{timeout:10000})
  await expect(page.locator('.live-events')).toContainText('Test fault')
  await page.screenshot({path:path.join(output,'fault.png'),fullPage:true})
  await page.getByRole('button',{name:'Normal',exact:true}).click()
  await expect(pressure).toContainText('6',{timeout:10000})
  await page.getByRole('button',{name:'Stop',exact:true}).click()
  await expect(selector).toBeEnabled()
  await selector.selectOption('live')
  await expect(pressure).not.toContainText('3.8')
  await page.getByRole('button',{name:/Production/,exact:false}).first().click()
  await page.getByRole('button',{name:'Reset demo MES…',exact:true}).click()
  const dialog = page.getByRole('dialog',{name:'Reset local demo MES?'})
  await expect(dialog).toBeVisible()
  await expect(dialog.getByRole('button',{name:'Reset MES and create new plan'})).toBeDisabled()
  await expect(dialog).toContainText('Lot A:')
  await expect(dialog).toContainText('Lot B:')
  await page.screenshot({path:path.join(output,'reset-preview.png'),fullPage:true})
  await dialog.getByRole('button',{name:'Cancel',exact:true}).click()
  expect(calls.filter(call=>call.path.endsWith('/reset/execute'))).toHaveLength(0)
  expect(errors).toEqual([])
  const result = {status:'passed',scope:'Rendered UI with browser-local synthetic fixtures; no physical/MES/LLM requests',
    checks:['one source selector','OPC read-back display','fault/recovery display','source switch clears stale values',
      'physical Start hidden in simulation','reset preview slots and confirmation gate'],calls,screenshots:['normal.png','fault.png','reset-preview.png']}
  await writeFile(path.join(output,'report.json'),JSON.stringify(result,null,2))
  console.log(JSON.stringify({status:result.status,report:path.join(output,'report.json')}))
} catch(error) {
  await writeFile(path.join(output,'report.json'),JSON.stringify({status:'failed',error:String(error),errors,calls},null,2))
  throw error
} finally {
  if(browser)await browser.close()
  await server.close()
}
