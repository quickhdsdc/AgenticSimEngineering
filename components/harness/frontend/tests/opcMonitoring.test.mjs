import {test} from 'node:test'
import assert from 'node:assert/strict'
import {appendPressure, projectReport, signalRows} from '../src/services/opcMonitoring.mjs'

const sample = {observed_at: 100, values: [{path:'Press.PressureBar', value:6}, {path:'Press.Fault', value:false}]}
const report = {source:'simulator', running:true, stale:false, state:{communication:'GOOD', latest:sample}}

test('Switching source rejects late readings from the previous OPC source', () => {
  assert.equal(projectReport(report, 'live'), null)
  assert.equal(projectReport({...report, source:'live'}, 'simulator'), null)
  assert.equal(projectReport(report, 'simulator').sample.values[0].value, 6)
})
test('Stopped or stale monitoring never presents readings as current evidence', () => {
  assert.equal(projectReport({...report, running:false, stale:true}, 'simulator').sample.success, false)
  assert.match(projectReport({...report, stale:true}, 'simulator').error, /unknown/)
  assert.equal(projectReport({...report, state:{communication:'UNKNOWN'}}, 'simulator').sample, null)
})
test('Simulator commands do not appear in the node table before read-back', () => {
  const bindings = [{path:'Press.Fault', value:true, node:'ns=2;s=Press.Fault'}]
  assert.equal(signalRows(null, bindings)[0].value, undefined)
  assert.equal(signalRows(sample, bindings)[0].value, false)
})
test('Pressure trend uses fresh timestamped readings and keeps only sixty seconds', () => {
  const history = appendPressure([], sample)
  assert.equal(appendPressure(history, sample), history)
  assert.deepEqual(appendPressure(history, {...sample, observed_at:161}), [{at:161,value:6}])
  assert.equal(appendPressure(history, {observed_at:105, values:[{path:'Press.PressureBar', value:'unknown'}]}), history)
})
