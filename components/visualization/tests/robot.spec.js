import { test, expect } from '@playwright/test';

test('original robot meshes and solver frames render with joint evidence', async ({ page, request }) => {
  const replay = await (await request.get('/replay.json')).json();
  const run = replay.runs.find(r => r.candidate_id === replay.decision.best_observed_candidate)
    ?? replay.runs.find(r => r.status === 'succeeded');
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  await expect(page.locator('#play')).toBeEnabled({ timeout: 30000 });
  await expect(page.locator('#viewport canvas')).toBeVisible();
  await expect(page.locator('#jobs tr')).toHaveCount(run.target_q.length);
  await expect(page.locator('#viewport canvas')).toHaveAttribute('data-body-count', String(run.scene.bodies.length));
  expect(Number(await page.locator('#viewport canvas').getAttribute('data-triangle-count'))).toBeGreaterThan(10000);
  await page.locator('#time').fill(String(run.duration));
  await expect(page.locator('#completed')).toHaveText(`${run.duration.toFixed(3)} s`);
  const last = run.joint_values.at(-1);
  await expect(page.locator('#jobs tr').first().locator('td').nth(1)).toHaveText(last[0].toFixed(4));
  const index = Math.floor(run.target_q.length / 2);
  await page.locator('#jobs tr').nth(index).click();
  await expect(page.locator('#selection')).toContainText(run.scene.bodies[run.scene.joint_bodies[index]]);
  await page.locator('#trace').uncheck();
  await page.screenshot({ path: `../../work/runs/browser-tests/${replay.model.id}-3d.png`, fullPage: true });
  const other = replay.runs.find(r => r.run_id !== run.run_id && r.status === 'succeeded');
  if (other) {
    await page.selectOption('#candidate', other.run_id);
    await expect(page.locator('#play')).toBeEnabled({ timeout: 30000 });
    await expect(page.locator('#clock')).toHaveText('0.000 s');
    await expect(page.locator('#jobs tr').first().locator('td').nth(1)).toHaveText(other.joint_values[0][0].toFixed(4));
  }
  await page.locator('#loop').uncheck();
  await page.locator('#play').click();
  await page.waitForTimeout(150);
  await page.locator('#play').click();
  expect(parseFloat(await page.locator('#clock').textContent())).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});

test('corrupt robot mesh fails visibly and disables playback', async ({ page }) => {
  await page.route('**/mesh-0.bin', route => route.fulfill({ body: Buffer.from('corrupt mesh'), contentType: 'application/octet-stream' }));
  await page.goto('/');
  await expect(page.locator('#message')).toContainText('文件摘要不匹配', { timeout: 30000 });
  await expect(page.locator('#play')).toBeDisabled();
  await expect(page.locator('#jobs tr')).toHaveCount(0);
});
