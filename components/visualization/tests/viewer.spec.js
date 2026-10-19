import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';

test('real 3D canvas, candidate isolation, event boundaries and job evidence', async ({ page }) => {
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  await expect(page.locator('#candidate')).toBeEnabled();
  await expect(page.locator('#viewport canvas')).toBeVisible();
  await expect(page.locator('#duration')).toHaveText('4.000 s');
  await page.selectOption('#candidate', { label: 'baseline' });
  await page.locator('#time').fill('4');
  await expect(page.locator('#queued')).toHaveText('1');
  await expect(page.locator('#processing')).toHaveText('1');
  await expect(page.locator('#completed')).toHaveText('2');
  await page.locator('tr[data-job="job-2"]').click();
  await expect(page.locator('#selection')).toContainText('开始 4.000 s');
  await page.screenshot({ path: '../../work/runs/browser-tests/queue-3d.png', fullPage: true });
  await page.selectOption('#candidate', { label: 'longer-service' });
  await expect(page.locator('#clock')).toHaveText('0.000 s');
  await expect(page.locator('#assessment')).toHaveText('模型约束未通过');
  await expect(page.locator('#completed')).toHaveText('0');
  await expect(page.locator('#duration')).toHaveText('12.000 s');
  await page.locator('#time').fill('12');
  await expect(page.locator('#completed')).toHaveText('4');
  await page.getByRole('button', { name: '回到开始' }).click();
  await expect(page.locator('#clock')).toHaveText('0.000 s');
  await page.getByRole('button', { name: '播放', exact: true }).click();
  await expect(page.getByRole('button', { name: '暂停', exact: true })).toBeVisible();
  await page.waitForTimeout(150);
  await page.getByRole('button', { name: '暂停', exact: true }).click();
  expect(Number.parseFloat(await page.locator('#clock').textContent())).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});

test('invalid file digest never produces a successful scene', async ({ page }) => {
  await page.route('**/replay.json', route => route.fulfill({ json: { changed: true } }));
  await page.goto('/');
  await expect(page.locator('#message')).toContainText('文件摘要不匹配');
  await expect(page.locator('#play')).toBeDisabled();
  await expect(page.locator('#viewport canvas')).toHaveCount(0);
});

test('failed candidate clears previous successful scene and timeline', async ({ page, request }) => {
  const replay = await (await request.get('/replay.json')).json();
  const manifest = await (await request.get('/bundle.json')).json();
  replay.runs[0] = { ...replay.runs[0], status: 'timed_out', jobs: [], events: [], event_times: [], duration: null,
    error: { message: 'synthetic browser timeout fixture' } };
  const body = JSON.stringify(replay);
  manifest.files['replay.json'] = createHash('sha256').update(body).digest('hex');
  await page.route('**/replay.json', route => route.fulfill({ body, contentType: 'application/json' }));
  await page.route('**/bundle.json', route => route.fulfill({ json: manifest }));
  await page.goto('/');
  await expect(page.locator('#candidate')).toBeEnabled();
  await page.selectOption('#candidate', { label: 'baseline' });
  await expect(page.locator('#message')).toContainText('没有可回放');
  await expect(page.locator('#play')).toBeDisabled();
  await expect(page.locator('#time')).toBeDisabled();
  await expect(page.locator('#completed')).toHaveText('—');
  await expect(page.locator('#jobs tr')).toHaveCount(0);
  await page.selectOption('#candidate', { label: 'shorter-service' });
  await expect(page.locator('#play')).toBeEnabled();
  await expect(page.locator('#completed')).toHaveText('0');
});
