import { defineConfig } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
if (!process.env.ASE_VIEWER_BUNDLE) throw new Error('Set ASE_VIEWER_BUNDLE to an exported playback bundle');
export default defineConfig({
  testDir: './tests',
  testMatch: process.env.ASE_VIEWER_KIND === 'robot' ? 'robot.spec.js' : 'viewer.spec.js',
  outputDir: '../../work/runs/browser-tests/playwright',
  workers: 1,
  reporter: [['list'], ['json', { outputFile: `../../work/runs/browser-tests/${process.env.ASE_VIEWER_CASE || 'report'}.json` }]],
  use: {
    baseURL: 'http://127.0.0.1:8766',
    viewport: { width: 1440, height: 1000 },
    launchOptions: {
      executablePath: process.env.ASE_CHROMIUM || undefined,
      args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'],
    },
  },
  webServer: {
    command: `"${process.env.ASE_VIEWER_PYTHON || path.join(root, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python')}" -m agenticsim visualize serve "${process.env.ASE_VIEWER_BUNDLE}" --port 8766`,
    cwd: root,
    url: 'http://127.0.0.1:8766',
    reuseExistingServer: false,
  },
});
