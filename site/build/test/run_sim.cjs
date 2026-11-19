// Runs build/test/sim.html in headless Chromium for each configuration and prints one JSON line per run.
// usage: node build/test/run_sim.cjs <port> <ctrl:baseline|new> <model> <extraMs,...> [settingsJSON] [jitterMs]
const { chromium } = require('playwright');
(async () => {
  const [port, ctrl, model, extras, settings = '{}', jitter = '0'] = process.argv.slice(2);
  const browser = await chromium.launch();
  for (const extra of extras.split(',')) {
    const page = await browser.newPage();
    const errs = [];
    page.on('pageerror', (e) => errs.push(e.message));
    const qs = new URLSearchParams({ ctrl, model, extra, jitter, settings });
    await page.goto(`http://localhost:${port}/build/test/sim.html?${qs}`);
    await page.waitForFunction(() => window.RESULT, null, { timeout: 180000 }).catch(() => errs.push('timeout'));
    const r = await page.evaluate(() => window.RESULT || window.LOG);
    console.log(JSON.stringify({ ...r, errs }));
    await page.close();
  }
  await browser.close();
})();
