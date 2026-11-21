// node build/test/run_lab.cjs <port> '<variants JSON>' ['<settings JSON>']
const { chromium } = require('playwright');
(async () => {
  const [port, variants, settings = '{}'] = process.argv.slice(2);
  const browser = await chromium.launch(); const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('pageerror', e.message));
  await page.goto(`http://localhost:${port}/build/test/startlab.html?variants=${encodeURIComponent(variants)}&settings=${encodeURIComponent(settings)}`);
  await page.waitForFunction(() => window.RESULT, null, { timeout: 600000 });
  const r = await page.evaluate(() => window.RESULT);
  if (r.error) { console.error(r.error); process.exit(1); }
  for (const { variant, res } of r) console.log(JSON.stringify(variant).padEnd(58), res.map((x) => `pitch ${x.max}@${x.at} hand ${x.handMin} T${x.tokens} ${x.frames}f ${x.dist}m ${x.tgt}`).join(' | '));
  await browser.close();
})();
