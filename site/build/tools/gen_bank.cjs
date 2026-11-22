// Generates site/data/bank.bin + bank.json by running build/tools/gen_bank.html in headless Chromium.
// usage: node build/tools/gen_bank.cjs <port of `node build/serve.mjs <port> --isolate`> [limit] [--raw]
// --raw: no joint limits, written to data/raw_output.bin/.json (input of tools/fit_hinges.py; not part of the site)
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const raw = process.argv.includes('--raw'), [port, limit = '0'] = process.argv.slice(2).filter((a) => a !== '--raw');
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('pageerror', e.message));
  await page.goto(`http://localhost:${port}/build/tools/gen_bank.html?limit=${limit}${raw ? '&anat=0' : ''}`);
  let last = 0;
  for (;;) {
    const st = await page.evaluate(() => ({ done: !!window.RESULT, log: window.LOG || [] }));
    for (const l of st.log.slice(last)) console.log(l); last = st.log.length;
    if (st.done) break;
    await page.waitForTimeout(2000);
  }
  const r = await page.evaluate(() => window.RESULT);
  if (r.error) { console.error(r.error); process.exit(1); }
  const out = path.resolve(__dirname, '../../data'), name = raw ? 'raw_output' : 'bank';
  fs.writeFileSync(path.join(out, `${name}.bin`), Buffer.from(r.b64, 'base64'));
  fs.writeFileSync(path.join(out, `${name}.json`), JSON.stringify(r.meta));
  console.log(`wrote ${r.meta.entries.length} entries, ${r.meta.entries.reduce((a, e) => a + e.frames, 0)} frames, ${(Buffer.from(r.b64, 'base64').length / 1e6).toFixed(2)} MB in ${r.seconds.toFixed(0)} s`);
  await browser.close();
})();
