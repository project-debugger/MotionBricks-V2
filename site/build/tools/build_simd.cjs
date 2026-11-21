// Compiles build/tools/mix_simd.wat and prints the base64 module embedded in js/animalnet.js (SIMD_MIX).
// usage: NODE_PATH=<dir with the wabt npm package> node build/tools/build_simd.cjs
const fs = require('fs'), path = require('path');
require('wabt')().then((wabt) => {
  const src = fs.readFileSync(path.join(__dirname, 'mix_simd.wat'), 'utf8');
  const mod = wabt.parseWat('mix_simd.wat', src, { simd: true });
  const { buffer } = mod.toBinary({});
  console.log(Buffer.from(buffer).toString('base64'));
});
