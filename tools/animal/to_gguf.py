#!/usr/bin/env python3
"""Pack the trained animal network (dognet.bin + dognet.json from train_mann.py) into one GGUF file.

GGUF is the single-file tensor container of ggml / llama.cpp. Here it simply holds this network's float16
weights plus its metadata; it is not a language model and llama.cpp cannot run it. site/js/gguf.js reads it.

    python3 tools/animal/to_gguf.py DIR_WITH_DOGNET out/animalnet.gguf
"""
import json
import struct
import sys
from pathlib import Path

ALIGN = 32
T_UINT32, T_STRING, T_UINT64 = 4, 8, 10
GGML_F16 = 1


def s(b: str) -> bytes:
    e = b.encode()
    return struct.pack('<Q', len(e)) + e


def kv(key: str, typ: int, val) -> bytes:
    out = s(key) + struct.pack('<I', typ)
    if typ == T_STRING: return out + s(val)
    if typ == T_UINT32: return out + struct.pack('<I', val)
    if typ == T_UINT64: return out + struct.pack('<Q', val)
    raise ValueError(typ)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.loads((src / 'dognet.json').read_text()); blob = (src / 'dognet.bin').read_bytes()
    tensors = meta.pop('tensors')
    kvs = [kv('general.architecture', T_STRING, 'mann-quadruped'), kv('general.name', T_STRING, 'MotionBricks animal motion network'),
           kv('general.license', T_STRING, meta.get('license', '')), kv('general.alignment', T_UINT32, ALIGN),
           kv('mann-quadruped.expert_count', T_UINT32, meta['experts']), kv('mann-quadruped.hidden_size', T_UINT32, meta['hidden']),
           kv('mann-quadruped.meta_json', T_STRING, json.dumps(meta, separators=(',', ':')))]
    infos, datas, off = [], [], 0
    for t in tensors:
        n = 1
        for d in t['shape']: n *= d
        raw = blob[t['offset']:t['offset'] + 2 * n]
        pad = -off % ALIGN; datas.append(b'\0' * pad); off += pad
        dims = list(reversed(t['shape']))  # ggml order: innermost dimension first
        infos.append(s(t['name']) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', GGML_F16, off))
        datas.append(raw); off += len(raw)
    head = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(kvs)) + b''.join(kvs) + b''.join(infos)
    head += b'\0' * (-len(head) % ALIGN)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(head + b''.join(datas))
    print(f'{dst}: {len(tensors)} tensors, {(len(head) + off) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
