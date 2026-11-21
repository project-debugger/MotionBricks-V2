#!/usr/bin/env python3
"""Pack the playground's animal models: Sketchfab glTF exports (CC licences, via the Objaverse mirror on Hugging Face)
re-saved for the web. Animations are dropped (the trained network animates the bones), textures are resized to at most
--max-texture pixels and stored as JPEG unless they need alpha; geometry, skins and materials are kept as they are.

    python3 -I tools/animal/pack_models.py SRC_DIR site/data/animal/models

SRC_DIR holds <uid>.glb files downloaded from
https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/<shard>/<uid>.glb (object-paths.json.gz maps
uid -> shard). Credits and licences: site/js/animalmesh.js (ANIMAL_MODELS) and NOTICE.
"""
import argparse
import io
import json
import struct
from pathlib import Path

from PIL import Image

MODELS = {  # output name -> Sketchfab / Objaverse uid
    'dog': '9abfce885a834399b2c3ccaed51cd474', 'wolf': '9e7aceacef0f4486b605abdf1f62e4ac', 'fox': '4e37c303f8a241aaa2511bd9f3019273',
    'cat': '827aa2c7951747599629700703050673', 'horse': 'bef74ed402cc40e996eea028668d3c1c', 'bear': 'ce0d5eb86cf5459bb6bd20244cb44b27',
    'elephant': '2705e7483341479e8015d996cd766611', 'tiger': 'fc2c7fd74dd04e39935803b884c77258', 'jaguar': '91c61c329d2a4668816f81f08dfcd492',
    'rabbit': 'e7213589744d436b9d96e2dbb31198a5',
}


def read_glb(data: bytes):
    assert data[:4] == b'glTF'
    jlen = struct.unpack('<I', data[12:16])[0]
    j = json.loads(data[20:20 + jlen])
    rest = data[20 + jlen:]
    blen = struct.unpack('<I', rest[:4])[0] if rest else 0
    return j, rest[8:8 + blen] if rest else b''


def write_glb(j: dict, binary: bytes) -> bytes:
    js = json.dumps(j, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    binary += b'\0' * (-len(binary) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return b''.join([struct.pack('<4sII', b'glTF', 2, total), struct.pack('<I4s', len(js), b'JSON'), js, struct.pack('<I4s', len(binary), b'BIN\0'), binary])


def pack(src: bytes, max_tex: int, quality: int) -> bytes:
    j, binary = read_glb(src)
    j.pop('animations', None)
    views = j.get('bufferViews', [])
    used = set()  # buffer views still referenced: accessors of meshes / skins, and images
    acc_used = set()
    for m in j.get('meshes', []):
        for p in m['primitives']:
            acc_used.update(p['attributes'].values())
            if 'indices' in p: acc_used.add(p['indices'])
            for t in p.get('targets', []): acc_used.update(t.values())
    for s in j.get('skins', []):
        if 'inverseBindMatrices' in s: acc_used.add(s['inverseBindMatrices'])
    for i, a in enumerate(j.get('accessors', [])):
        if i in acc_used and 'bufferView' in a: used.add(a['bufferView'])
        if i in acc_used and 'sparse' in a: used.update([a['sparse']['indices']['bufferView'], a['sparse']['values']['bufferView']])
    new_bytes = {}
    for img in j.get('images', []):
        if 'bufferView' not in img: continue
        v = views[img['bufferView']]
        raw = binary[v.get('byteOffset', 0):v.get('byteOffset', 0) + v['byteLength']]
        im = Image.open(io.BytesIO(raw)); im.load()
        if max(im.size) > max_tex:
            im.thumbnail((max_tex, max_tex), Image.LANCZOS)
        alpha = im.mode in ('RGBA', 'LA', 'P') and im.convert('RGBA').getchannel('A').getextrema()[0] < 250
        out = io.BytesIO()
        if alpha:
            im.convert('RGBA').save(out, 'PNG', optimize=True); img['mimeType'] = 'image/png'
        else:
            im.convert('RGB').save(out, 'JPEG', quality=quality, optimize=True); img['mimeType'] = 'image/jpeg'
        new_bytes[img['bufferView']] = out.getvalue()
        used.add(img['bufferView'])
    # rebuild the binary chunk with the used views only
    remap, chunks, off = {}, [], 0
    for i, v in enumerate(views):
        if i not in used: continue
        data = new_bytes.get(i) or binary[v.get('byteOffset', 0):v.get('byteOffset', 0) + v['byteLength']]
        pad = -off % 4
        chunks.append(b'\0' * pad); off += pad
        nv = {k: val for k, val in v.items() if k not in ('byteOffset', 'byteLength', 'buffer')}
        nv.update({'buffer': 0, 'byteOffset': off, 'byteLength': len(data)})
        remap[i] = len(remap); chunks.append(data); off += len(data)
        views[i] = nv
    j['bufferViews'] = [views[i] for i in sorted(remap, key=remap.get)]
    for a in j.get('accessors', []):
        if 'bufferView' in a: a['bufferView'] = remap.get(a['bufferView'], 0)
        if 'sparse' in a:
            for k in ('indices', 'values'): a['sparse'][k]['bufferView'] = remap[a['sparse'][k]['bufferView']]
    for img in j.get('images', []):
        if 'bufferView' in img: img['bufferView'] = remap[img['bufferView']]
    # drop the accessors only animations used, and renumber the rest
    amap = {old: new for new, old in enumerate(sorted(acc_used))}
    j['accessors'] = [j['accessors'][i] for i in sorted(acc_used)]
    for m in j.get('meshes', []):
        for p in m['primitives']:
            p['attributes'] = {k: amap[v] for k, v in p['attributes'].items()}
            if 'indices' in p: p['indices'] = amap[p['indices']]
            if 'targets' in p: p['targets'] = [{k: amap[v] for k, v in t.items()} for t in p['targets']]
    for s in j.get('skins', []):
        if 'inverseBindMatrices' in s: s['inverseBindMatrices'] = amap[s['inverseBindMatrices']]
    binary = b''.join(chunks)
    j['buffers'] = [{'byteLength': len(binary)}]
    return write_glb(j, binary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--max-texture', type=int, default=1024); ap.add_argument('--quality', type=int, default=84)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, uid in MODELS.items():
        src = a.src / f'{uid}.glb'
        if not src.exists():
            print(f'skip {name}: {src} missing'); continue
        data = pack(src.read_bytes(), a.max_texture, a.quality)
        (a.out / f'{name}.glb').write_bytes(data)
        print(f'{name:9s} {src.stat().st_size / 1e6:6.2f} MB -> {len(data) / 1e6:5.2f} MB')


if __name__ == '__main__':
    main()
