#!/usr/bin/env python3
"""e3_cluster.py — analyse de la fenetre decode run7 v2 (cluster final apres dernier gap>30ms).
Usage: python3 e3_cluster.py <prefix> <wall_ms>
Sortie: gates + union/occupation/idle/overlap + familles + streams + gaps.
"""
import re, sys
from collections import defaultdict, Counter
prefix, WALL_MS = sys.argv[1], float(sys.argv[2])
WALL = int(WALL_MS * 1e6)  # ns
pat = re.compile(r"^(KER|CPY) seq=(\d+) gpu_s=(\d+) gpu_e=(\d+) dur_ns=(\d+) stream=(\d+) corr=(\d+) t_real_us=(\d+) name=(.*)$")
rows = []
with open(prefix + ".names") as f:
    for line in f:
        m = pat.match(line.rstrip("\n"))
        if m: rows.append((int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)), m.group(9), m.group(1)))
rows.sort(key=lambda r: r[0])
ker = [r for r in rows if r[5] == "KER"]
last_rel = ker[-1][0]
seg = [r for r in ker if r[0] > last_rel - 60_000_000_000]
gaps = []
for i in range(1, len(seg)):
    g = seg[i][0] - seg[i-1][1]
    if g > 30_000_000: gaps.append((i, g))
sb = gaps[-1][0]
dec = seg[sb:]
s0, e1 = dec[0][0], dec[-1][1]
span = e1 - s0
tot = sum(e - s for s, e, d, st, nm, k in dec)
def union_iv(iv):
    iv = sorted(iv); u = 0; cs = ce = None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None: u += ce - cs
            cs, ce = s, e
        else: ce = max(ce, e)
    if ce is not None: u += ce - cs
    return u
u = union_iv([(s, e) for s, e, d, st, nm, k in dec])
print(f"=== FENETRE DECODE run7 v2 (cluster apres dernier gap>30ms) ===")
print(f"n kernels : {len(dec)}   span trace : {span/1e6:.1f} ms   wall serveur : {WALL_MS:.1f} ms")
print(f"somme kernels : {tot/1e6:.1f} ms")
print(f"UNION        : {u/1e6:.1f} ms")
print(f"occupation union/span : {u/span*100:.2f} %")
print(f"occupation union/wall : {u/WALL*100:.2f} %   G5 ecart = {abs(WALL-u)/WALL*100:.2f} % -> {'PASS' if abs(WALL-u)/WALL*100 < 2 else 'FAIL'}")
print(f"chevauchement (somme-union) : {(tot-u)/1e6:.3f} ms")
print(f"idle dans le span : {(span-u)/1e6:.1f} ms ({(span-u)/span*100:.2f} %)")
print(f"premiere part du cluster avant debut eval (span-wall) : {max(0, span-WALL)/1e6:.1f} ms")
# gaps internes
ivs = sorted((s, e) for s, e, d, st, nm, k in dec)
gs = []
for i in range(1, len(ivs)):
    g = ivs[i][0] - ivs[i-1][1]
    if g > 0: gs.append(g)
big = [g for g in gs if g > 200_000]
print(f"gaps internes >200us : {len(big)} ; somme gaps = {sum(gs)/1e6:.1f} ms ; top: {['%.2f ms'%(g/1e6) for g in sorted(gs, reverse=True)[:8]]}")
# memcpy
cp = [r for r in rows if r[5] == "CPY" and s0 <= r[0] < e1]
print(f"memcpy dans la fenetre : {len(cp)} ; somme : {sum(r[2] for r in cp)/1e3:.1f} us")
# streams
st = Counter()
std = defaultdict(int)
for s, e, d, stm, nm, k in dec:
    st[stm] += 1; std[stm] += d
print("streams :", {k: f"{v} ker" for k, v in st.most_common(6)})
print("temps par stream :", {k: f"{v/1e6:.1f} ms" for k, v in sorted(std.items(), key=lambda x: -x[1])[:6]})
# familles
def base(nm): return re.sub(r"_[0-9]+_", "_", nm.split("(")[0])[:78]
fam = defaultdict(lambda: [0, 0])
for s, e, d, stm, nm, k in dec:
    fam[base(nm)][0] += d; fam[base(nm)][1] += 1
print(f"\nTOP 15 familles :")
for k, (d, c) in sorted(fam.items(), key=lambda x: -x[1][0])[:15]:
    print(f"  {d/1e6:8.2f} ms  x{c:<6d} {d*100.0/tot:5.1f} %  {k}")
print(f"\nrates : kernels/tok = {len(dec)/64:.0f} ; wall/tok = {WALL/64/1e3:.2f} ms ; wall/round(3.71) = {WALL*3.71/64/1e3:.2f} ms")
