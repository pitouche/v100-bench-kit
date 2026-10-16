#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RUN8 — diagnostic + analyse fenêtre decode (script dédié).
Le bracket (us, CLOCK_REALTIME) + le domaine CUPTI v2 = REALTIME en ns (G4).
Note : cupath_analyze_v2.py a un bug d'unité (union en ns vs wall_us en µs)
dans le calcul de l'occupation G5 → ici on calcule correctement en ns."""
import re, collections, sys

NAMES = "/tmp/cupath8_230436.667897.names"
TS0_US = 1792019195127885   # bracket départ (us)
TS1_US = 1792019198395716   # bracket fin   (us)
WALL_MS = 1092.97           # eval time serveur (ms)

pat = re.compile(r"^(KER|CPY) seq=(\d+) gpu_s=(\d+) gpu_e=(\d+) dur_ns=(\d+) "
                 r"stream=(\d+) corr=(\d+) t_real_us=(\d+) name=(.*)$")

TS0 = TS0_US * 1000  # ns
TS1 = TS1_US * 1000  # ns

def classify(nm):
    if "q8v14partial" in nm: return "ggml_q8v14partial"
    if "mul_mat_vec_q_mapped_rows" in nm: return "mul_mat_vec_q_mapped_rows"
    if "mul_mat_vec_q" in nm: return "mul_mat_vec_q (gemv)"
    if "mul_mat_q" in nm: return "mul_mat_q (mm/prefill)"
    if "flash_attn" in nm: return "flash_attn"
    if "k_set_rows" in nm: return "k_set_rows"
    if "k_get_rows" in nm: return "k_get_rows"
    if "quantize_q8_1" in nm: return "quantize_q8_1"
    if "gated_delta_net" in nm: return "gated_delta_net"
    if "cutlass" in nm: return "cutlass gemm (draft/fa)"
    if "convert_unary" in nm: return "convert_unary"
    if "dequantize" in nm: return "dequantize"
    if "cpy_scalar" in nm: return "cpy_scalar"
    return "autres"

ker = []
cpys = 0
for l in open(NAMES):
    m = pat.match(l.rstrip("\n"))
    if not m:
        continue
    s = int(m.group(3))
    if not (TS0 <= s < TS1):
        continue
    if m.group(1) == "KER":
        ker.append((int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(9)))
    elif m.group(1) == "CPY":
        cpys += 1

tot = sum(d for s, e, d, nm in ker)
fam = collections.defaultdict(lambda: [0, 0])
for s, e, d, nm in ker:
    fam[classify(nm)][0] += d
    fam[classify(nm)][1] += 1

# union
ivs = sorted((s, e) for s, e, _, _ in ker)
union = 0; cs = ce = None
for s, e in ivs:
    if ce is None or s > ce:
        if ce is not None: union += ce - cs
        cs, ce = s, e
    else:
        ce = max(ce, e)
if ce is not None: union += ce - cs

print(f"=== RUN8 fenêtre decode (lib H2) ===")
print(f"bracket curl : {(TS1_US-TS0_US)/1e6:.3f} s | wall serveur (eval) : {WALL_MS} ms")
print(f"kernels fenêtre : {len(ker)} | memcpy : {cpys}")
print(f"somme durées kernels : {tot/1e6:.2f} ms")
print(f"UNION des intervalles : {union/1e6:.2f} ms")
print(f"occupation (union/wall) : {union/1e6/WALL_MS*100:.2f} %   G5 écart = {abs(WALL_MS-union/1e6)/WALL_MS*100:.2f} %  (PASS si <2)")
print(f"chevauchement (somme-union) : {(tot-union)/1e6:.3f} ms")
print()
print("TOP familles (durée totale, fenêtre) :")
for k, (d, c) in sorted(fam.items(), key=lambda x: -x[1][0])[:14]:
    print(f"  {d/1e6:8.2f} ms  x{c:<6d} {d*100.0/max(tot,1):5.1f} %  {k}")

q = sorted(d for s, e, d, nm in ker if "q8v14partial" in nm)
if q:
    print(f"\nq8v14partial : {len(q)} lances, total {sum(q)/1e6:.2f} ms, "
          f"médiane {q[len(q)//2]/1e3:.1f} µs/lance, max {q[-1]/1e3:.1f} µs")
    # combien de rounds ? (64 tok, mean_len ~3 -> ~17 rounds ; E3 : 17 lances/rd)
    print(f"  → ~{len(q)/17:.1f} lances/round (si ~17 rounds de decode effectifs)")

gaps = [ivs[i][0] - ivs[i-1][1] for i in range(1, len(ivs))]
big = [g for g in gaps if g > 200000]
print(f"\ngaps > 200 µs : {len(big)} ; top : {[round(g/1e6,1) for g in sorted(big, reverse=True)[:8]]} ms")
print(f"gaps médian : {sorted(gaps)[len(gaps)//2]/1e3:.2f} µs ; nb gaps : {len(gaps)}")

# check : y a-t-il des noms de kernel tronqués (1-2 chars) dans la fenêtre ?
short = [nm for s, e, d, nm in ker if len(nm.split('(')[0]) <= 3]
print(f"\n[nom court ≤3 chars dans la fenêtre] : {len(short)}  (ex : {short[:5]})")
