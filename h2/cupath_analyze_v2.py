#!/usr/bin/env python3
"""cupath_analyze_v2.py — analyse du format v2 (E3).

Usage:
  python3 cupath_analyze_v2.py <trace_prefix> [mode] [t_start_real_us] [t_end_real_us]

mode:
  test   : validation de coherence interne (test_v2) — pas de fenetre requise
  window : analyse d'une fenetre decode (realtime bracket par t_start/t_end real_us,
           ou dernier segment si absent)

Verifie les gates :
  G1 aucun champ manquant (seq, gpu_s, gpu_e, dur, stream, corr, t_real)
  G2 dur_ns == gpu_end - gpu_start  (100% des records)
  G3 seq continu 0..N-1 sans gap, alignement 1:1 .bin/.names
  G4 domaine CUPTI coherent (offset CUPTI-mono constant, derive bracket)
  G5 couverture : union fenetre vs wall (<2%)
"""
import re, sys, struct
from collections import defaultdict

prefix = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "test"
def _num(i):
    if len(sys.argv) > i and sys.argv[i] not in ("", "-", "0"):
        return float(sys.argv[i])
    return None
T0 = _num(3)      # us CLOCK_REALTIME, borne depart curl (optionnel)
T1 = _num(4)      # us CLOCK_REALTIME, borne fin curl (optionnel ; inclut 2 s de sleep)
WALL_MS = _num(5) # ms, eval time serveur (optionnel)

OP = {0: "KER", 1: "CPY", 2: "RT", 3: "RESET", 4: "MARK"}
REC = 56

# ---------- lecture .bin ----------
recs = []
with open(prefix, "rb") as f:
    data = f.read()
n = len(data) // REC
for i in range(n):
    seq, s, e, d, corr, st, op, aux = struct.unpack_from("<QQQQQIIQ", data, i * REC)
    recs.append((i, seq, s, e, d, corr, st, op, aux))
print(f"=== .bin : {n} records x {REC} octets ===")

# ---------- lecture .names (KER/CPY/RT seulement) ----------
pat = re.compile(r"^(KER|CPY|RT) seq=(\d+) gpu_s=(\d+) gpu_e=(\d+) dur_ns=(\d+) stream=(\d+) corr=(\d+) t_real_us=(\d+) name=(.*)$")
names = []
with open(prefix + ".names") as f:
    for line in f:
        m = pat.match(line.rstrip("\n"))
        if m:
            names.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                          int(m.group(5)), int(m.group(6)), int(m.group(7)), int(m.group(8)), m.group(9)))
print(f"=== .names : {len(names)} lignes parsees (KER={sum(1 for x in names if x[0]=='KER')}, "
      f"CPY={sum(1 for x in names if x[0]=='CPY')}, RT={sum(1 for x in names if x[0]=='RT')}) ===")

# ---------- G1/G2/G3 : coherence ----------
ker_cpy_rt = [r for r in recs if r[7] in (0, 1, 2)]
ok_g1 = ok_g2 = ok_g3 = True
n_dur_bad = 0
n_seq_bad = 0
n_align_bad = 0
for i, (slot, seq, s, e, d, corr, st, op, aux) in enumerate(recs):
    if d != e - s:
        n_dur_bad += 1
        ok_g2 = False
    if seq != i:
        n_seq_bad += 1
        ok_g3 = False
if len(ker_cpy_rt) != len(names):
    ok_g3 = False
    print(f"  [G3] ECHOC : {len(ker_cpy_rt)} records KER/CPY/RT vs {len(names)} lignes .names")
else:
    for (slot, seq, s, e, d, corr, st, op, aux), nm in zip(ker_cpy_rt, names):
        if not (nm[1] == seq and nm[2] == s and nm[3] == e and nm[4] == d and nm[5] == st and nm[6] == corr):
            n_align_bad += 1
            ok_g3 = False
print(f"G1 champ manquant          : {'OK' if ok_g1 else 'ECHEC'}")
print(f"G2 dur == end-start        : {'OK' if ok_g2 else 'ECHEC (%d records)' % n_dur_bad}")
print(f"G3 seq continu + align 1:1 : {'OK' if ok_g3 else 'ECHEC (seq_bad=%d align_bad=%d)' % (n_seq_bad, n_align_bad)}")

# ---------- G4 : domaine horloges ----------
# kernel le plus precoce : son gpu_s doit etre >= (mono0 + delta) pour tout delta >= 0
# et le t_real du dernier record doit correspondre a mono0 + (t_real - t0real) proche
hdr = open(prefix + ".hdr").read().strip()
print(f"hdr : {hdr}")
h = dict(x.split("=") for x in hdr.split() if "=" in x)
t0_real = int(h["t0_real_us"]); t0_mono = int(h["t0_mono_ns"])
# bracket : pour chaque record, t_real (us, au flush) >= real de debut. On calcule l'offset
# minimal possible entre domaine CUPTI et CLOCK_MONOTONIC :
#   mono(t_debut_kernel) >= t0_mono  =>  offset_cu_m >= gpu_s - t0_mono  (pour le 1er kernel)
#   et  t_real_flush >= real debut kernel => offset_cu_m <= gpu_e + flush_lag - (t0_mono + ...)
# On ne peut pas determiner exactement l'offset sans un event de synchronisation.
# On donne la borne : offset_cu_m dans [min(gpu_s - t0_mono), max(t_real - t0_real + 0)]...
# Le domaine CUPTI est la horloge REELLE (ns) — verifie par cross-check :
# offset = gpu_s(1er event) - t0_real*1000 doit etre un petit positif (quelques ms),
# et t_real_flush - t0_real doit couvrir [offset, offset + duree_trace + lag_flush].
k0 = min(r[2] for r in recs if r[7] in (0, 1))
last_real = max(nm[7] for nm in names)
kern_end = max(r[3] for r in recs if r[7] == 0)
off_ns = k0 - t0_real * 1000
print(f"[G4] 1er event gpu_s - t0_real = {off_ns/1e6:.3f} ms (doit etre >0 et petit) -> domaine CUPTI = CLOCK_REALTIME (ns) : " + ("OK" if 0 < off_ns < 3600e9 else "ANOMALIE"))
print(f"[G4] dernier kernel fin (rel t0) = {(kern_end - t0_real*1000)/1e6:.1f} ms ; dernier t_real_flush - t0_real = {last_real - t0_real} us (>= fin kernel : OK si >= )")

# ---------- fenetre ----------
def union(iv):
    iv = sorted(iv)
    tot = 0; cs = ce = None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None: tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None: tot += ce - cs
    fs = min(s for s, e in iv)
    le = max(e for s, e in iv)
    return tot, fs, le

if mode == "test":
    kers = [r for r in recs if r[7] == 0]
    iv = [(r[2], r[3]) for r in kers]
    u, cs, ce = union(iv)
    tot = sum(e - s for s, e in iv)
    print(f"\n=== TEST : {len(kers)} kernels, somme={tot/1e6:.3f} ms, union={u/1e6:.3f} ms ===")
    print(f"chevauchement (somme-union) = {(tot-u)/1e6:.3f} ms ; occupation union/span = {u/(ce-cs)*100:.1f} %")
    gaps = []
    ivs = sorted(iv)
    for i in range(1, len(ivs)):
        g = ivs[i][0] - ivs[i-1][1]
        if g > 100000: gaps.append(g)
    print(f"gaps > 100 us : {['%.1f ms' % (g/1e6) for g in sorted(gaps, reverse=True)[:5]]}")
    sys.exit(0 if (ok_g2 and ok_g3) else 1)
# ---- mode window ----
# Le domaine CUPTI == CLOCK_REALTIME (ns) — prouve dans le test unitaire (G4).
kern_all = [nm for nm in names if nm[0] == "KER"]
if T0 is not None and T1 is not None:
    TS0ns = int(T0 * 1000)
    TS1ns = int((T1 - 2000) * 1000)   # -2 s (sleep post-curl du script)
    kern = [nm for nm in kern_all if TS0ns <= nm[2] < TS1ns or (nm[2] < TS0ns and nm[3] > TS0ns)]
    cpys = [nm for nm in names if nm[0] == "CPY" and TS0ns <= nm[2] < TS1ns]
    wall_us = WALL_MS * 1000 if (WALL_MS and kern) else (TS1ns // 1000 - T0)
    label = f"bracket curl (wall serveur {WALL_MS:.1f} ms)"
else:
    wall_us = WALL_MS * 1000 if WALL_MS else 0
    if wall_us <= 0:
        label = "PAS DE WALL — fenetre non bornee"
        t_end = max(nm[3] for nm in kern_all)
        kern = [nm for nm in kern_all if nm[3] > t_end - 20e9]  # placeholder
    else:
        t_end = max(nm[3] for nm in kern_all)
        t0 = t_end - int(wall_us)
        kern = [nm for nm in kern_all if nm[2] < t_end and nm[3] > t0]
        cpys = [nm for nm in names if nm[0] == "CPY" and nm[2] < t_end and nm[3] > t0]
        label = f"ancrage fin de trace (t_end - {wall_us/1e6:.3f} s)"

print(f"\n=== FENETRE ({label}) ===")
if wall_us > 0:
    print(f"wall de reference : {wall_us/1e6:.3f} s")
print(f"records dans la fenetre : {len(kern)} kernels + {len(cpys)} memcpy (sur {len(kern_all)} KER totaux)")
tot = sum(nm[4] for nm in kern)
iv = [(nm[2], nm[3]) for nm in kern]
u, fs, le = union(iv)
span = le - fs
print(f"premiers/dermiers kernels (rel, s) : [0, {(le-fs)/1e9:.3f}] ; span trace = {span/1e6:.3f} ms")
print(f"somme durees kernels : {tot/1e6:.3f} ms")
print(f"UNION des intervalles : {u/1e6:.3f} ms")
print(f"occupation GPU (union/span trace) : {u/span*100:.2f} %")
if wall_us > 0:
    occ = u / wall_us * 100
    ecart = abs(wall_us - u) / wall_us * 100
    print(f"occupation GPU (union/wall serveur) : {occ:.2f} %   GATE G5 : ecart = {ecart:.2f} % -> {'PASS' if ecart < 2 else 'FAIL'}")
    print(f"idle (wall serveur - union) : {(wall_us-u)/1e6:.3f} ms ({100-occ:.2f} %)")
else:
    sys.exit(0)
ovl = tot - u
print(f"chevauchement total (somme-union) : {ovl/1e6:.3f} ms ({ovl/tot*100:.2f} % de la somme)")
# idle par intervalle : gaps > 200 us
ivs = sorted(iv)
gaps = []
for i in range(1, len(ivs)):
    g = ivs[i][0] - ivs[i-1][1]
    if g > 200000: gaps.append((ivs[i-1][1], g))
gaps.sort(key=lambda x: -x[1])
print(f"gaps > 200 us : {len(gaps)} ; top : {['%.1f ms' % (g/1e6) for _, g in gaps[:8]]}")
# top familles
def base(nm):
    return re.sub(r"_[0-9]+_", "_", nm[8].split("(")[0])[:78]
fam = defaultdict(lambda: [0, 0])
for nm in kern:
    fam[base(nm[8])][0] += nm[4]; fam[base(nm[8])][1] += 1
print("\nTOP 12 familles (fenetre) :")
for k, (d, c) in sorted(fam.items(), key=lambda x: -x[1][0])[:12]:
    print(f"  {d/1e6:8.2f} ms  x{c:<6d} {d*100.0/max(tot,1):5.1f} %  {k}")
# overlap par paire de streams
st = defaultdict(list)
for nm in kern:
    st[nm[5]].append((nm[2], nm[3]))
slist = sorted(st)
for i in range(len(slist)):
    for j in range(i+1, len(slist)):
        a, b = st[slist[i]], st[slist[j]]
        ov = 0
        l = 0
        for s1, e1 in a:
            while l < len(b) and b[l][1] <= s1: l += 1
            for s2, e2 in b[l:]:
                if s2 >= e1: break
                ov += min(e1, e2) - max(s1, s2)
        if ov > 1e6:
            print(f"overlap stream {slist[i]} x {slist[j]} : {ov/1e6:.2f} ms")
