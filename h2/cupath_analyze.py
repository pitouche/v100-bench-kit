#!/usr/bin/env python3
"""cupath_analyze.py — re-verification independante de la trace CUPTI 12.9 (run6).

Usage : python3 cupath_analyze.py [trace.names] [wall_eval_ms]
- Reconstitue la fenetre decode a partir de la fin de trace et du wall (ms) du log serveur.
- Recompte : nb records, fenetre, streams, familles, variantes mul_mat_vec_q,
  union des intervalles (chevauchement), coherent comptages / lancements / durees.
Aucun effet de cote : lecture seule du fichier .names.
"""
import re, sys, subprocess
from collections import defaultdict, Counter

NAMES = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cupath_llama_144318.names"
WALL_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 1576.52  # eval time task 236 (log serveur)

pat = re.compile(r"^(KER|CPY) t=(\d+) stream=(\d+) dur_ns=(\d+) name=(.*)$")
rows = []
with open(NAMES) as f:
    for line in f:
        m = pat.match(line.rstrip("\n"))
        if not m:
            continue
        kind, t, st, d, n = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
        rows.append((kind, t, st, d, n))

n_ker_all = sum(1 for r in rows if r[0] == "KER")
n_cpy_all = sum(1 for r in rows if r[0] == "CPY")
print(f"=== GLOBAL (processus entier, depuis le LD_PRELOAD) ===")
print(f"lignes parsees : {len(rows)}  (KER={n_ker_all}, CPY={n_cpy_all})")
print(f"t min={rows[0][1]}  t max={max(r[1] for r in rows)}  (us, CLOCK_REALTIME)  [dur en ns]", )

# --- reconstitution de la fenetre decode (task 236) ---
ker = [r for r in rows if r[0] == "KER"]
ker.sort(key=lambda r: r[1])
t_end = max(r[1] + r[3] // 1000 for r in ker)  # fin du dernier kernel (us ; dur_ns converti)
t0 = t_end - int(WALL_MS * 1000)  # us
win = [r for r in rows if t0 - 200_000 <= r[1] <= t_end + 200_000]
wk = [r for r in win if r[0] == "KER"]
wc = [r for r in win if r[0] == "CPY"]
print(f"\n=== FENETRE DECODE task 236 ===")
spank = (max(r[1]+r[3]//1000 for r in wk)-min(r[1] for r in wk))/1e3
print(f"fenetre : [{t0}, {t_end}] us  (largeur tracee {spank:.1f} ms)")
print(f"kernels : {len(wk)}   memcpy : {len(wc)}")

# premier kernel avant t0 (contexte)
pre = [r for r in ker if r[1] < t0][-3:]
for r in pre:
    print(f"  pre-t0 : t={r[1]} (+{(r[1]-t0)/1e3:+.3f} s) stream={r[2]} dur={r[3]/1e3:.1f}us {r[4][:60]}")

tot = sum(r[3] for r in wk)
print(f"somme durees kernels (toutes streams) : {tot/1e6:.3f} ms  | wall log serveur : {WALL_MS} ms")
print(f"ratio somme/wall : {tot/1e6/WALL_MS*100:.2f} %")

# --- par stream ---
st = defaultdict(lambda: [0, 0])
for r in wk:
    st[r[2]][0] += r[3]; st[r[2]][1] += 1
print(f"\n=== PAR STREAM (fenetre) ===")
for s, (d, c) in sorted(st.items(), key=lambda x: -x[1][0]):
    print(f"  stream {s:<3d} : {c:6d} launches  {d/1e6:9.3f} ms  ({d*100.0/tot:5.2f} % du cumule)")

# --- union des intervalles (busy reel, chevauchements) ---
iv = sorted((r[1], r[1] + r[3]//1000) for r in wk)
union = 0; cur_s = cur_e = None
for s, e in iv:
    if cur_e is None or s > cur_e:
        if cur_e is not None: union += cur_e - cur_s
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
union += cur_e - cur_s
span = max(r[1]+r[3]//1000 for r in wk) - min(r[1] for r in wk)
print(f"\n=== CHEVAUCEMENT ===")
print(f"union intervalles (busy reel) : {union/1e6:.3f} ms  | span {span/1e6:.3f} ms")
print(f"busy reel / wall : {union/1e6/WALL_MS*100:.2f} %  | double-compte (somme-union) : {(tot-union)/1e6:.3f} ms")
# gaps > 200 us
gaps = []
for i in range(1, len(iv)):
    g = iv[i][0] - iv[i-1][1]
    if g > 200_000: gaps.append(g)  # > 200 ms
gaps.sort(reverse=True)
print(f"gaps > 200us : {len(gaps)} ; top : {['%.1fus'%(g/1e3) for g in gaps[:8]]}")

# --- familles (normalisation identique au script run_cupath_llama.sh) ---
def base(name):
    return re.sub(r"_[0-9]+_", "_", name.split("(")[0])[:78]
fam = defaultdict(lambda: [0, 0])
for r in wk:
    fam[base(r[4])][0] += r[3]; fam[base(r[4])][1] += 1
print(f"\n=== TOP 20 FAMILLES (fenetre, normalisation base) ===")
for k, (d, c) in sorted(fam.items(), key=lambda x: -x[1][0])[:20]:
    print(f"  {d/1e6:8.2f} ms  x{c:<6d} {d*100.0/tot:5.2f} %  {k}")

# --- variantes exactes mul_mat_vec_q (demanling) ---
mmv = [r for r in wk if r[4].startswith("_Z13mul_mat_vec_q") or "mul_mat_vec_q" in r[4]]
def demangle(name):
    p = subprocess.run(["c++filt", name], capture_output=True, text=True)
    return p.stdout.strip() or name
cnt = Counter()
dur = defaultdict(int)
for r in mmv:
    cnt[r[4]] += 1
    dur[r[4]] += r[3]
print(f"\n=== mul_mat_vec_q (exact) ===")
print(f"total : {sum(cnt.values())} launches, {sum(dur.values())/1e6:.2f} ms ({sum(dur.values())*100.0/tot:.2f} % du cumule)")
for n, c in cnt.most_common(12):
    dm = demangle(n)
    print(f"  x{c:<6d} {dur[n]/1e6:9.2f} ms  mean={dur[n]/c/1e3:7.1f} us  {dm[:150]}")

# --- ggml_q8v14partial + quantize + flash_attn + gated_delta_net : detail ---
for key in ["ggml_q8v14partial", "flash_attn_ext_vec", "quantize_q8_1", "gated_delta_net", "mul_mat_vec_q_mapped_rows"]:
    sub = [r for r in wk if key in r[4]]
    if not sub:
        print(f"\n[{key}] absent de la fenetre")
        continue
    d = sum(r[3] for r in sub)
    ss = Counter(r[2] for r in sub)
    durs = sorted(r[3] for r in sub)
    import statistics as S
    print(f"\n[{key}] : {len(sub)} launches, {d/1e6:.2f} ms ({d*100.0/tot:.2f} %), streams={dict(ss)}, "
          f"mean={S.mean(durs)/1e3:.1f}us med={S.median(durs)/1e3:.1f}us max={max(durs)/1e3:.1f}us min={min(durs)/1e3:.2f}us")

# --- parite lancements/round ---
ROUNDS = 64 / 3.71
print(f"\n=== PARITE ===")
print(f"rounds (64 tok / mean_len 3.71) = {ROUNDS:.2f}")
print(f"kernels/token = {len(wk)/64:.1f}   kernels/round = {len(wk)/ROUNDS:.0f}")
for k in list(sorted(fam.items(), key=lambda x: -x[1][0])[:6]):
    nm, (d, c) = k
    print(f"  {nm[:60]:60s} {c/ROUNDS:9.0f} lances/rd  {c/64:9.1f} lances/tok")
