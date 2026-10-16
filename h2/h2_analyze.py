#!/usr/bin/env python3
# =============================================================================
# h2_analyze.py — analyse le sqlite exporté par nsys (fiche H2) et produit:
#   rapport-h2.md (dans le dir de run) + tables csv
#
# Usage: python3 h2_analyze.py <run-dir>
#   <run-dir>/prof.sqlite   export nsys --type=sqlite
#   <run-dir>/prof.nsys-rep (fallback: le re-exporte)
#
# Mesures (protocole h2/PROTOCOLE.md):
#   GPU : fenetre, temps occupe, occupation, familles de kernels, gaps
#         (mediane/p90/p95/max, top-20), classification des gaps (CPU bloque
#         vs CPU en cours) via OSRT.
#   CPU : API CUDA (graph launch / syncs), OSRT par type, thread d'orchestration.
#   MTP : cycles (rounds) par bursts GPU (1 round = verify + 3 drafts),
#         wall par round, intra-phase vs inter-phase gaps, cross-check
#         CUPTI_ACTIVITY_KIND_GRAPH_TRACE si present.
# =============================================================================
import csv, os, re, sqlite3, statistics, sys
from bisect import bisect_left, bisect_right

NSYS = "/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys"

# ---------------------------------------------------------------- helpers
def colnames(con, table):
    try:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()

def pick(cols, *cands):
    for c in cands:
        if c in cols:
            return c
    return None

def find_table(con, *names):
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for n in names:
        if n in have:
            return n
    return None

def q1(con, sql, args=()):
    try:
        r = con.execute(sql, args).fetchone()
        return r[0] if r else None
    except sqlite3.Error:
        return None

def load_rows(con, table, cols_needed, name_col=None):
    """lit une table en dict, avec colonnes tolerantes"""
    cols = colnames(con, table)
    out = []
    sel = ["start", "end"]
    for c in cols_needed:
        if c in cols:
            sel.append(c)
    if name_col:
        for cand in name_col:
            if cand in cols:
                sel.append(cand)
                break
    try:
        for row in con.execute(f"SELECT {', '.join(sel)} FROM {table} ORDER BY start"):
            out.append(dict(zip(sel, row)))
    except sqlite3.Error:
        pass
    return out

# familles de kernels (ordre = priorite)
FAMILIES = [
    ("poids_mmvq",      re.compile(r"mul_mat_vec_q|vec_dot_q", re.I)),
    ("poids_mmq",       re.compile(r"mul_mat_q|mul_mat_id|dp4a|mmq", re.I)),
    ("cublas",          re.compile(r"cublas|nvjet|gemm|sgemm|hgemm", re.I)),
    ("attention",       re.compile(r"flash_attn|fattn|fmha|fmha_ext|softmax|softcap", re.I)),
    ("gdn_ssm",         re.compile(r"gated_delta_net|gdn|scan|ssm", re.I)),
    ("requant_quant",   re.compile(r"quantize|requant", re.I)),
    ("sampling_mtp",    re.compile(r"sample|top_k|min_p|shortlist|proposal|mtp", re.I)),
    ("rope_norm",       re.compile(r"rope|layrn|norm|rms", re.I)),
    ("elementwise",     re.compile(r"cadd|cdiv|cmul|cmax|cmin|nary|unary|contig|t_vec|perp_|set_|fill_|scale|cumsum|argmax|acc_|copy|csum|repeat|concat|diagonal|transpose|upcast|downcast|view|reshape|pad|sum_rows", re.I)),
]

def fam_of(name):
    for label, rx in FAMILIES:
        if rx.search(name or ""):
            return label
    return "autres"

SLEEP_TYPES = {"usleep", "nanosleep", "clock_nanosleep", "poll", "epoll_wait"}
FUTEX_TYPES = {"futex"}

# ---------------------------------------------------------------- GPU intervals
def gpu_intervals(con):
    """retourne (list of (start_ns, end_ns, kind, name, stream)) + infos"""
    t = find_table(con, "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
                   "CUPTI_ACTIVITY_KIND_KERNEL")
    kernels = []
    if t:
        rows = load_rows(con, t, [], name_col=("shortName", "demangledName", "name", "mangledName"))
        for r in rows:
            kernels.append((r["start"], r["end"], "kernel",
                            r.get("shortName") or r.get("demangledName") or r.get("name") or "?",
                            r.get("streamId")))
    t = find_table(con, "CUPTI_ACTIVITY_KIND_MEMCPY")
    mems = []
    if t:
        for r in load_rows(con, t, []):
            mems.append((r["start"], r["end"], "memcpy", "memcpy", r.get("streamId")))
    t = find_table(con, "CUPTI_ACTIVITY_KIND_MEMSET")
    if t:
        for r in load_rows(con, t, []):
            mems.append((r["start"], r["end"], "memset", "memset", r.get("streamId")))
    return kernels, mems

def merge_intervals(iv):
    """iv: list (s,e) -> merged sorted"""
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append([s, e])
    return out

def busy_time(merged):
    return sum(e - s for s, e in merged)

# ---------------------------------------------------------------- main
def main():
    rundir = sys.argv[1] if len(sys.argv) > 1 else "."
    db = os.path.join(rundir, "prof.sqlite")
    if not os.path.exists(db):
        rep = os.path.join(rundir, "prof.nsys-rep")
        if not os.path.exists(rep):
            sys.exit("ni prof.sqlite ni prof.nsys-rep dans " + rundir)
        os.system(f"{NSYS} export --type=sqlite -f --output={db} {rep}")
    con = sqlite3.connect(db)

    kernels, mems = gpu_intervals(con)
    if not kernels and not mems:
        sys.exit("aucun intervalle GPU trouve dans le sqlite (tables: " +
                 ", ".join(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'"))[:400] + ")")

    iv = [(s, e) for s, e, *_ in kernels + mems]
    merged = merge_intervals(iv)
    span0, span1 = merged[0][0], merged[-1][1]
    span = span1 - span0
    busy = busy_time(merged)
    util = busy / span * 100 if span else 0.0

    # familles
    fam_t = {}
    fam_n = {}
    for s, e, kind, name, _ in kernels + mems:
        f = fam_of(name if kind == "kernel" else "memcpy/memset")
        dt = e - s
        fam_t[f] = fam_t.get(f, 0) + dt
        fam_n[f] = fam_n.get(f, 0) + 1

    # gaps (les gaps > 1 s = artefacts de bords de capture : comptes a part, hors stats)
    GAP_CAP_NS = 1_000_000_000
    gaps = []
    gap_artefacts = []
    for i in range(len(merged) - 1):
        g = merged[i + 1][0] - merged[i][1]
        if g > 0:
            (gap_artefacts if g > GAP_CAP_NS else gaps).append((merged[i][1], merged[i + 1][0], g))
    glen = [g for *_, g in gaps]
    gsum = sum(glen)

    # OSRT
    osrt = []
    t = find_table(con, "OSRT_RUNTIME")
    osrt_by_type = {}
    if t:
        for r in load_rows(con, t, ("type",)):
            osrt.append((r["start"], r["end"], r.get("type") or "?"))
            dt = r["end"] - r["start"]
            osrt_by_type[r.get("type") or "?"] = osrt_by_type.get(r.get("type") or "?", 0) + dt
    osrt_starts = [r[0] for r in osrt]
    osrt_ends = [r[1] for r in osrt]

    def gap_class(a, b):
        i, j = bisect_left(osrt_starts, b), bisect_left(osrt_ends, a)
        best = None
        for k in range(i, j):
            ty = osrt[k][2]
            if ty in FUTEX_TYPES:
                best = "cpu-bloque-futex"
                break
            if ty in SLEEP_TYPES:
                best = "cpu-bloque-sleep"
            elif best is None or best == "cpu-syscall":
                best = "cpu-syscall"
        return best or "cpu-user"

    cls_t = {}
    cls_n = {}
    for a, b, g in gaps:
        c = gap_class(a, b)
        cls_t[c] = cls_t.get(c, 0) + g
        cls_n[c] = cls_n.get(c, 0) + 1

    # API CUDA
    api = {}
    t = find_table(con, "CUPTI_ACTIVITY_KIND_RUNTIME")
    if t:
        for r in load_rows(con, t, [], name_col=("demangledName", "shortName", "name", "mangledName")):
            nm = r.get("demangledName") or r.get("shortName") or r.get("name") or "?"
            api[nm] = api.get(nm, [0, 0])
            api[nm][0] += 1
            api[nm][1] += r["end"] - r["start"]

    # graph trace
    graphs = []
    t = find_table(con, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
    if t:
        for r in load_rows(con, t, ("graphId", "launchId")):
            graphs.append((r["start"], r["end"], r.get("graphId"), r.get("launchId")))

    # MTP rounds — cascade:
    #  1) graphId present dans la table kernels (mode --cuda-graph-trace=graph)
    #  2) launches cudaGraphLaunch (table RUNTIME) -> 1 step/launch, round = 4 steps
    #  3) repli bursts (seuil 1 ms) -> 4 bursts/round (heuristique, avertisse)
    mtp_method = None
    rounds = []

    def busy_in(a, b):
        return busy_time(merge_intervals([(s, e) for s, e in merged if a <= s and e <= b]))

    # -- 1) graphId sur les kernels
    t = find_table(con, "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL", "CUPTI_ACTIVITY_KIND_KERNEL")
    kcols = colnames(con, t) if t else set()
    if "graphId" in kcols:
        per_graph = {}
        for r in load_rows(con, t, ("graphId",)):
            g = r.get("graphId")
            if g is None:
                continue
            per_graph.setdefault(g, []).append((r["start"], r["end"]))
        graphs_ordered = sorted(per_graph.items(), key=lambda kv: min(s for s, _ in kv[1]))
        if len(graphs_ordered) >= 8:
            mtp_method = "graphId (1 graph = 1 step, round = 4 steps)"
            for k in range(0, len(graphs_ordered) - 3, 4):
                grp = graphs_ordered[k:k + 4]
                s0 = min(min(s for s, _ in v) for _, v in grp)
                s1 = max(max(e for _, e in v) for _, v in grp)
                end = s1
                if k + 4 < len(graphs_ordered):
                    end = min(min(s for s, _ in v) for _, v in graphs_ordered[k + 4:k + 8])
                b = busy_time(merge_intervals([iv for _, v in grp for iv in v]))
                vbusy = [busy_time(merge_intervals(v)) for _, v in grp]
                vidx = max(range(4), key=lambda i: vbusy[i])
                rounds.append({"wall": end - s0, "busy": b, "verify": vbusy[vidx],
                               "drafts": sum(vbusy) - vbusy[vidx],
                               "intra": end - s0 - b, "inter": 0})

    # -- 2) launches cudaGraphLaunch dans RUNTIME
    trt = find_table(con, "CUPTI_ACTIVITY_KIND_RUNTIME")
    if not rounds and trt:
        launches = []
        for r in load_rows(con, trt, [], name_col=("demangledName", "shortName", "name", "mangledName")):
            nm = r.get("demangledName") or r.get("shortName") or r.get("name") or ""
            if "cudaGraphLaunch" in nm:
                launches.append(r["start"])
        launches = [l for l in launches if span0 <= l <= span1]
        if len(launches) >= 16:
            mtp_method = "cudaGraphLaunch RUNTIME (round = 4 launches consecutifs)"
            L = len(launches)
            def bnd(i): return launches[i] if i < L else span1
            for k in range(0, L - 3, 4):
                a, bndk4 = launches[k], bnd(k + 4)
                b = busy_time(merge_intervals([(s, e) for s, e in merged if a <= s < bndk4]))
                # verify = le step (entre 2 launches consecutifs) le plus busy
                vbusy = [busy_time(merge_intervals([(s, e) for s, e in merged if bnd(k + j) <= s < bnd(k + j + 1)]))
                         for j in range(4)]
                vidx = max(range(4), key=lambda i: vbusy[i])
                rounds.append({"wall": bndk4 - a, "busy": b, "verify": vbusy[vidx],
                               "drafts": sum(vbusy) - vbusy[vidx],
                               "intra": bndk4 - a - b, "inter": 0})

    # -- 3) repli bursts
    if not rounds:
        SPLIT = 1_000_000  # 1 ms
        bursts = []
        for s, e in merged:
            if bursts and s - bursts[-1][1] <= SPLIT:
                bursts[-1] = (bursts[-1][0], e)
            else:
                bursts.append([s, e])
        bursts = [tuple(b) for b in bursts]
        bdur = [e - s for s, e in bursts]
        med_burst = statistics.median(bdur) if bdur else 0
        if med_burst < 25_000_000:
            # les bursts sont a l'echelle d'un STEP (draft/verify) -> 4 steps = 1 round
            def score(offset):
                s = 0.0
                for i in range(offset, len(bursts) - 3, 4):
                    grp = bdur[i:i + 4]
                    rest = max(grp[1:])
                    s += (grp[0] - rest) - max(0.0, rest - grp[0]) * 2.0
                return s
            off = max(range(min(3, len(bursts))), key=score)
            for i in range(off, len(bursts) - 3, 4):
                g = bursts[i:i + 4]
                if len(g) < 4:
                    break
                vidx = max(range(4), key=lambda k: g[k][1] - g[k][0])
                rounds.append({"start": g[0][0], "end": g[3][1],
                               "wall": g[3][1] - g[0][0],
                               "verify": g[vidx][1] - g[vidx][0],
                               "drafts": sum(e - s for k, (s, e) in enumerate(g) if k != vidx),
                               "busy": sum(e - s for s, e in g),
                               "intra": sum(g[k + 1][0] - g[k][1] for k in range(3))})
            mtp_method = (f"repli bursts STEP (seuil 1 ms, 4 bursts/round) — HEURISTIQUE "
                          f"({len(bursts)} bursts, pas de graph trace)")
        else:
            # les bursts sont a l'echelle d'un ROUND (les 4 steps sont contigus < 1 ms)
            for s, e in bursts:
                busy = sum(es - ss for ss, es in merged if s <= ss and es <= e)
                rounds.append({"start": s, "end": e, "wall": e - s, "busy": busy,
                               "verify": None, "drafts": None,
                               "intra": (e - s) - busy})
            mtp_method = (f"repli bursts ROUND (1 burst = 1 round, split intra-step < 1 ms) — HEURISTIQUE, "
                          f"verify/drafts non separables ({len(bursts)} bursts)")
    if rounds:
        for k in range(len(rounds) - 1):
            if "start" in rounds[k + 1] and "end" in rounds[k]:
                rounds[k]["inter"] = rounds[k + 1]["start"] - rounds[k]["end"]
            elif "inter" not in rounds[k]:
                rounds[k]["inter"] = 0
        if "inter" not in rounds[-1]:
            rounds[-1]["inter"] = 0

    stable = rounds[len(rounds)//4:] if len(rounds) > 8 else rounds
    def med(key, rs=stable):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return statistics.median(vals) if vals else None
    def fmt_ms(v):
        return f"{v/1e6:,.1f}" if v is not None else "—"

    # ---------------------------------------------------------------- rapport
    L = []
    A = L.append
    A("# H2 — rapport de capture (decode MTP long contexte)")
    A("")
    A(f"Run : `{rundir}`")
    A(f"Kernels traces : **{len(kernels)}** (+ {len(mems)} memcpy/memset)")
    A(f"Fenetre GPU : **{span/1e6:,.1f} ms** · temps occupe : **{busy/1e6:,.1f} ms** · "
      f"**occupation GPU = {util:.1f} %**")
    A("")
    A("## 1. Temps GPU par famille")
    A("")
    A("| famille | ms | % du busy | nb | us moyen |")
    A("|---|---:|---:|---:|---:|")
    for f, dt in sorted(fam_t.items(), key=lambda x: -x[1]):
        A(f"| {f} | {dt/1e6:,.1f} | {dt/busy*100:.1f} % | {fam_n[f]} | {dt/fam_n[f]/1e3:.1f} |")
    A("")
    A("## 2. Gaps entre activités GPU")
    A("")
    ngap = len(gaps)
    p = sorted(glen)
    def pct(q):
        return p[min(len(p) - 1, int(q * len(p)))] / 1e6 if p else 0
    A(f"- nb de gaps : {ngap}"
      + (f"  (+ {len(gap_artefacts)} artefact(s) de bords > 1 s exclus : {sum(g for *_, g in gap_artefacts)/1e6:,.0f} ms)" if gap_artefacts else ""))
    A(f"- somme : **{gsum/1e6:,.1f} ms** ({gsum/span*100:.1f} % de la fenetre)")
    A(f"- mediane {statistics.median(glen)/1e3 if glen else 0:.2f} us · p90 {pct(0.90):.2f} ms · "
      f"p95 {pct(0.95):.2f} ms · max {max(glen)/1e6 if glen else 0:,.3f} ms")
    A("")
    A("### Classification des gaps (OSRT)")
    A("")
    A("| classe | ms | % des gaps | nb |")
    A("|---|---:|---:|---:|")
    for c, dt in sorted(cls_t.items(), key=lambda x: -x[1]):
        A(f"| {c} | {dt/1e6:,.1f} | {dt/gsum*100 if gsum else 0:.1f} % | {cls_n[c]} |")
    A("")
    A("Lecture : `cpu-bloque-futex` = au moins un thread bloque sur un futex dans le gap "
      "(sync GPU / lock) ; `cpu-user` = aucun appel bloqueant trace → le CPU exécute du code "
      "(orchestration) pendant le trou.")
    A("")
    A("### Top 20 gaps")
    A("")
    A("| # | start (ms, relatif) | durée (ms) | classe |")
    A("|---|---:|---:|---|")
    for rank, (a, b, g) in enumerate(sorted(gaps, key=lambda x: -x[2])[:20], 1):
        A(f"| {rank} | {(a-span0)/1e6:,.1f} | {g/1e6:,.3f} | {gap_class(a,b)} |")
    A("")
    A("## 3. CPU — API CUDA")
    A("")
    A("| API | nb | ms total |")
    A("|---|---:|---:|")
    for nm, (n, dt) in sorted(api.items(), key=lambda x: -x[1][1])[:20]:
        A(f"| {nm} | {n} | {dt/1e6:,.1f} |")
    A("")
    if t:
        A("### OSRT par type (top 10)")
        A("")
        A("| type | ms total |")
        A("|---|---:|")
        for ty, dt in sorted(osrt_by_type.items(), key=lambda x: -x[1])[:10]:
            A(f"| {ty} | {dt/1e6:,.1f} |")
        A("")
    A(f"## 4. Cycles MTP — {mtp_method or 'non identifiés'}")
    A("")
    if not rounds:
        A("")
        A("Aucun cycle MTP identifiable (pas de graphId, pas de cudaGraphLaunch, pas de bursts).")
        A("")
    if rounds:
        A("")
        A(f"Round (mediane, rounds stables) : wall **{fmt_ms(med('wall'))} ms** · "
          f"verify {fmt_ms(med('verify'))} ms · drafts {fmt_ms(med('drafts'))} ms · "
          f"busy GPU {fmt_ms(med('busy'))} ms · gaps intra-round {fmt_ms(med('intra'))} ms · "
          f"gap inter-round {fmt_ms(med('inter'))} ms")
        A("")
        A("Top 10 rounds les plus longs (anomalies) :")
        A("")
        A("| # | wall (ms) | busy (ms) | intra (ms) | inter (ms) |")
        A("|---|---:|---:|---:|---:|")
        for k, r in enumerate(sorted(rounds, key=lambda r: -r["wall"])[:10]):
            A(f"| {k+1} | {r['wall']/1e6:,.1f} | {r['busy']/1e6:,.1f} | {max(0.0, r['intra'])/1e6:,.2f} | {max(0.0, r.get('inter',0))/1e6:,.2f} |")
        if med("verify") is None:
            A("")
            A("> Mode bursts-ROUND : le split verify/drafts n'est pas resolvable a la resolution du seuil "
              "(4 steps contigus) — seul le wall du round est fiable.")
    A("")
    A("## 5. Décomposition du cycle")
    A("")
    if rounds:
        w = med("wall"); b = med("busy"); ig = med("intra"); inter = med("inter")
        if w is None:
            A("(cycles non mesurables)")
        else:
            A(f"```")
            A(f"Cycle MTP ≈ {w/1e6:,.0f} ms (médiane mesurée)")
            A(f"├── kernels GPU            {b/1e6:,.1f} ms   ({b/w*100 if w else 0:.0f} %)")
            A(f"│     (cf. §1 par famille, temps busy)")
            A(f"├── gaps GPU (intra-round) {ig/1e6 if ig is not None else 0:,.1f} ms")
            for c, dt in sorted(cls_t.items(), key=lambda x: -x[1]):
                A(f"│       dont {c}: {dt/1e6:,.1f} ms (fenêtre globale)")
            A(f"└── transition inter-round {inter/1e6 if inter is not None else 0:,.1f} ms")
            A(f"```")
        A("")
        A("> Renormalisation charge réelle : le cycle mesuré est à ~100k ctx avec acceptance")
        A("> filler élevée (mean_len ~3.5–4). La référence prod (76 ms/round à ~145k,")
        A("> mean_len 2.62) diffère par l'acceptance — la structure (phases/gaps/familles)")
        A("> est la même ; comparer les fractions, pas l'absolu.")
    A("")
    A("## 6. Indicateurs de verdict")
    A("")
    gap_share = gsum / span * 100
    A(f"| hypothèse | indicateur | valeur | seuil |")
    A(f"|---|---|---:|---|")
    A(f"| H2a gaps CPU↔GPU | occupation GPU | {util:.0f} % | < 80 % |")
    A(f"| H2a | part des gaps dans la fenêtre | {gap_share:.0f} % | > 20 % |")
    cpu_gap = cls_t.get("cpu-user", 0) + cls_t.get("cpu-syscall", 0)
    A(f"| H2a | gaps 'cpu-user+syscall' (travail CPU réel) | {cpu_gap/1e6:,.0f} ms / fenêtre | — |")
    A(f"| H2b sous-régime GPU | occupation | {util:.0f} % | > 90 % |")
    fams_sorted = sorted(fam_t.items(), key=lambda x: -x[1])
    A(f"| H2b | famille la moins attendue au top 5 | {fams_sorted[3][0] if len(fams_sorted) > 3 else 'n/a'} | — |")
    intra_m, inter_m = med('intra'), med('inter')
    A(f"| H2c overhead MTP | gap inter-round vs intra | {intra_m/1e3 if intra_m is not None else float('nan'):,.1f} vs {inter_m/1e3 if inter_m is not None else float('nan'):,.1f} us | inter >> intra |")
    A("")
    open(os.path.join(rundir, "rapport-h2.md"), "w").write("\n".join(L))

    # csv
    with open(os.path.join(rundir, "h2_familles.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["famille", "ms", "pct_busy", "nb", "us_moyen"])
        for fm, dt in sorted(fam_t.items(), key=lambda x: -x[1]):
            w.writerow([fm, f"{dt/1e6:.3f}", f"{dt/busy*100:.2f}", fam_n[fm], f"{dt/fam_n[fm]/1e3:.3f}"])
    with open(os.path.join(rundir, "h2_gaps.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["start_rel_ms", "duree_ms", "classe"])
        for a, b, g in sorted(gaps, key=lambda x: -x[2]):
            w.writerow([f"{(a-span0)/1e6:.3f}", f"{g/1e6:.3f}", gap_class(a, b)])
    with open(os.path.join(rundir, "h2_rounds.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx", "wall_ms", "verify_ms", "drafts_ms", "busy_ms", "intra_ms", "inter_ms"])
        for k, r in enumerate(rounds):
            w.writerow([k, f"{r['wall']/1e6:.3f}",
                        "" if r.get("verify") is None else f"{r['verify']/1e6:.3f}",
                        "" if r.get("drafts") is None else f"{r['drafts']/1e6:.3f}",
                        f"{r['busy']/1e6:.3f}", f"{r['intra']/1e6:.3f}",
                        f"{r.get('inter', 0)/1e6:.3f}"])

    print("\n".join(L))
    print(f"\n[OK] rapport-h2.md + h2_*.csv dans {rundir}")

if __name__ == "__main__":
    main()
