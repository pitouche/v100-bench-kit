#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_bench_20261002.py — figures PNG des 2 bench du 02/10

  1. bench_prefill_20261002.png : COLD (préfill complet) + WARM (diff
     selon position) + Ollama cold (comparatif) — depuis
     bench_direct_results.jsonl (dernier run).
  2. bench_gen_20261002.png     : tg (decode) vs contexte n_past,
     + ms/step et acceptance en axes secondaires — depuis
     bench_gen_results.jsonl (dernier run).

Les deux figures sont SÉPARÉES (pas de mélange préfill/génération).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.fontsize": 9,
})
C_COLD, C_WARM, C_GEN, C_AUX = "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"


def load_last_run(path):
    rows = [json.loads(l) for l in open(path)]
    ts = max(r["ts"] for r in rows if r.get("tag") == "run_start")
    return [r for r in rows if r.get("ts", 0) >= ts and r.get("tag") != "run_start"]


def ptps(r):
    j = r["journal"]
    return (j["prompt_eval_tokens"] / (j["prompt_eval_ms"] / 1000)
            if j.get("prompt_eval_ms") else None)


def runs_split(path):
    rows = [json.loads(l) for l in open(path)]
    starts = [r for r in rows if r.get("tag") == "run_start"]
    ts_nomtp = max(s["ts"] for s in starts if s.get("mtp") is False)
    ts_mtp = max(s["ts"] for s in starts if s.get("mtp") is not False and s["ts"] < ts_nomtp)
    def win(a, b):
        return [r for r in rows if str(r.get("tag", "")).startswith("gen_") and a <= r.get("ts", 0) < b]
    return win(ts_mtp, ts_nomtp), win(ts_nomtp, ts_nomtp + 10**9)


# ============================ FIGURE 1 — PREFILL ==============================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
rows = load_last_run(os.path.join(D, "bench_direct_results.jsonl"))

# --- panneau A : COLD ---
ax = axes[0]
cold = sorted((r for r in rows if r["phase"] == "direct" and r["mode"] == "cold"),
              key=lambda r: r["C_target_tok"])
xc = [r["journal"]["prompt_eval_tokens"] / 1000 for r in cold]
yc = [ptps(r) for r in cold]
ax.plot(xc, yc, "o-", color=C_COLD, lw=1.5, ms=6, label="DIRECT cold (préfill complet)")
oc = sorted((r for r in rows if r["phase"] == "ollama" and r["mode"] == "cold"),
            key=lambda r: r["C_target_tok"])
xo = [r["journal"]["prompt_eval_tokens"] / 1000 for r in oc]
yo = [ptps(r) for r in oc]
ax.plot(xo, yo, "o--", color="#8c564b", lw=1.2, ms=5,
        label="OLLAMA cold (même GGUF, via runner 11434)")
ax.set_xlabel("tokens préfillés (k)")
ax.set_ylabel("préfill t/s")
ax.set_title("A — COLD : coût d'un préfill complet\n(plateau ~840 t/s < 8k → 397 t/s à 195k, −53 %)")
ax.set_ylim(250, 950)
ax.legend(loc="lower left")
for x, y in zip(xc, yc):
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)

# --- panneau B : WARM ---
ax = axes[1]
warm = sorted((r for r in rows if r["phase"] == "direct"
               and r["mode"] in ("warm", "warm_init")),
              key=lambda r: r["C_target_tok"])
xw = [r["journal"]["task_n_tokens"] / 1000 for r in warm]
yw = [ptps(r) for r in warm]
ax.plot(xw, yw, "s-", color=C_WARM, lw=1.5, ms=6,
        label="DIRECT warm (diff préfillée, ~2-32k selon position)")
ax.set_xlabel("position dans le KV (n_past, k)")
ax.set_ylabel("vitesse de la DIFF (t/s)")
ax.set_title("B — WARM : coût marginal d'un contexte conservé\n"
             "la même diff ~32k : 547 t/s (pos 64k) → 275 t/s (pos 195k)")
ax.set_ylim(200, 900)
ax.legend(loc="lower left")
for x, y in zip(xw, yw):
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)

fig.suptitle("Bench PREFILL — Qwen3.8-27B Q6_K · V100-32GB · KV q8_0 · FA · "
             "MTP draft=3 · −c 204800 · −b 512 −ub 512 · 1 slot   (run 02/10, ratios corrigés)",
             fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(D, "bench_prefill_20261002.png"), bbox_inches="tight")
plt.close(fig)
print("ok bench_prefill_20261002.png")

# ============================ FIGURE 2 — GENERATION ===========================
fig, ax = plt.subplots(figsize=(9.5, 5.6))
grow = runs_split(os.path.join(D, "bench_gen_results.jsonl"))[0]  # le run MTP

x = [r["context_real_tok"] / 1000 for r in grow]
y = [r["gen_tps"] for r in grow]
first = grow[0]  # 2k : premier run de session (réserve)
rest = grow[1:]

# courbe des 7 points restaurés (la courbe propre)
xr = [r["context_real_tok"] / 1000 for r in rest]
yr = [r["gen_tps"] for r in rest]
ax.plot(xr, yr, "o-", color=C_GEN, lw=1.8, ms=7,
        label="tg mesuré — points restaurés (courbe propre)")
# point 2k mis à part
ax.plot([x[0]], [y[0]], "o", mfc="none", mec=C_GEN, ms=9, mew=1.8,
        label="2k — 1er run de session (CUDA graphs froids, à nuancer)")
for r, xx, yy in zip(grow, x, y):
    ax.annotate(f"{yy:.0f}", (xx, yy), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8.5)
ax.set_xlabel("contexte KV présent avant génération (n_past, k)")
ax.set_ylabel("génération tg (t/s)", color=C_GEN)
ax.tick_params(axis="y", labelcolor=C_GEN)
ax.set_title("Bench GENERATION (decode) — tg en fonction du contexte KV existant\n"
             "Qwen3.8-27B Q6_K · V100-32GB · KV q8_0 · FA · MTP draft=3 · −c 204800 · "
             "−b 512 −ub 512 · 1 slot · 300 tok générés/point\n"
             "métrique : `eval time` du log (decode) ÷ tokens générés — préfill/checkpoint/"
             "chargement/réseau exclus",
             fontsize=9.5)

# axe 2 : ms/step (coût d'un pas de vérification)
ax2 = ax.twinx()
msp = [r["gen_eval_ms"] / (r["gen_tokens"] / r["mean_len"]) for r in grow]
ax2.plot(xr, [r["gen_eval_ms"] / (r["gen_tokens"] / r["mean_len"]) for r in rest],
         "^--", color=C_AUX, lw=1.2, ms=5.5,
         label="ms/step (coût d'un pas, cible+3 drafts)")
ax2.plot([x[0]], [msp[0]], "^", mfc="none", mec=C_AUX, ms=7, mew=1.5)
ax2.set_ylabel("ms/step", color=C_AUX)
ax2.tick_params(axis="y", labelcolor=C_AUX)
ax2.set_ylim(50, 160)
ax2.annotate("pente ≈ 3,4 ms/step par 10k de contexte", xy=(120, 105),
             fontsize=8.5, color=C_AUX,
             arrowprops=dict(arrowstyle="->", color=C_AUX, lw=0.8),
             xytext=(55, 148))

# acceptance en nuance de point (auxiliaire, pas de conclusion)
for r, xx, yy in zip(grow, x, y):
    acc = r["acceptance"]
    ax.plot([xx], [yy], ".", ms=30 * (0.5 + acc),
            color="#aec7e8", alpha=0.35, zorder=1)
ax.plot([], [], ".", ms=26, color="#aec7e8", alpha=0.5,
        label="taille du halo = acceptance MTP (auxiliaire, biaisée sur texte répéteur)")

# repères des points demandés
for ck, lab in ((2, "2k"), (32, "32k"), (64, "64k"), (128, "128k"), (192, "~196k")):
    ax.axvline(ck, color="0.75", lw=0.7, ls=":")
    ax.text(ck, ax.get_ylim()[0] + 8, lab, ha="center", fontsize=8, color="0.45")

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(D, "bench_gen_20261002.png"), bbox_inches="tight")
plt.close(fig)
print("ok bench_gen_20261002.png")


# ====================== FIGURE 3 — COMPARAISON MTP vs VANILLA ===================
mtp_r, van_r = runs_split(os.path.join(D, "bench_gen_results.jsonl"))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

ax = axes[0]
xm = [r["context_real_tok"] / 1000 for r in mtp_r]
ym = [r["gen_tps"] for r in mtp_r]
xv = [r["context_real_tok"] / 1000 for r in van_r]
yv = [r["gen_tps"] for r in van_r]
ax.plot(xv, yv, "s--", color=C_WARM, lw=1.8, ms=6, label="VANILLA (MTP désactivé)")
ax.plot(xm, ym, "o-", color=C_GEN, lw=1.8, ms=6, label="MTP draft=3 (run précédent)")
for (xx, a, b) in zip(xm, ym, yv):
    ax.annotate(f"×{a/b:.2f}", (xx, (a + b) / 2), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8, color="0.35")
for xx, yy in zip(xm, ym):
    ax.annotate(f"{yy:.0f}", (xx, yy), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=8, color=C_GEN)
for xx, yy in zip(xv, yv):
    ax.annotate(f"{yy:.0f}", (xx, yy), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, color=C_WARM)
ax.set_xlabel("contexte KV présent avant génération (n_past, k)")
ax.set_ylabel("génération tg (t/s)")
ax.set_title("A — tg mesuré : MTP vs vanilla\n(le gain MTP est quasi constant ~1.9-2.2×, et plus fort au long contexte)")
ax.set_ylim(10, 50)
ax.legend(loc="center right", fontsize=9)
ax.axvline(2, color="0.75", lw=0.7, ls=":")
ax.axvline(32, color="0.75", lw=0.7, ls=":")
ax.axvline(64, color="0.75", lw=0.7, ls=":")
ax.axvline(128, color="0.75", lw=0.7, ls=":")
ax.axvline(192, color="0.75", lw=0.7, ls=":")

ax = axes[1]
ax.plot(xv, [r["ms_per_tok"] for r in van_r], "s--", color=C_WARM, lw=1.8, ms=6,
        label="vanilla : 38.2 → 79.1 ms/tok (×2.08)")
ax.plot(xm, [r["ms_per_tok"] for r in mtp_r], "o-", color=C_GEN, lw=1.8, ms=6,
        label="MTP : 25.7 → 35.1 ms/tok (×1.37)")
for xx, yy in zip(xv, [r["ms_per_tok"] for r in van_r]):
    ax.annotate(f"{yy:.0f}", (xx, yy), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, color=C_WARM)
for xx, yy in zip(xm, [r["ms_per_tok"] for r in mtp_r]):
    ax.annotate(f"{yy:.0f}", (xx, yy), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=8, color=C_GEN)
ax.set_xlabel("contexte KV présent avant génération (n_past, k)")
ax.set_ylabel("ms/token (decode)")
ax.set_title("B — coût par token : la pente du contexte\n"
             "vanilla +2.26 ms/tok par 10k  vs  MTP +0.82 ms/tok par 10k (÷2.8)")
ax.set_ylim(20, 90)
ax.legend(loc="upper left", fontsize=9)
ax.axvline(2, color="0.75", lw=0.7, ls=":")
ax.axvline(32, color="0.75", lw=0.7, ls=":")
ax.axvline(64, color="0.75", lw=0.7, ls=":")
ax.axvline(128, color="0.75", lw=0.7, ls=":")
ax.axvline(192, color="0.75", lw=0.7, ls=":")

fig.suptitle("Bench GENERATION — MTP draft=3 vs VANILLA (MTP désactivé) — mêmes 8 points, même corpus, "
             "même runner\nQwen3.8-27B Q6_K · V100-32GB · KV q8_0 · FA · −c 204800 · −b 512 −ub 512 · 1 slot · "
             "300 tok/point · métrique `eval time` decode ÷ tokens générés", fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(D, "bench_gen_mtp_vs_vanilla_20261002.png"), bbox_inches="tight")
plt.close(fig)
print("ok bench_gen_mtp_vs_vanilla_20261002.png")
