# v100-bench-kit — tests d'optimisation V100 : kit transmissible

**Ce que c'est.** Tout ce qu'il faut pour comprendre, reproduire et contester les tests
d'optimisation effectués sur une **Tesla V100-PCIE-32GB (sm_70, Volta)** servant
**Qwen3.8-27B Q6_K_M + MTP (draft-mtp n_max=3), KV q8_0, flash attention, ctx 200k**
via un fork Ollama/llama.cpp optimisé Volta (fenêtre de référence : 3 → 16 octobre 2026).

**Ce que le kit contient** (voir arborescence plus bas) :
- un **README d'entrée** (celui-ci) — architecture et démarrage à contexte vide ;
- un **historique technique honnête** (`docs/HISTORIQUE.md`) — validé / expérimental / limites ;
- les **scripts et configs** pour reproduire les essais sur une V100 (`bench/`, `h2/`) ;
- les **résultats bruts** (`results/`, `h2/ab_results_*`, `h2/nightrun8/ab_run9*`) et les
  **protocoles** pour les confirmer ou les contester (`docs/protocols/`) ;
- **licences, dépendances, prérequis** (`LICENSE`, `NOTICE.md`, `REQUIREMENTS.md`) ;
- **aucun secret** : les tokens sont lus de fichiers 0600 hors repo, les chemins locaux
  sont remplacés par la config explicite `config/kit.env.example`.

---

## 1. L'architecture en une figure

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MACHINE DE RÉFÉRENCE                                                   │
│  GPU : Tesla V100-PCIE-32GB (sm_70, Volta, 32 Go, ~794 GB/s mesurés)    │
│  + GTX 1070 (reranker, écartée des tests)                               │
│  OS : Linux, driver NVIDIA 580.178.04, CUDA 12.6                        │
├─────────────────────────────────────────────────────────────────────────┤
│  PRODUCTION (NE JAMAIS TOUCHÉE PAR LES TESTS)                           │
│  service systemd ollama-v100 : port 11434                               │
│  Ollama v0.34.4-5 (pin 16b4376a) + fork llama.cpp b11081-v100-full      │
│  .so prod : libggml-cuda.so md5 e4afbb4a (build 2026-10-04)             │
│  modèle : qwen3.8-27b-q6-mtp (GGUF 23 088 409 504 o)                    │
│  kernels custom Volta dans le .so : W4R4 Q6_K, Q5_X4, Q8 FA TC,        │
│  GDN 128×4, shortlist-head MTP, defer-prompt                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ZONE DE TEST (ce que le kit orchestre)                                 │
│  1. SERVEUR DE CAPTURE : llama-server DIRECT sur port dédié (44901/44911)│
│     - .so de test via GGML_BACKEND_PATH (REF / H2 / H2b)                │
│     - même modèle, même env prod (W4R4, shortlist, defer-prompt)        │
│     - la prod est stoppée/restaurée par trap (jamais modifiée)          │
│  2. SUPERVISEUR (bench/supervisor_bench.py) :                          │
│     - lance le bench, survive à la coupure du service (setsid/--detach) │
│     - restaure le service si le bench a échoué, reprend la génération   │
│       OWUI en erreur (voir docs/protocols/01)                           │
│  3. OBSERVATION : CUPTI 12.9 (shim LD_PRELOAD, h2/cupath_shim*.c),      │
│     perf, nvidia-smi — nsys/CUPTI 13.x INOPÉRANTS sur Volta (documenté) │
└─────────────────────────────────────────────────────────────────────────┘
```

**Règle d'or de tous les essais** (invariant vérifié après chaque run, consigné dans
`docs/HISTORIQUE.md`) : le `.so` déployé, l'unité systemd et le GGUF sont
**md5-vérifiés identiques avant et après** — un A/B ne change que
`GGML_BACKEND_PATH` sur le serveur de capture.

## 2. Démarrage à contexte vide (5 min)

1. **Lire** `docs/HISTORIQUE.md` §1 (ce qui est validé) — 10 min, tout est là :
   l'unique gain déployé (kernel attention H2, −12,5 % @107k), les pistes fermées,
   les limites des mesures.
2. **Reproduire une mesure** (la plus courte, ~5 min) : le bench de test du superviseur
   (`bench/terminal_test_bench.py`) ou un `auto_bench.py` sur le port de capture.
   Prérequis : une machine comme celle de la §1 + `config/kit.env` (copier
   `config/kit.env.example` et adapter les chemins).
3. **Contester un résultat** : suivre `docs/protocols/02-AB-ref-h2.md` — le protocole
   complet (gates, seuils fixés AVANT la mesure, ordre alterné, mesure canonique) et
   les données brutes pour recalculer sans re-lancer un run.

## 3. Arborescence

```
README.md               ← vous êtes ici
LICENSE                 MIT (code du kit)
NOTICE.md               licences des composants repris (llama.cpp, Ollama, CUPTI, modèle)
REQUIREMENTS.md         prérequis machine, dépendances, pins
config/
  kit.env.example       TOUS les chemins machine-dépendants, explicites
docs/
  HISTORIQUE.md         historique technique honnête (validé/expérimental/limites)
  SYNTHSE-20261008.md   état global au 8 oct (avant la bascule H2) — document de référence
  protocols/
    01-superviseur-bench.md   mécanique du superviseur + watchdog (reprise OWUI)
    02-AB-ref-h2.md           protocole A/B de référence : confirmer/contester
    03-build-lib-h2.md        reconstruire les libs H2/H2b isolément (jamais la prod)
bench/                  harnais génériques (superviseur, auto_bench, bench_direct, bench_gen)
h2/
  ETAT-raw.md           journal brut de la phase H2 (9→15 oct) — 1984 lignes, toutes décisions
  PROTOCOLE.md          protocole de forage initial (invariants, gates, critères H2a/b/c)
  ETAT-audit-v100-12-10.md  audit des pistes génériques V100 (matrice A–F)
  patch-h2-*.patch      LES patches source (H2 : 2 lignes, H2b : 6 sites)
  e2-src/               harnais microbench + build isolé (e1_bench, .cuh, scripts)
  nightrun8/            harnais de référence run9c3 + protocoles run9b/c/c2/c3
  ab_results_* / nmax_runs / sync_runs / perf-n3   résultats bruts (server.out, gen, TSV)
  prompt1200.json / prefill-107k.json  prompts déterministes partagés (sha256 publiés)
results/                résultats bruts des benchs préfill/génération (1-2 oct)
autopsy/                autopsie Q8_0 (P0→P7a, 5-7 oct) : ETAT + design note + swap/revert
p7/bench                kernel Q6_K X4/hoisting (8 oct) — clôturé, non intégré
bigfiles/               hors git : libs H2/H2b + binaires de microbench (sha256 publiés)
tools/                  scripts d'assainissement (chemins → env) utilisés sur ce kit
```

## 4. Repro sur une V100 — niveau d'effort par essai

| Essai | Script | Durée | Effort |
|---|---|---|---|
| Bench préfill/gén vs contexte | `bench/auto_bench.py` | ~30 min | config/kit.env + service ollama-v100 |
| A/B REF vs H2 @107k (le résultat central) | `h2/nightrun8/run9c3_bench.py` | ~25 min | + libs bigfiles/ + prompt (fourni) |
| A/B @41k | `h2/nightrun8/run9b_bench.py` | ~8 min | idem |
| Trace CUPTI d'un decode MTP | `h2/run_cupath_llama.sh` | ~20 min | + build du shim (cu 12.6, ~2 min) |
| Microbench kernel attention | `h2/e2-src/e1_bench.cu` + `h2b_gate_c*.sh` | ~10 min | nvcc sm_70 |
| Build des libs H2/H2b | `docs/protocols/03` + `h2/e2-src/h2lib/` | ~30 min | + fork source (voir NOTICE) |

**Tous** les essais stoppent/restaurent le service de prod (fenêtre 15-25 min max, trap
de restauration) et sont conçus pour être lancés via `bench/supervisor_bench.py run
<script> --detach` (survie à la coupure, reprise de l'agent).

## 5. Ce qu'il faut savoir avant de relancer quoi que ce soit

- **Le gain H2 est lié à la longueur de contexte** (−12,5 % à ~107k, ≈0 à 41k) et
  **change la séquence générée** (associativité fp32, contenu correct). C'est une
  décision d'utilisateur, pas un bug — détails et données dans `docs/HISTORIQUE.md` §3.
- **H2b** (découplage tuile/partitionnement) restaure la bit-identité au prix du gain
  applicatif : c'est la variante « propre » (patch fourni, lib en `bigfiles/`).
- **nsys / CUPTI 13.x ne tournent PAS sur Volta/driver 580** (verdict documenté, puis
  CUPTI 12.9 qui fonctionne, en prefix isolé) — ne pas gaspiller du temps dessus.
- **Les compteurs MTP du log serveur sont cumulatifs par processus** et réinitialisés
  à chaque rechargement de modèle : fenêtre de mesure = delta des 2 dernières lignes
  `spec common_specu` (c'est ce que fait le harnais de référence).
- **La mesure canonique** est la ligne `eval time = X ms / N tokens` du log serveur,
  pas le mur HTTP du bench (le mur inclut le transport + le sampling du client).
