# ETAT — Optimisations CUDA Qwen / kernel P7 (X4)
> Snapshot opérationnel pour clôture du chat. Mis à jour le 2026-10-08 20:10 (Paris).
> Objectif du fil : décider si le kernel Q6_K **P7 (X4)** remplace le **PROD (W4R4)** dans la build `.so` de prod.

---

## 1. État du système (vérifié à la clôture)
- `ollama-v100.service` : **actif**
- `.so` de prod installé : **md5 `e4afbb4a`** (= known-good, byte-identique) — **revert clean confirmé**
- Chemin `.so` : `/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so`
- V100-PCIE-32GB (idx 1) : 32 366 / 32 768 MiB (modèle chargé)
- GTX 1070 (idx 0) : 2 293 MiB (idle)
- **Aucun risque prod** : on est sur la baseline connue-good.

## 2. Contexte / fil du travail
- Bench synthétique déjà bouclé (2 oct.) — matrice 3 runs (MTP·prod / van·prod / van·vanilla), 8 points 2k→196k :
  - **MTP = ×1,85–2,26** (croît avec le contexte)
  - **kernels V100 = +3 à +9 %** (croît avec le contexte)
  - Figures + analysis dans `~/llama-experiments/bench/` (`bench_gen_analysis_20261002.md`, `bench_gen_mtp_vs_vanilla_20261002.png`, `plot_bench_20261002.py`)
- Objectif du **jour** (8 oct.) : mesurer l'utilité **réelle/applicative** du kernel P7 en conditions Ollama réelles (3 tâches fixes, T=0, MTP actif), pas en bench synthétique.

## 3. Méthode (script `p7/bench/p7_ab.sh`)
- **A** = baseline `.so` prod `e4afbb4a` (W4R4)  →  **B** = `.so` P7 `a1b3b776` (X4)
- Seul le `.so` change (`autopsy-20261005/q8-qpn/swap_revert.sh` swap/revert), même service / GGUF / env / modèles.
- Modèle : `qwen3.8-27b-q6-mtp:latest` (Qwen3.8-27B Q6_K, KV q8_0, FA, MTP draft=3), V100 idx 1, port 11434.
- 3 tâches fixes (T=0, reproductible) : T1=256 tok (history lighthouses) / T2=512 tok (merge sorted lists + tests) / T3=256 tok (problème train Paris-Lyon).
- Métriques : eval t/s, latency, **acceptance MTP** + taux par position, mean len, wall, VRAM, stabilité.
- **Rollback garanti** : revert vers known-good à la fin (même en cas d'erreur).

## 4. Résultats du jour — 2 A/B lancés (le 2ᵉ = run de référence)
- Run 1 : `res-20261008-095300`  ·  Run 2 : `res-20261008-100240` ← **celui-ci**
- `last_res.txt` → `RES=/home/guile/llama-experiments/p7/bench/res-20261008-100240`

### Données brutes (run 2) — `A.tsv` (prod) vs `B.tsv` (P7)
| Tâche | prod W4R4 | P7 X4 | delta P7 |
|---|---:|---:|---:|
| T1 (256 tok) | 45,68 t/s · 5 604 ms | 46,22 t/s · 5 539 ms | **+1,2 %** |
| T2 (512 tok) | 68,67 t/s · 7 456 ms | 69,44 t/s · 7 374 ms | **+1,0 %** |
| T3 (256 tok) | 67,26 t/s · 3 806 ms | 67,97 t/s · 3 766 ms | **+1,9 %** |
| **Global (1024 tok)** | **16 866 ms → 60,7 t/s** | **16 679 ms → 61,4 t/s** | **+1,1 %** |

- Acceptance MTP **identique A/B** : T1=0,478 / T2=0,894 / T3=0,868 (déterministe T=0 → cohérent, P7 ne change pas la distribution).
- VRAM identique (31 186 MiB), zéro crash / Xid.
- **Lecture** : P7 = **petit gain positif ~+1 % sur court contexte** (tâches < 1 k tok). Dans l'ordre de grandeur de l'effet kernels V100 au bench synthétique (mais celui-ci montait à +9 % au long contexte, là on n'a pas encore testé le long contexte en applicatif).

## 5. Problème ouvert — RÉSOLU (2026-10-08 18:48, clôture P7)
- **Le rapport de fin de `p7_ab.sh` est corrigé** : le vrai bug était dans le f-string de `summary()` — `{... if accs else 'NA':.4f}` applique le format `:.4f` **au résultat entier de l'expression ternaire**, donc à la string `"NA"` quand `means` est vide → `ValueError: Unknown format code 'f'`. Fix : formater chaque composante séparément (variables `tps`/`acc`/`mean`), NA toléré sans format.
- **Comparaison régénérée depuis `A.tsv`/`B.tsv` existants SANS relancer de run** (`regen_compare.sh res-20261008-100240`, ajouté à `console.log`) :

  | métrique | A (prod W4R4) | B (P7 X4) | delta |
  |---|---:|---:|---:|
  | eval tok total | 1024 | 1024 | — |
  | eval ms total | 16 866 | 16 679 | −187 ms |
  | **eval t/s (global)** | **60,71** | **61,39** | **+1,1 %** |
  | acceptance moyenne | 0,7466 | 0,7466 | +0,0 % |
  | mean len | NA | NA | grep journal obsolète (cosmétique) |

- Résultat **cohérent au 1/1000 près** avec l'analyse des 2 itérations faite le matin (60,70 → 61,37 t/s). Rapport final complet, P7 clôturable.

## 6. CLÔTURE OFFICIELLE P7 (2026-10-08 18:50) — décision utilisateur
> « P7 → clôture propre. H2 → prochain forage. »

- **Verdict définitif du kernel Q6_K P7 (X4)** : gain réel, reproductible, sans régression (+1,1 % applicatif court contexte, acceptance bit-identique, VRAM identique, 0 Xid) — **mais sous le seuil de rentabilité 2 ms/cycle** (~0,4 ms/cycle) → **non intégré en prod**. La prod reste en known-good `e4afbb4a` (revert clean confirmé, service actif).
- **Artefacts conservés** : patch X4 (`llama-volta-merged-p7`, `vecdotq.cuh` + `mmvq.cu`), builds `build-p7`/`build-p7b` (md5 `a1b3b776` / `c39da9b3`), TSV gate (`p7a_results_*.tsv`), A/B runs `res-20261008-*`.
- **Ce qui est acquis pour la suite (chiffres figés)** : Q6_K @ ne1=4 = 494,7 GB/s (déficit ~197 GB/s vs ancre Q8_0 691,9) ; le déficit est structurel (VDR=1, dp4a émulé sm_70, 96 regs → latence/ILP-bound) ; la marge « patch local » est épuisée (~+1,1 %) ; fermer le gap complet exigerait une réécriture kernel (VDR=2 / TC) — hors scope actuel.
- **Prochain chantier : H2 (orchestration / gaps / CPU)** — c'est là que le travail bascule maintenant. Point de départ exact : section P7b « Bascule H2 » de `autopsy-20261005/q8-qpn/ETAT.md` (outil `p5_ktrace.so`, `print_timing`, `p7a_dispatch` comme harnais ; contrainte : le decode prod est en replay CUDA graph → l'interpose ne voit rien du decode, il faut des events CUDA par étape du flux ou rejouer le vrai graph en fenêtre).

## 7. Fichiers de référence
- Script : `~/llama-experiments/p7/bench/p7_ab.sh`
- Run de référence : `~/llama-experiments/p7/bench/res-20261008-100240/` (`A.tsv`, `B.tsv`, `console.log`, `*-T*.json`, `task*.json`)
- iters : `~/llama-experiments/p7/bench/iters.log`, `iter_run1.log`
- Swap/revert : `~/llama-experiments/autopsy-20261005/q8-qpn/swap_revert.sh` (known-good sauvegardé `reference/libggml-cuda.so.e4afbb4a.known-good`)
- Builds : `~/llama-experiments/p7/work/build-p7` (P7), `build-p7b` ; sources `llama-volta-merged-p7{,b}`
- Bin : `~/llama-experiments/p7/bin/`

## 8. Prochaines actions (à partir d'ici : H2)
1. **[FAIT] Corriger `summary()`** + régénérer la comparaison depuis les TSV existants — voir §5/§6 ci-dessus.
2. **[FAIT] Décision** : +1,1 % court contexte < seuil 2 ms/cycle, P7 non intégré, prod connu-good. (Le A/B long contexte ~150–196k n'est plus nécessaire pour trancher P7 — la clôture est actée ; il reste utile comme bench de référence H2 si on veut un ancre applicatif long contexte.)
3. **[SUIVANT] H2** : localiser le trou ~30–43 ms/cycle (≈50 % du cycle 76 ms) — gaps CPU↔GPU, orchestration MTP (3 décodeurs/cycle, ~4400 nodes du graph), requant, samplers, sync. Méthode candidate : events CUDA par étape du flux MTP (draft/verify/shortlist/KV ops) en rejouant le vrai graph prod en fenêtre (pattern P5), ou `print_timing` + nsys sur une fenêtre dédiée (le script `night-20261004/nsys-t4-profile.sh` existe, à adapter — il ciblait le split W4R4/FA/GDN, rendu caduc par P7a, mais la fenêtre nsys reste l'outil le plus direct pour les gaps).

## 9. À emporter dans le nouveau chat (prompt d'amorçage)
> Reprends le fil « Optimisations CUDA Qwen » à partir de `~/llama-experiments/p7/bench/ETAT.md` (clôture P7, §6) + `~/llama-experiments/SYNTHSE.md` (vue d'ensemble).
> État : P7 FERMÉ, non intégré (gain +1,1 % < seuil 2 ms/cycle). Prod sur known-good `e4afbb4a`, service actif. Le déficit Q6_K (494,7 GB/s @ ne1=4) est figé comme structurel ; la marge kernel par patch local est épuisée.
> **Prochain forage : H2** — le trou ~30–43 ms/cycle (~50 % du cycle MTP 76 ms) non attribué : gaps CPU↔GPU, orchestration MTP, requant, samplers. Outils : `p5_ktrace.so` (inutile sur le decode en replay CUDA graph → préférer events CUDA par étape du flux), `print_timing`, harnais `p7a_dispatch`, fenêtre autoclose pattern P5. Cible : mesurer d'abord SANS toucher à rien (prod in touchée, même discipline que P0→P7a).
