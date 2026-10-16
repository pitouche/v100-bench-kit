# Autopsie famille « KV Q8_0 » — run6 decode (14/10, données existantes, sans capture)

**Fenêtre** : decode task 236, seq 661583..705293 (43 711 recs), **W = 1576,52 ms**
(`eval time = 1576.52 ms / 64 tok`, log serveur). Source : `/tmp/cupath_llama_144318` (bin) +
`.names` alignés 1:1. Seuls `dur_ns` + `stream` + `seq` + le **nom manglé** + le **code** sont
exploités. **Aucune** occupation/chevauchement inférée (verrou F1, §18).

> **Recadrage important (fait)** : les 3 kernels groupés ici ne font **pas** la même famille.
> - `ggml_q8v14partial` + `q8v13reduce` = le vrai **flash-attention Q8** (KV en Q8_0) → **496,4 ms**.
> - `quantize_q8_1` = la **re-quant Q8_1 de l'opérande A du gemv** (pas du KV), 1:1 devant chaque
>   `mul_mat_vec_q` → **28,2 ms**, et appartient à la famille mmvq (§19), pas au KV.
> Le « 2ᵉ grand poste KV Q8_0 » est donc **essentiellement `ggml_q8v14partial` seul (469,6 ms)**.

---

## 1. `ggml_q8v14partial` — LE coût (469,6 ms, **29,8 % du decode**)

**Identité (code)** : `ggml-cuda/fattn-q8-volta.cuh`, `ggml_q8v::partial_kernel`, dispatchée par
`fattn.cu:850` quand `cc==GGML_CUDA_CC_VOLTA && Q->ne[0]==256 && V->ne[0]==256 && qwen38_q8 &&
Q->ne[1]∈(16,512]`. C'est le **QK^T + softmax + PV** de l'attention flash, **Q8_0**, en
`mma.sync.m8n8k4` (Volta, pas de `ldmatrix`), avec **déquant int8→half dans la boucle** (`deq8`).

**Paramètres effectifs (code, `fattn-q8-volta.cuh:144-152`)** :
`D=256 (head dim), G=6 (GQA), T=4 (batch), BC=16 (tile KV), BR=32, THREADS=128, D_SLICE=64,
LONG_SPLITS=160`. `splits = split_count(n_kv)`.
- **Grid** : `dim3(kvh_count, splits)` = **(4, 160)** ; **128 threads**, `__launch_bounds__(128,2)`
  → **2 blocks/SM = 256 thr/SM = 12,5 % d'occupation** (V100 : 2048 thr/SM).
- **splits = 160** pour `n_kv = 107 351` (task 236, log : `cached n_tokens=107347`,
  `n_ctx_slot=200192`).

**Comptages (mesurés, run6)** :
- **288 lances**, **toutes stream 16** (chemin verify), **16 par verify-forward × 17 verify-forward**
  (= les **16 couches flash-attention** sur 64 ; les 48 autres sont SSM, sans ce kernel).
- Durée : **min 1457 / médiane 1615 / p90 1793 / p95 1796 / max 1840 µs, mean 1631 µs**.
  **Distribution uniforme, pas de bimodalité** (contrairement au Q8_0/nc4 du §19).
- Total **469,6 ms**.

**Volumes de données par lance (MODÈLE code+géométrie — ESTIMATION, pas un compteur DRAM)** :
| flux | formule | taille |
|---|---|---|
| **KV lue** (K+V, 4 kv-heads) | `2 × 4 × n_kv × D × 1,0625 B` | **≈ 234 Mo** |
| **pacc écrite** (accumulateur partiel) | `splits × T × qh(24) × D × 2 B` | ≈ 7,9 Mo |
| **pm + pl** (stats softmax) | `2 × splits × T × qh × 4 B` | ≈ 0,1 Mo |
| **≈ DRAM total / lance** | | **≈ 242 Mo** |

- **Débit DRAM effectif estimé** : 242 Mo / 1615 µs ≈ **~150 GB/s ≈ 19 % du plafond HBM V100 (~800 GB/s)**.
  → **Le partial n'est PAS saturé en bande passante.** (À lire comme une estimation : le modèle
  d'octets est établi par le code + la géométrie, mais **aucun compteur DRAM** ne le confirme — F1.)
- **Plancher mémoire** si le kernel atteignait 800 GB/s : 234 Mo / 800 ≈ **292 µs/lance** (vs 1615 µs
  mesurés) → **marge ~5,5×** sur la composante KV.
- **Plancher TC** : QK^T+PV ≈ 10,6 GFLOP/lance → ~85 µs à 125 TFLOPS fp16 → **5 % du temps mesuré**.
  → **ni mem-bound (19 %), ni TC-bound (5 %).**

### Pourquoi autant de temps cumulé (Q4) — deux causes multiplicative
1. **Volume KV énorme par lance** : 234 Mo (K+V de 107k clés × 4 kv-heads × D=256, Q8_0) × **16
   couches FA** × 17 verify-fwd. C'est le **seul** endroit qui relit la cache KV complète par couche.
2. **Efficacité du kernel très basse** (~150 GB/s estimé, 19 % du plafond) : **occupation 12,5 %**
   (128 thr × 2 blocks/SM) + **Volta sans `ldmatrix`** + **déquant int8→half dans la boucle** +
   `__syncthreads` par tuile de 16 clés → **malaisscance du latence mémoire**. Le kernel gaspille
   ~4/5 du potentiel DRAM.

## 2. `q8v13reduce` — négligeable (26,8 ms, 1,7 %)

**Identité** : `ggml_q8v::reduce_kernel` (même fichier), fusionne les `splits=160` partielles en
l'output d'attention. Grid `dim3(qh_count, T)` = (24, 4), 256 thr. **288 lances, stream 16**,
**1:1 apparié** avec les partials (**288/288 partials immédiatement suivis d'un reduce**, gap=1).
Durée : min 85,4 / **médiane 92,4** / max 101 µs, mean 93 µs. Lit pacc+pm+pl (≈ 8 Mo) → ~86 GB/s
estimé. **Coût négligeable — pas une cible.**

## 3. `quantize_q8_1` — c'est le gemv, pas le KV (28,2 ms, 1,8 %)

**Identité** : `quantize.cu:54`, lançée par `quantize_row_q8_1_cuda` (`quantize.cu:571`), appelée
depuis `mmvq.cu:1884` (shortlist) et `mmvq.cu:2029` (gemv ordinaire). Elle **re-quant l'opérande A
(logits/activations) en Q8_1** pour le `mul_mat_vec_q` (dp4a), **avant** chaque gemv.
- **9 408 lances** (8 946 st16 + 462 st17), min 1,9 / **médiane 3,0** / max 4,4 µs, mean 3,0 µs.
- **9 357 sont immédiatement suivies (gap=1) d'un `mul_mat_vec_q` ordinaire** (= le compteur mmvq
  ordinaire du §19, 9 357) ; les **51 restantes = l'opérande A du shortlist lm_head** (§19).
- **Aucun lien avec flash-attention** : dans le chemin FA, K/V sont **déjà en Q8_0**
  (`--cache-type-k/v q8_0`), il n'y a **aucune quantization** avant `partial`. Ce kernel **appartient
  à la famille mmvq** et son coût (28,2 ms) est déjà comptabilisé au §19.

---

## 4. Chaîne de dépendances (Q2)

**Chemin flash-attention Q8 (les vrais « KV Q8_0 »)** — séquentiel par couche FA, stream 16 :
```
Q,F32 ─┐
K,Q8_0 ├─►  ggml_q8v14partial  (grid 4 kvh × 160 splits, QK^T+softmax+PV, écrit pacc/pm/pl)
V,Q8_0 ─┘            │
                     ▼
              q8v13reduce  (grid 24 qh × 4 tok, fusionne les 160 splits → out F32)
```
- `partial` → `reduce` sont **1:1 et adjacents** (288/288). Le split-KV (160) est le couplage
  structurant : il **écrit puis relit** ~8 Mo (pacc) → c'est le prix du parallelisme sur 160
  partitions.
- **Pas de `quantize_q8_1` dans cette chaîne** (KV déjà Q8_0).

**Chemin gemv (famille mmvq, rappel §19)** : `activations F32 ─► quantize_q8_1 (A→Q8_1) ─►
mul_mat_vec_q (poids quant)`. Indépendant du FA.

**Établi par le code vs estimation** :
- **Établi (code)** : D=256, G=6, T=4, BC=16, THREADS=128, LONG_SPLITS=160, grid,
  launch_bounds(128,2), splits=split_count(n_kv), tailles pacc/pm/pl, chaîne partial→reduce,
  K/V déjà Q8_0, quantize_q8_1 = A-gemv.
- **Établi (run6)** : 288/288/9408 lances, durées, streams, 16 FA/verify-fwd, appariement 1:1,
  distribution uniforme du partial.
- **ESTIMATION (modèle, pas compteur)** : octets DRAM/lance (~234 Mo KV, ~242 Mo total),
  débit ~150 GB/s, planchers 292 µs (mémoire) / 85 µs (TC).

## 5. Transformations / fusions envisageables (Q5)

- **Le vrai levier est l'efficacité du `partial`** (19 % du plafond DRAM estimé), **pas** la
  quantité de données (KV déjà en Q8_0, précision minimale raisonnable) ni le `reduce` (négligeable).
- Leviers **kernel** (hors scope, à ne pas lancer) : **augmenter l'occupation** (128 thr × 2 blocks/SM
  → 12,5 %) ; **pré-déquant K/V hors boucle** ou via `ldmatrix`-like (Volta n'en a pas) ; **réduire
  les `__syncthreads`** (1/tuile de 16 clés) ; **fusion partial+reduce** pour supprimer la passe
  pacc (mais le split-KV de 160 est ce qui parallélise sur 107k clés → fusion = moins de parallélisme,
  trade-off).
- **Fusion quantization** : déjà absente du FA (KV pré-quant Q8_0). Rien à fusionner côté KV.
- **Conclusion Q5** : il y a un **réel potentiel kernel** (marge ~5,5× si mem-bound), **mais** (a)
  c'est une modification de kernel (hors périmètre) et (b) **on ne peut pas chiffrer le gain en
  temps mural** sans l'union/chevauchement (F1) — voir §6.

## 6. Potentiel de gain — borne honnête sans l'union temporelle (Q6)

- **Borne d'énergie (sérialisée, pas du temps mural)** : si `partial` passait de 1615 µs au plancher
  mémoire 292 µs, le total partial chuterait de **469,6 → ~84 ms**, soit **~385 ms d'énergie
  sérialisée** sur 288 lances. C'est une **borne supérieure**, **pas** un gain de wall : le gain
  mural réel dépend de l'overlap/idle par round, **inconnu (F1)**.
- **Ce que run6 permet de conclure** :
  - `partial` est **le 2ᵉ poste** (29,8 % W), derrière `mul_mat_vec_q` (53,3 %).
  - Il est **limité par le kernel** (occupation/déquant/sync), **pas** par la bande passante ni le TC.
  - Le `reduce` et le `quantize_q8_1` **ne sont pas des cibles**.
- **Ce que run6 NE permet PAS de conclure** : le gain **mural** d'une amélioration du partial
  (nécessite l'union/overlap → run7 v2), et **combien** du temps partial est effectivement sur le
  chemin critique (vs caché par l'overlap des 2 streams 16/17).

---

## 7. Hypothèses + niveau de confiance

| # | hypothèse | conf. | preuve |
|---|---|---|---|
| H1 | `partial` = flash-attn Q8 QK^T+PV Volta, 16 FA-layers/verify-fwd | **HAUTE** | code + dispatch + comptage 16/fwd |
| H2 | partial **pas** mem-bound (~150 GB/s estimé ≈ 19 % plafond), ni TC-bound (5 %) | **MOYENNE-HAUTE** | modèle d'octets (code+géom.) — **pas** de compteur DRAM (F1) |
| H3 | cause racine = occupation 12,5 % + déquant in-loop + Volta sans ldmatrix + sync/tuile | **MOYENNE** | code (launch_bounds, deq8, BC=16) ; pas mesuré (pas de profil kernel) |
| H4 | `quantize_q8_1` = re-quant A du gemv (9357 mmvq + 51 shortlist), **pas** du KV | **HAUTE** | adjacence 1:1 gap=1 + code mmvq.cu:1884/2029 |
| H5 | marge ~5,5× (1615→292 µs) si partial atteignait le plafond DRAM | **MOYENNE** | borne modèle ; dépend de H2 |

## 8. Inconnues que run6 ne peut pas résoudre

1. **Octets DRAM réellement transférés** par `partial` (le ~242 Mo/lance et ~150 GB/s sont un
   modèle ; **pas de compteur DRAM**). → besoin d'un profil kernel (NCU/nsys) ou run7 v2 + DRAM.
2. **Gain en temps mural** d'une amélioration du partial (nécessite **union/overlap/idle** → F1/run7 v2).
3. **Répartition temps-critique vs overlap** du partial entre les streams 16/17 (pas de timestamps GPU).
4. **Si le split-KV=160 est optimal** (trade-off occupation vs taille pacc) — pas mesurable sans bench.
5. **Lequel des 12,5 % occupation / déquant / sync** pèse le plus (pas de profil d'occupancy/stall).

## 9. 3 expériences candidates (classées valeur informative / coût) — **non lancées**

**E1 — Mesure DRAM réelle du `partial`** (débloque H2/H5 ; valeur max, coût faible-moyen)
- *Hypothèse falsifiable* : le `partial` lit **≈ 234 Mo** et tourne à **< 300 GB/s** effectifs
  (donc H2/H3 confirmées : kernel-bound, pas mem-bound).
- *Critère de succès* : profil NCU/nsys kernel (ou run7 v2 + DRAM counter) : DRAM read ≈ 230–245 Mo
  **ET** BW < 300 GB/s sur `partial`. Falsifié si BW > 600 GB/s (→ c'est mem-bound, H2/H3 mortes,
  il faut réduire les octets, pas le kernel).
- *Coût* : un profil kernel isolé (pas de modif, pas de benchmark applicatif) — **faible**.

**E2 — Sweep d'occupation / paramètre du `partial`** (si E1 confirme kernel-bound ; valeur moy, coût moy)
- *Hypothèse falsifiable* : relever `THREADS`/blocks-per-SM (ex. 256 thr) ou **pré-déquant K/V**
  fait passer `partial` de 1615 µs à **< 800 µs** (≥ 2×).
- *Critère de succès* : A/B kernel isolé (même n_kv=107k) : Δpartial ≥ −40 % **ET** sortie bit-identique.
  Falsifié si Δ < −10 % → l'occupation n'est pas la cause (pensez déquant/TC).
- *Coût* : modif kernel + bench isolé — **moyen** (hors scope courant, à demander).

**E3 — run7 v2 (union/overlap) pour chiffrer le gain MURAL du partial** (débloque §6 ; valeur max global,
  coût élevé — déjà requis par F1)
- *Hypothèse falsifiable* : le gain **mural** d'un `partial` 2× plus rapide est **< 2× le gain en
  énergie** (partiellement masqué par l'overlap stream 16/17).
- *Critère de succès* : run7 v2 (gpu_start/end, §18) donne l'union ; le gain mural modélisé du partial
  < 0,5 × 385 ms = **< ~190 ms**. Falsifié si gain mural ≈ gain énergie (partial quasi sérialisé sur
  le chemin critique).
- *Coût* : **run7 v2 complet** (shim modifié + recapture) — **élevé**, mais **déjà exigé** par le verrou
  F1 pour toute question d'occupation/overlap. **Le faire une fois couvre E3 + §18 + tout le reste.**

> **Recommandation d'ordre** : **E1** d'abord (profil kernel isolé, faible coût, tranche H2/H3) ;
> si kernel-bound confirmé → **E2** ; **E3** se fait quand run7 v2 est de toute façon lancé (F1).
> Aucune expérience n'est lancée ici (contrainte).

---
*Consigné : code `fattn-q8-volta.cuh` / `quantize.cu` / `mmvq.cu`, log serveur (n_kv=107351, W=1576,52
ms), run6 bin/names alignés. Les débits/planchers sont des **estimations de modèle** (pas de
compteur DRAM) et **aucun** gain de temps mural n'est affiché sans l'union (F1).*
