# Autopsie `mul_mat_vec_q` — run6 decode (13/10, données existantes, sans capture)

**Fenêtre** : decode task 236, segment seq 661583..705293 (43 711 recs), W = 1576,52 ms.
**Source** : `/tmp/cupath_llama_144318` (bin) + `.names`, alignés 1:1 (0 mismatch dur_ns/stream,
§18 du verdict run7). Seuls `dur_ns` + `stream` + `seq` + le **nom manglé** sont exploités.
**Aucune** occupation GPU / chevauchement n'est inférée (borne [89,2 ; 100] % du §18, non tranchée).

Total `mul_mat_vec_q*` = **840,27 ms (53,3 % KER)** = 9 408 lances =
- **793,9 ms** de mmvq **ordinaire** (nc1 + nc4)
- **46,4 ms** de `mapped_rows` (lm_head shortlist) → voir §1.

---

## 1. Les 51 kernels `*mapped*` : IDENTIFIÉS — ce n'est ni un artefact ni un mmvq ordinaire

**Nom exact** : `mul_mat_vec_q_mapped_rows<(ggml_type)8, 1>` (namespace anonyme `_GLOBAL__N__5bce44bc_7_mmvq_cu`).
51 lances, **tous sur stream 17**, durée min 897 µs / **médiane 906 µs** / p90 928 µs / max 934 µs, mean 910 µs.

**C'est le kernel du MTP SHORTLIST du head de sortie** (lm_head), pas une variante du gemv de couche :
- **Code** : `ggml-cuda/mmvq.cu` → `mul_mat_vec_q6_K_mapped_rows<>` + `launch_mtp_shortlist_q6_K_switch`
  + `ggml_cuda_try_mul_mat_vec_q_mtp_shortlist` (condition : `cc==GGML_CUDA_CC_VOLTA`,
  `src0->ne[0]==5120 && src0->ne[1]==248320`, shortlist chargé). Le `.so` déployé contient la
  spécialisation `mapped_rows<(ggml_type)8,1>` (le .so template sur Q8_0, le source lu template sur Q6_K —
  drift build/source mineur, **même fonction** : GEMV du lm_head sur les rangées mapées).
- **Log serveur** : `CUDA MTP shortlist: loaded 131072 mapped output-head rows from
  .../qwen38-27b-exact-131072.i32 on device 0` + `GATE SHORTLIST OK`.
- **Géométrie + coût** : 131 072 rangées × 5 120 cols, lues comme Q8_0 (1,125 B/élément) = **755 Mo**
  de poids. À la médiane 906 µs → **débit effectif ≈ 833 GB/s = plafond HBM V100** (~900 GB/s peak).
- **Fréquence** : **1 fois par forward**, sur le chemin draft/verify (st17). 51 forwards au total.

**Pourquoi 910 µs vs 112,8 µs** : le shortlist du lm_head **relit 755 Mo de poids** (head de sortie
entier shortlisté) alors qu'un gemv de couche FFN n'en lit qu'une fraction (~dizaines de Mo). C'est un
**coût légitime d'une opération structurellement plus grosse**, pas un problème d'attribution.
**Verdict §1 : ne PAS agréger aux variantes ordinaires** — c'est un 4ᵉ poste (lm_head shortlist =
**2,9 % du decode / 5,5 % du mmvq**).

---

## 2. Décomposition Q6_K/ncols=4 (4 050 lances, 456,83 ms, **28,9 % du decode**)

**Signature unique** (1 seule config, vérifiée sur les 4 050) :
`mul_mat_vec_q<type=14(Q6_K), ncols_dst=4, has_fusion=0, small_k=0, halve_iters=0, q5_x4=0, rows_per_block_override=4>`
→ **rpb4** (4 lignes de poids par CUDA block), pas de fusion, pas de small_k. Tout est `rows_per_block_override=4`.

**Distribution (4 050 durées)** :
| mean | médiane | p90 | p95 | p99 | max | min |
|---|---|---|---|---|---|---|
| 112,8 µs | 132,2 µs | 152,5 µs | 155,4 µs | 160,6 µs | 164,2 µs | 14,1 µs |

- **Groupe dominant** : ~3 996 lances dans la bande 100–165 µs (le « plateau » 132 µs médian).
- **Sous-groupe rapide** : **54 lances < 50 µs** (min 14,1 µs) — même nom manglé, donc **même
  instantiation** mais **forme plus petite** (moins de rangées/cols) → ce sont les gemv Q6_K de
  dimensions réduites (attention / petites FFN) qui tombent sur le même template. C'est la queue gauche
  qui fait mean < médiane. **Deux groupes de coût distincts sous un même symbole.**
- **Fréquence** : 238,2 lances / verify-forward (= le nombre de gemv Q6_K T=4 par forward de 4 positions).
- **Correspondance couches** : le nom ne porte **pas** `ne[0]/ne[1]` → on ne peut **pas** attribuer à
  `attn_q/kv` vs `ffn_up/down` à partir du run6 seul. Ce qui est établi : c'est le **gemv Q6_K de
  couche à T=4** (batch 4, `rows_per_block=4`), la famille dominante.

---

## 3. Comparaison des variantes quantifiées (à T=4, nc4, verify)

| variante | n | lances/verify-fw | médiane | mean | p90 | max | config clé |
|---|---|---|---|---|---|---|---|
| **Q6_K**/nc4 | 4 050 | 238,2 | 132,2 µs | 112,8 | 152,5 | 164,2 | rpb4 |
| **Q5_K**/nc4 | 1 278 | 75,2 | 120,1 µs | 103,9 | 139,7 | 153,5 | q5_x4=1 |
| **Q8_0**/nc4 | 3 617 | 212,8 | **12,6 µs** | 46,5 | 133,0 | 1887 | **bimodal** |
| Q4_K/nc4 | 36 | 2,1 | 69,8 µs | 70,0 | 85,4 | 93,0 | — |
| IQ4_NL/nc4 | 18 | 1,1 | 90,9 µs | 92,0 | 100,3 | 101,0 | — |

**Le mean Q8_0 (46,5 µs) est un artefact de mélange de deux formes** — distribution **bimodale** :
- **rapide** (< 40 µs) : **2 232 lances**, médiane **8,3 µs**, 131,3/fw → petites formes (~8 µs).
- **lente** (≥ 40 µs) : **1 385 lances**, médiane **58,4 µs**, 81,5/fw → grosses formes.

→ **Interdiction du brief respectée** : on ne peut pas dire « Q8_0 est intrinsèquement plus rapide ».
À **forme comparable** (le plateau lent), Q6_K (~132 µs) ≈ Q5_K (~120 µs) > Q8_0-lent (~58 µs). Les
moyennes portent sur des **volumes de travail différents** (bytes/élément : Q6_K 0,664 B, Q5_K 0,64 B,
Q8_0 1,125 B) et des **compteurs de lances différents** (238 vs 213 vs 75 /fw). La seule conclusion
robuste : **Q6_K/nc4 est la famille la plus coûteuse en absolu** (28,9 %) parce qu'elle combine
**beaucoup de lances (238/fw)** et un **coût par lance élevé (~132 µs)**.

**Draft (st17, T=1)** : Q6_K/nc1 (255 lances, méd 82 µs, dont 153 avec `has_fusion=1`) + Q8_0/nc1
(103, méd 12,1 µs). Le draft ne fait que du nc1 (petit) + le lm_head shortlist.

---

## 4. Lien aux travaux P0–P7 (profiling : renforce / affaiblit)

| piste | statut | ce que dit le run6 | effet |
|---|---|---|---|
| **P0/P1/P2** Q8_0 QPN/TC decode | fermées (dp4a imbattable T≤9) | Q6_K/nc4 (chemin **dp4a**) = 28,9 %, mem-bound ; aucun kernel TC en decode | **RENFORCE** le NO-GO TC decode |
| **P3** decode T=4 mem-bound 84 % DRAM | établi | lm_head shortlist = **833 GB/s ≈ 92 % du pic HBM** | **RENFORCE / confirme** le régime mem-bound |
| **P4/P5/P6** ncols=8 | fermées (inaccessible, ne1∈{1,4}) | **seuls ncols=1 et ncols=4 apparaissent**, zéro ncols=8 | **RENFORCE** la fermeture (structurel) |
| **P7** kernel Q6_K X4 (+1,1 %, <2 ms, non intégré) | clôturée | Q6_K/nc4 = la **1ʳᵉ famille** (456,8 ms) ; c'est **la** cible kernel | **RENFORCE** que Q6_K est la cible, mais confirme que la **marge est petite** (~1 %) → clôture reste justifiée |
| **H-BW** requant q4_K (−27,7 % octets, +14,5 %) | appliqué | les familles requantées (Q6_K/Q5_K/Q8_0 = les grosses) sont bien celles qui dominent | **cohérent** (l'effet utile porte sur le bon poste) |

**Lecture** : le run6 ne **réouvre** aucune piste fermée. Il **confirme** (i) le régime mem-bound
(P3), (ii) l'inaccessibilité structurelle du ncols=8 (P4–P6), (iii) que Q6_K/nc4 est **le** poste à
toucher pour un gain kernel — mais avec une **marge déjà mesurée à ~1 % par P7**, donc sous le seuil.

---

## 5. Hypothèses et niveau de confiance

| # | hypothèse | conf. | preuve |
|---|---|---|---|
| H1 | `mapped_rows` = lm_head shortlist MTP, coût légitime (755 Mo / forward) | **HAUTE** | code + log + géométrie + 833 GB/s = plafond |
| H2 | Q6_K/nc4 = gemv Q6_K de couche T=4, famille dominante | **HAUTE** | 238 lances/fw, rpb4, plateau 132 µs |
| H3 | Q8_0/nc4 bimodal (8 µs petites formes / 58 µs grosses) → mean 46,5 µs non représentatif | **HAUTE** | hist. 2 232 vs 1 385, médiane 12,6 |
| H4 | les 54 lances Q6_K/nc4 < 50 µs = sous-groupe de forme réduite | **MOYENNE** | même symbole, queue gauche ; formes non visibles dans le run6 |
| H5 | correspondance couches (attn vs ffn) des gemv | **BASSE** | le nom manglé ne porte pas ne[0]/ne[1] |

## 6. Questions que run6 NE PEUT PAS répondre (sans données supplémentaires)

1. **Attribution couche** (attn q/k/v vs ffn up/gate/down) de chaque gemv — le run6 n'a ni `ne[]` ni
   corrélation avec le graphe. (→ run7 v2 avec `gpu_start/end` + export des formes, ou un dump du graphe.)
2. **Lequel des 2 groupes Q6_K/nc4** (plateau 132 µs vs queue < 50 µs) correspond à quelle couche.
3. **Occupation/chevauchement** réel (U, O, idle par round) — verrou F1, tranche seul le run7 v2.
4. **Si le lm_head shortlist (755 Mo) est compressible** sans perte (131 072 rangées sur 248 320) —
   hors scope run6 (question de données/qualité, pas de mesure).

## 7. 3 expériences candidates (chacune : hypothèse falsifiable + critère de succès)

> Aucune n'est lancée ici (contrainte : pas de benchmark/run7 à ce stade).

**E1 — Attribution couche des gemv Q6_K/nc4** (débloque H4/H5)
- *Hypothèse falsifiable* : ≥ 90 % des 456,8 ms Q6_K/nc4 proviennent de **2 couches au plus**
  (ffn_up+ffn_down), et les 54 lances < 50 µs = une couche d'attention identifiable.
- *Critère de succès* : après export `ne[0]×ne[1]` par record (run7 v2 ou dump graphe), la somme des
  top-2 formes ≥ 90 % du temps Q6_K/nc4 **ET** la forme < 50 µs = exactement la couche d'attention.
  Falsifié si le temps est étalé sur > 3 formes.

**E2 — Le lm_head shortlist est-il le vrai 4ᵉ poste à optimiser ?**
- *Hypothèse falsifiable* : réduire la shortlist de 131 072 → 65 536 rangées diviserait le coût
  lm_head (46,4 ms, 2,9 %) par ~2 **sans** dégrader l'acceptance MTP (> 95 % conservée).
- *Critère de succès* : A/B applicatif — Δacceptance MTP < 5 % **ET** Δwall/token ≥ −1 %.
  Falsifié si acceptance chute > 5 % (la shortlist est déjà minimale) → on referme le poste.

**E3 — Le plateau Q6_K/nc4 (132 µs) est-il vraiment au plafond DRAM ?**
- *Hypothèse falsifiable* : le gemv Q6_K/nc4 de plateau tourne à **< 90 %** du débit DRAM effectif
  (donc un gain kernel > 1 % existe, P7 sous-estimait).
- *Critère de succès* : mesurer bytes lus / durée (via DRAM counter ou bytes estimés × rpb4) — si
  < 750 GB/s sur le plateau → marge réelle ; si ≈ 833 GB/s (comme le lm_head) → **P7 confirmé**,
  on referme définitivement la piste kernel Q6_K. Falsifie P7 seulement si le plateau est à < 750 GB/s.

---
*Consigné : données brutes run6 (bin/names alignés), code `mmvq.cu`, log shortlist, .so déployé.
Aucune occupation GPU ni chevauchement inféré des durées cumulées (borne §18 maintenue).*
