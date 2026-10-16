# DESIGN NOTE P0 — Kernel Q8_0 ncols_dst=4, route QPN Volta (V100)

> **P0 est une hypothèse d'optimisation documentée, pas une tâche active. Aucun code ni benchmark n'est requis tant qu'une reprise explicite n'est pas décidée.**

Cette note fixe le **contrat d'implémentation**. Objectif : dans trois semaines, reprendre le sujet
sans refaire l'autopsie — tout ce qui suit est déjà vérifié sur les sources locales.

Références complémentaires : `RAPPORT-MEMOIRE-OP2-Q8_0-QPN.md` (même dossier).

---

## 0. Pourquoi ce P0 (faits, déjà établis)

- Observation mmvq (run 21:20, `.so` 634b95e) : 2816 appels, 52 clés, pas de MoE (ids=0 partout),
  fusion SwiGLU morte au decode (10/2816, tous ncols=1), W4R4 couvre déjà tout le FFN Q6_K ncols=4.
- Split FLOP-proxy decode : **Q8_0 ncols=4 = 32%** (2e tranche derrière W4R4/Q6_K à 42%).
- Le Q8_0 actuel passe par `vec_dot_q8_0_q8_1` = dp4a SIMT int8 (`vecdotq.cuh:851`), pas de tensor core.
- Le Q8_0 de prod est en Q8_0/G32 (`block_q8_0` = `ggml_half d; int8_t qs[32]`,
  `ggml-common.h:252-256`) : **même forme de bloc que le W8G32 de NInfer**
  (32 codes int8 + 1 scale par groupe de 32). Le port de la géométrie QPN est donc quasi isomorphe.
- NInfer mesure 752 GB/s à T=1 sur du W8 identique en format (794 GB/s de ceiling mesuré sur
  V100 comparable) : le roofline est atteignable par construction sur cette carte.

## 1. Garde-fous de la note

- Pas de code, pas de microbench, pas de manipulation de `ollama-v100` tant que la reprise n'est pas décidée.
- La prod reste `e4afbb4a3aa3c4ffd250fb0d97877f4f` (`ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so`).
- L'arbre `src/llama-volta-merged` (HEAD `d7f1e525b`) ne doit recevoir le kernel que **après** go/no-go.
- À la reprise, le premier livrable est le microbench du §9, pas l'intégration.

## 2. Gate exact

Entrée du kernel : `mul_mat_vec_q_switch_ncols_dst`, `mmvq.cu:1454` (`case 4:`), en parallèle du
bloc existant Q6_K/W4R4 (`mmvq.cu:1456-1475`) qui sert de modèle d'insertion.

Le chemin QPN Q8_0 est pris **si et seulement si** :

```text
type          == GGML_TYPE_Q8_0
ncols_dst     == 4
ids           == nullptr            (pas de MoE)
has_fusion    == false              (aucun gate/bias/scale fusionné)
cc            == GGML_CUDA_CC_VOLTA
env           GGML_CUDA_VOLTA_Q8_QPN != "0"    (nom proposé ; actif par défaut si absent)
```

Tout le reste retombe sur le fallback générique actuel (`mmvq.cu:1477`,
`mul_mat_vec_q_switch_fusion<GGML_TYPE_Q8_0, 4>`), sans changement de comportement.
Le gate est lisible et désactivable par env, exactement comme W4R4
(`GGML_CUDA_VOLTA_Q6_W4R4`, `mmvq.cu:1460-1462`).

**Règle de première étape (verrouillée) : pas de routage global `Q8_0/ncols=4` au premier
essai.** Au premier essai, le kernel QPN n'est appelé que depuis le microbench, jamais depuis
le dispatch prod. Le gate final, une fois les mesures à l'appui, pourra devenir **shape-aware**
si le comportement observé chez NInfer se confirme (cliff à N=1024 : « collapses to 26 GB/s »,
`w8_dispatch.cpp:~200`, d'où `kVoltaQpnMinRows = 4096` chez eux).

## 3. Contrat entrées / poids

**Activations (src1, FP32)**

- `src1` reste en `GGML_TYPE_F32` (condition amont inchangée : `ggml-cuda.cu:2034`).
- Conversion **FP32 → FP16 dans le kernel**, en registres, à la lecture de la ligne d'activation.
- Aucun buffer temporaire, aucune requantisation Q8_1, aucun passage par smem pour les activations.
- (Le chemin dp4a actuel requantise F32→Q8_1 en vol ; le nouveau chemin ne fait que convertir en FP16.)

**Poids (src0, Q8_0)**

- `block_q8_0` consommé **tel quel** : `ggml_half d` + `int8_t qs[32]` par groupe de 32 K.
- Ordre K naturel du GGUF : **aucune requantification, aucun prepack, aucun layout additionnel au chargement**.
- C'est le point qui différencie P0 du QPN de v100-skinny (qui paie un prepack fragment-order de la
  taille des poids, `fork_patches/marlin.py:49-51`) : ici le décodage doit reconstruire les fragments
  B directement depuis l'ordre naturel, comme le fait NInfer pour le W8
  (`w8_volta_qpn_gemm.cuh:25-31`).

## 4. Contrat calcul

- Instruction : **`mma.sync.m8n8k4`** (Volta), en quadpair split-N, géométrie QPN inspirée de
  NInfer `w8_volta_qpn_gemm.cuh` et de Skinny `skinny_kernels.cu:1395`.
- T (ncols_dst=4) porté sur l'**axe M (8 lignes)** de la tile MMA, N split entre les 4 quadpairs :
  - tile warp = 8(T) × 32(N) ; CTA = 4 warps × 32 lanes = 128 threads ; les 4 warps **split K** ;
  - grid = N/32 CTAs ; `kTiles = 1` suffit pour T ≤ 8 (ici T=4).
- Justification de la géométrie (NInfer `q4_volta_qpn_gemm.cuh:18-20`) : à T=4, la route 32×8
  n'utilise que 12% des lignes A ; la route 8×32 en utilise 50%, et un fragment A alimente
  4 MMAs au lieu d'1 (trafic activation divisé par 4).
- Accumulation **FP32** (`f32` fragments C), toute la durée de la boucle K.
- Le C-map (écriture des 8 résultats par lane) suit la cartographie byte-verified de
  NInfer/Skinny (`w8_volta_qpn_gemm.cuh:192-200`, `skinny_kernels.cu:1145-1147`) ; à recopier,
  pas à redériver.

## 5. Contrat mémoire

- **Poids streamés global → registres** (lecture vectorisée `uint4` = 128 B par lane par itération,
  comme NInfer `w8_volta_qpn_gemm.cuh:107-116`). Pas de smem pour les poids dans la boucle principale.
- **Pas de shared memory ni de barrier dans la boucle K.** Le seul `__syncthreads()` de tout le
  kernel est le reduce K final entre les 4 warps du CTA (NInfer `w8_volta_qpn_gemm.cuh:201`).
- **Lecture des scales — point de vigilance contractuel.** Plan de scales = 1 `ggml_half` par
  groupe de 32 → N/32 × K/32 × 2 B. Le piège NInfer (`w8_volta_qpn_gemm.cuh:117-124`) : une lecture
  scalaire 1-scale/lane amplifie le trafic (16× chez eux, fp16) ; la règle est de charger
  **kBlk scales d'un bloc par vecteur** (ici `uint2`/`uint4` sur les fp32 de `block_q8_0`,
  soit 4-8 scales par lecture). Cette règle doit passer le microbench (vérifier le plan de scales
  ne dépasse pas ~10-15% du poids en DRAM).
- Décodage int8→FP16 en registres sans repack : le pattern NInfer (`w8_volta_qpn_gemm.cuh:29-31`,
  `:144-152`) — XOR 0x80, biais fp16 1152.0 (`0x6480`), un `__hsub2` + un `__hmul2` par half2 —
  est le pattern de référence (à adapter : nos scales sont en fp32, convertir chaque scale une fois
  par groupe en `half2` avant multiplication).

## 6. Shapes de référence (réellement rencontrées en decode)

Du TSV `results/observe-mmvq-20261004-212026/mmvq-instrument.tsv` (type 8 = Q8_0, enum fork) :

| # | Shape (N × K) | Rôle supposé | Appels ncols=4 |
|---|---|---|---|
| 1 | 1024 × 5120  | attention KV (4×256) | 125 + 28 (T=2) + 8 (T=1) |
| 2 | 6144 × 5120  | attention Q (24×256) | 43 + 9 (T=2) |
| 3 | 5120 × 17408 | down-projection | 187 + 43 (T=2) |

Shape secondaire majeur : **248320 × 5120** (lm_head, 2 appels ncols=4 + 8 à T=1) — même kernel,
à valider au microbench sans en faire une shape de référence.

Lignes Q8_0 observées non attribuées à l'autopsie (à vérifier à la reprise si le kernel
s'élargit) : N=48 K=5120 (438 appels, la plus fréquente), N=10240/12288 K=5120, N=17408 K=5120.

**Contrainte de shape à connaître avant toute attente** : NInfer n'active QPN que pour
**N ≥ 4096** (`w8_dispatch.cpp:226-228`) ; à N=1024, 32 CTAs seulement et leur route QPN
« collapses to 26 GB/s » (`w8_dispatch.cpp:~200`). Le shape 1 (N=1024) peut donc légitimement
ne pas atteindre le seuil de go/no-go — **le seuil s'applique par shape, pas en global**.

## 7. Contrat correctness

1. **Comparaison numérique kernel-par-kernel, avant tout A/B Ollama.**
   - Mêmes poids, mêmes activations, mêmes shapes du §6 : sortie du kernel QPN vs sortie du
     chemin dp4a actuel (`vec_dot_q8_0_q8_1`), les deux en FP32.
   - Les deux chemins n'ont pas la même arithmétique (int8 exact vs poids FP16 + activation FP16) :
     la comparaison est **en tolérance, pas bitwise**.
   - Métrique : max abs/rel error par shape + moyenne ; le seuil d'acceptation sera fixé à la
     reprise en comparant au comportement numérique déjà toléré pour W4R4 (lequel est passé
     l'A/B temp=0 avec la même famille de déquant FP16).
2. **A/B Ollama temp=0** sur prompt identique : sorties token-identiques ou dérive immédiate
   détectée → le kernel est rejeté quelle que soit sa vitesse.
3. Ordre imposé : microbench correctness → microbench perf → A/B 33 tokens. Un échec à
   n'importe quelle étape arrête la chaîne.

## 8. Mesure finale

- **A/B Ollama 33 tokens, exactement le protocole W4R4** : bascule par env var
  (`GGML_CUDA_VOLTA_Q8_QPN=1/0`), même prompt, même shortlist actif, même build, mesure
  tok/s + acceptance. Le W4R4 de référence a été mesuré ainsi (gain ~2%, « A/B de la nuit »).
- Lecture attendue : la tranche Q8_0 ncols=4 pèse 32% du FLOP-proxy decode ; un gain de 1,5×-3×
  sur cette tranche correspond à ~+5%-15% de tok/s global (estimation, pas une promesse).

## 9. Protocole verrouillé (ordre imposé, go/no-go inclus)

1. **Ne rien toucher à la prod.** La prod reste `e4afbb4a...` ; le microbench tourne dans
   `autopsy-20261005/`, avec nvcc 12.6 local, sur GPU 1, sans swap de `.so`.
2. **Microbench baseline Q8_0 `ncols_dst=4` d'abord.** Le chemin dp4a actuel
   (`mul_mat_vec_q<GGML_TYPE_Q8_0, 4>` via `mmvq.cu:1477`) est mesuré seul, sur les shapes du §6,
   avant d'écrire une seule ligne du kernel QPN. C'est le dénominateur du seuil de 1,5×.
3. **Porter uniquement le QPN `m8n8k4` nécessaire**, avec FP32→FP16 in-kernel (contrats §2-§5).
   Port minimal : une seule instanciation, `kTiles=1`, les 3 shapes de référence. Pas de
   sweep de géométrie au premier essai (pas de table `kBlk`/`SPLITK` type Skinny/NInfer).
4. **Tester par shape** : 1024×5120, 6144×5120, 5120×17408, puis les autres shapes Q8_0
   pertinentes (248320×5120 lm_head ; les lignes non attribuées N=48 K=5120 si le port s'élargit).
5. **Comparer numériquement au dp4a** (contrat §7 : tolérance, max abs/rel par shape).
6. **Mesurer le débit mémoire / speedup** par shape (GB/s effectifs vs baseline, pas seulement
   le ratio de temps).
7. **≥1,5× le baseline GEMV sur une shape → cette shape est candidate pour intégration**
   (gate shape-aware, pas global au premier essai — §2). Les shapes sous le seuil restent sur
   le fallback dp4a.
8. **Ensuite seulement : A/B Ollama 33 tokens**, exactement le protocole W4R4 (bascule par env,
   même prompt, même shortlist actif, même build, tok/s + acceptance).

Le microbench n'est pas lancé maintenant ; ce protocole s'applique à la prochaine reprise
explicite. Un échec à l'étape 5 ou 7 arrête la chaîne pour la shape concernée.

## 10. Risque

- Kernel nouveau sur un chemin critique (tous les GEMM decode Q8_0 : attention main + draft MTP).
- **Mais le layout est particulièrement favorable** : `Q8_0 = int8 × scale / groupe 32`,
  isomorphe du W8G32 de NInfer, sans prepack ni requant — c'est le port le plus court du
  répertoire QPN.
- Risques résiduels identifiés :
  1. plan de scales fp32 (2× l'empreinte fp16 de NInfer) → règle de lecture vectorisée du §5 ;
  2. T=4 = 50% d'utilisation des lignes A (inhérent à la géométrie split-N) ;
  3. shape N=1024 peu de CTAs (seuil par shape du §9) ;
  4. précision FP16 vs dp4a int8 exact (contrat tolérance du §7).

## 11. Non-objectifs

- Pas de FP8 / NVFP4 (pas de tensor core FP8/NVFP4 sur V100 ; les kernels Skinny/NInfer
  déquantent vers FP16/MMA).
- Pas de `m16n8k16`, pas de `ldmatrix`, pas de `cp.async` (toutes sm80+, hors cible).
- Pas de changement de GGUF, pas de requant, pas de prepack au chargement.
- Pas de fusion SwiGLU (morte au decode, §0).
- Pas d'élargissement à ncols_dst=2/1, au lm_head, ou aux shapes non attribuées : possibles
  extensions après le go, jamais dans ce P0.

## 12. Références figées (fichier:line)

**Notre code** — `src/llama-volta-merged/ggml/src/ggml-cuda/`

- `mmvq.cu:1454` `case 4:` ; `:1456-1475` bloc modèle Q6_K/W4R4 ; `:1477` fallback Q8_0 actuel
- `mmvq.cu:1281` dispatch commun ; `:1102-1113` fusion limitée à ncols=1
- `vecdotq.cuh:851,865` `vec_dot_q8_0_q8_1` (baseline dp4a) ; `:243` `VDR=2`
- `common.cuh:720` `ggml_cuda_dp4a`
- `ggml-common.h:252-256` définition `block_q8_0`
- `ggml-cuda.cu:2034` condition amont (src1 F32) ; `:2107` appel mmvq

**NInfer** — `autopsy-20261005/ninfer-v100/src/ops/linear/w8/`

- `w8_volta_qpn_gemm.cuh:6-31` rationale W8 + mesures (752 GB/s T=1, 794 ceiling)
- `w8_volta_qpn_gemm.cuh:47-51` schedule 4 warps/32 cols/8 rows
- `w8_volta_qpn_gemm.cuh:107-124` boucle poids + règle de lecture des scales
- `w8_volta_qpn_gemm.cuh:130-160` décodage int8→FP16 + MMA par slice
- `w8_volta_qpn_gemm.cuh:192-201` C-map + unique barrier
- `q4_volta_qpn_gemm.cuh:5,18-20,25,31,36,41` rationale 32×8 vs 8×32, mapping byte-verified
- `w8_dispatch.cpp:211-228` band QPN mesurée + constantes (T 3..8, N ≥ 4096)
- `w8_config.h:62-83` geometries + bornes T exactes

**Skinny** — `autopsy-20261005/v100-skinny/`

- `kernels/skinny_kernels.cu:1385-1395` QPN2 (637 GB/s M=8 vs 441, M ≤ 8)
- `kernels/skinny_kernels.cu:1577,1677` QPN8 FP8 + MT=2 (référence M 9..16)
- `kernels/skinny_kernels.cu:1145-1147` macro MMA_8N8K4
- `fork_patches/marlin.py:46-61` coûts/bénéfices du prepack (ce que P0 évite)
