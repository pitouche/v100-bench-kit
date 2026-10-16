# ETAT — Audit des pistes génériques d'optimisation V100 (2026-10-12)

Mission : évaluer un document générique « optimiser une V100 » (6 pistes A–F) contre NOTRE
configuration, sans code, sans benchmark, sans rouvrir n_max (consigné à 3). Résultat : matrice
synthétique + liste des pistes réellement crédibles, classées par intérêt × incertitude.

**Notre config de référence**
- Tesla V100-PCIE-32GB, sm_70 (Volta) — FP16 Tensor Cores (mma m8n8k4, accum FP32), **pas** de TC
  INT8/INT4 (Ampere+), **pas** de FP8 (Ada/Hopper). PCIe Gen3 (~900 GB/s DRAM théo, ~794 mesuré).
- Ollama/llama.cpp fork `b11081-v100-full @ d7f1e525b` (prod `e4afbb4a`), sm_70 only.
- Qwen3.8-27B **Q6_K_M** + MTP draft-mtp n_max=3, KV **Q8_0**, FA, ctx 200k, `-np 1`.
- Tracing : `libcupath.so` + CUPTI 12.9 (run6, `/tmp/cupath_llama_144318`).

**Règles d'usage des données CUPTI (imposées)**
- **Valides** : `dur_ns`, comptages par famille — après validation parsing/agrégation (fait :
  F1/F4/F9 de la revue 12-10, bin↔names alignés 705 294, agrégation par segment pid 387290 =
  l'eval task 236).
- **Exclus** : timestamps `t`, unions temporelles, chevauchements (base temporelle non validée,
  F1/F16) → l'affirmation « GPU ~100 % busy / ne s'arrête jamais » du §15.1 **n'est pas
  démontrable** avec les données actuelles ; on n'utilise QUE les parts en durées.
- **Vérification d'incohérence du rapport initial** → cf. §3 ci-dessous (une incohérence trouvée).

---

## A. Tensor Cores FP16 et calcul FP32
**Mécanisme** : Volta sm_70 = TC FP16 avec accumulation FP32 (`mma.sync.m8n8k4`). Trois choses
distinctes : (1) calcul FP16/accum-FP32 natif ; (2) conversion simple FP32→FP16 (perte de
précision, pas un accélérateur par elle-même) ; (3) **QPN / décomposition multi-passe FP16**
(Ozaki : splitter FP32 ou poids quant en plusieurs composantes FP16) pour approcher FP32 sur les TC.
**Conditions d'amélioration** : TC gagne seulement quand le batch est assez grand pour amortir le
pré-pack et la relecture des poids — pas quand on est en gemv batch-tiny.
**Pertinence notre chemin** :
- **Decode T=4 (notre cas, ne1=4)** : NO-GO démontré. La masse est `mul_mat_vec_q` (gemv K-quant
  déquantisé→fp16/fp32→**dp4a**). Le dp4a est imbattable à T≤9 car traffic DRAM minimal et ALU
  dp4a quasi nul. Le port QPN exact de NInfer (byte-exact, 25/25 correctness) est **2,7× plus
  lent que NInfer** à T=4 — à cause du **régime PCIe/L2** (7 760 CTAs, 6 Mo de L2 → relecture
  poids ~1,45×), pas du portage. TC > dp4a seulement à **T≥16**.
- **Prefill / batch (T≥16)** : TC **pertinent** — et déjà utilisé (Q8 FA tensor cores T=4 exact,
  kernel « P2 32×8 fused TC » écrit et validé dans l'autopsie). Le préfill 100k est gouverné par
  la position n_past / la traversée-KV longue, pas par le FLOPs.
**Déjà présent / historique** : **P0/P1/P2 (autopsie Q8_0, 5–7 oct) = NO-GO decode, clôturé** ;
  Q8-FA-TC en prod ; P2-32×8-fused-TC validé (prefill).
**Preuves** : autopsie `q8-qpn/ETAT.md` (verdicts P1/P2, tableaux 3-voies) ; CUPTI : 53 % du
  decode = `mul_mat_vec_q` (gemv, pas un GEMM TC-adapté).
**Incertitudes** : le régime PCIe/L2 est attribué (1,45× relecture mesurée) mais non prouvé au
  niveau nsys (nsys CLOS sur Volta/driver 580, §1 ETAT.md).
**STATUT : déjà couvert (P0/P1/P2 fermés) — decode NO-GO, préfill TC déjà en place. Ne pas
rouvrir pour le decode.**

## B. Alignement des dimensions (règle « multiple de 8 »)
**Mécanisme** : les règles génériques de multiples de 8/16 sont des heuristiques ; les dimensions
qui comptent ici sont `ncols` (largeur de tile), `ne1` (batch/rows de droite), `ne0` (cols poids),
et le tuilage propre à Volta.
**Pertinence notre chemin** : la « dimension » qui paie est le **batch**, pas l'alignement.
`mul_mat_vec_q<...,ncols=8>` = **×1,49–1,63** en µs/token (P4) — **mais P6 : la prod n'y arrive
JAMAIS** (MTP 3+1 avec `-np 1` → `ne1 ∈ {1,4}` exclusivement, zéro occurrence ne1=8
structurellement). Le decode travaille à T=4 ; les poids sont relus T fois sans amortissement.
**Déjà présent / historique** : **P4/P5/P6 (clôturé)** — ncols=8 existe et paie mais est
inaccessible dans le chemin MTP-`-np 1`.
**Preuves** : autopsie `q8-qpn/ETAT.md` P4–P6 (dispatch réel par fingerprint 16/16 configs) ;
  CUPTI : le decode est bien sur la variante `<...,4,...>` (ncols=4), pas ncols=8.
**Incertitudes** : si on changeait la structure MTP (n_max fixe à 3, `-np 1` fixe), ne1=8 resterait
hors d'atteinte.
**STATUT : déjà couvert / clôturé (P4–P6) — non prioritaire, structurellement inaccessible dans
le chemin actuel.**

## C. Fusion et compilation de graphes (torch.compile, Triton, XLA, kernel fusion)
**Mécanisme** : torch.compile/Triton/XLA = compilation/fusion à niveau framework (PyTorch). La
fusion de kernels réduit le nombre de lances et le round-trip mémoire entre ops.
**Pertinence notre chemin** : on ne **tourne PAS PyTorch** — c'est llama.cpp/Ollama avec kernels
CUDA custom. torch.compile / Triton / XLA **ne s'appliquent pas** à ce chemin (ce ne sont pas des
opti génériques transférables).
**Mécanismes équivalents DÉJÀ dans notre code** :
- **CUDA Graphs** (en prod, `graphs reused = 32` sur task 236) — capture/replay du decode,
  c'est L'outil de fusion de lances équivalent.
- **Kernels déjà fusionnés** dans ggml-cuda : `add_rms_norm_mul_f32` (add+rms+mul), `rms_norm_scale_f32`,
  `unary_gated_op_kernel` (silu gating), `gated_delta_net_cuda` (SSM).
**Pertinence CUPTI** : 665 kernels/round, médiane <5 µs, 58,6 % <5 µs → beaucoup de petits
kernels. MAIS H-SYNC (§13) a déjà **infirmé** que les syncs/launches soient dominants (11 % du
round seulement). Le vrai levier « fusion » n'est pas le nombre de lances : c'est **reduire le
travail des familles dominantes** (mmvq 53 %, q8v14partial ~30 %, cf. §3) — i.e. fusionner/décaler
le **requant KV q8_0** (dequant `q8v14partial` + `quantize_q8_1` ≈ 32 % cumulé), pas « compiler ».
**STATUT : torch.compile/Triton/XLA = non applicable (mauvais framework) ; la fusion native
(cuda graphs + kernels ggml déjà fusionnés) est déjà en place. Le levier réel = les familles
dominantes (→ §3), pas la « compilation ».**

## D. Moteurs d'inférence alternatifs (TensorRT, vLLM, TGI)
**Mécanisme / conditions** : TensorRT (ONNX, fusion agressive), vLLM (PagedAttention, continuous
batching), TGI (HuggingFace). Gains typiques : batching continu + attention paged + kernels
pré-fusionnés.
**Pertinence notre chemin** : **non applicable** pour 3 raisons :
1. **Architecture du modèle** : Qwen3.8-27B = hybrid **SSM/GatedDeltaNet + full-attention**
   (`full_attention_interval=4`, `nextn_predict_layers=1`) — très custom, pas un transformer dense
   standard ; TRT/vLLM/TGI ne couvrent pas nativement ce mix ni le **draft-mtp** (MTP n_max=3).
2. **Investissement Volta custom** : W4R4 Q6_K, Q5_X4, Q8-FA-TC, GDN Volta 128×4, shortlist-head
   (131 072 lignes, 99,2 %), deferred-prompt — tous dans le `.so` prod. Une migration TRT/vLLM/TGI
   **jette tout** (re-quant + re-implémentation des kernels).
3. **Objectif** : notre but est le decode MTP long-ctx sur UNE V100, pas le serving multi-tenant ;
   le batching continu (le vrai avantage vLLM) est inopérant en `-np 1`.
**Preuves / incertitudes** : aucune compatibilité démontrée pour le modèle+MTP+K-quant ; aucune
mesure (mission : pas de benchmark).
**STATUT : non applicable (modèle custom + MTP + kernels custom ; migration = perte de
l'investissement). Ne pas ouvrir de migration.**

## E. Quantification (AWQ, GPTQ, INT4, FP16, FP8)
**Distinguer** : (a) format de stockage des poids, (b) op de calcul, (c) capacité hardware.
- **Notre Q6_K_M** : K-quant (super-block 256 / sub-block 32, 6-bit + scales), ~6,56 bpw. Calcul :
  déquant→fp16/fp32→dp4a. C'est déjà un format **weight-only quant** efficace en bande passante.
- **AWQ / GPTQ** : quant weight-only INT4/FP16, scaling activation-aware (AWQ) / hessienne
  (GPTQ). Famille de formats **différente** des K-quants ; passer dedans = re-quantisation + perte
  des kernels K-quant custom (W4R4, Q5_X4). Pas « automatiquement plus rapide ».
- **INT4** : sm_70 **n'a PAS de TC INT4/INT8** (Ampere+) → INT4 sur Volta = déquant→fp16/SIMT,
  **pas de gain TC**. Seul effet utile : poids plus petits → moins de DRAM. Mais Q6_K (6,56 bpw)
  est déjà **plus précis** que INT4 (4,5 bpw) ; descendre à Q4_K réduit les octets (cf. H-BW) au
  prix de la précision.
- **FP16** : 16 bpw = **2,4× les octets** de Q6_K → plus de trafic DRAM → **plus lent** en decode
  mem-bound. Perte nette.
- **FP8** : sm_70 **n'a PAS de support FP8** (Ada sm_89 / Hopper sm_90) → émulation logicielle,
  **aucun gain**.
**Pertinence CUPTI** : le decode est **mem-bound sur les poids** (P3 : 84 % du DRAM atteint sur
  lm_head). Donc le levier « quantification » = **réduire les octets de poids lus**, pas changer de
  format de calcul. C'est exactement **H-BW** (§12) : requant q4_K → **−27,7 % d'octets → −6,4 %
  wall/round**, **déjà appliqué (+14,5 % tok/s, −5,7 GB)**. La piste quant est donc **déjà
récoltée** sur son seul effet utile (bande passante).
**STATUT : AWQ/GPTQ = autre famille, perte des kernels custom (non applicable tel quel) ; INT4-TC
/ FP8 = **non supporté matériellement sur sm_70** ; FP16 = perte nette (mem-bound). Le seul effet
utile (réduire les octets) est déjà fait via H-BW. Non prioritaire.**

## F. Parallélisme et gestion de processus (DataLoader, num_workers, pin_memory, MPS)
**DataLoader / num_workers / pin_memory** : mécanismes de **chargement de données pour
l'entraînement** PyTorch. **Aucun mécanisme mesuré** les rend pertinents pour le serving/inférence
llama.cpp. → **écartés** (conforme à la consigne).
**MPS (Multi-Process Service)** : utile seulement quand **plusieurs processus** se partagent le GPU
et se concurrencent. Notre décode mono-modèle tourne dans **un seul process CUDA** par fenêtre de
decode (le decode task 236 = segment pid 387290 seul, F2/F15). Les 62 forks de la trace font du
CUDA l'un après l'autre, pas en parallèle sur la même fenêtre. **MPS n'accélère PAS le decode
mono-modèle** (pas de contention multi-process à résoudre).
**Preuves / incertitudes** : CUPTI (F2) montre bien les forks mais un seul process actif par
fenêtre de decode ; aucune mesure de contention multi-process.
**STATUT : DataLoader/num_workers/pin_memory = non applicable (entraînement) ; MPS = non
prioritaire (pas de scénario multi-process pertinent pour le decode mono-modèle).**

---

## 3. Croisement CUPTI — INCOHÉRENCE trouvée dans le rapport initial (§15.1)

L'audit impose de vérifier les incohérences du rapport avant d'utiliser les %. **Le tableau du
§15.1 (ETAT.md) contient des incohérences internes** — recalculé sur `dur_ns` du segment
pid 387290 (42 618 ker, 1 575,8 ms, 17,25 rounds) :

| famille | §15.1 (us/rd, %) | **recalcul dur_ns (ms total, %)** | verdict |
|---|---|---|---|
| `mul_mat_vec_q` (toutes variantes) | ~80 000 us/rd, **53 %** | 840,3 ms, **53,3 %** | % OK, **us/rd faux** (48 710, pas 80 000) |
| `ggml_q8v14partial_kernel` | ~27 600 us/rd, **17,5 %** | 469,6 ms, **29,8 %** | **% sous-estimé** (29,8 %, pas 17,5) |
| `flash_attn_ext_vec` | ~4 400 us/rd, 2,8 % | 75,4 ms, 4,8 % | % sous-estimé |
| `quantize_q8_1` | ~1 660 us/rd, 1,0 % | 28,2 ms, 1,8 % | % sous-estimé |

**Lecture** : (1) la hiérarchie « `mul_mat_vec_q` dominant (53 %) » **tient** (le % est bon).
(2) MAIS `ggml_q8v14partial` (le prétraitement/déquant **KV q8_0**) est en réalité **29,8 % —
2e famille**, pas 17,5 % : le **coût de requant/déquant KV est sous-estimé d'un tiers** dans le
rapport, donc c'est un levier **plus gros** que présenté. (3) Le tableau du §15.1 **ne somme pas
à 100 %** (colonnes us/rd et % incohérentes entre elles). (4) L'affirmation « GPU ~100 % busy »
reste **non démontrable** (F1) — on ne retient que les parts en durées.
**Impact sur l'audit** : la piste « réduire le travail KV-q8_0 » (fusion/décalage de la requant,
piste C au sens réel, pas « compilation ») passe de « second ordre » à **2e poste après mmvq**.

---

## Matrice synthétique

| Piste | Applicabilité Volta (sm_70) | Pertinence notre workload | Travaux antérieurs | Preuves | Statut |
|---|---|---|---|---|---|
| **A** TC FP16 / FP32 / QPN | TC FP16 oui ; **pas** TC INT4/FP8 | **Decode T=4 : non** (gemv mem-bound, dp4a imbattable) ; préfill T≥16 : oui | **P0/P1/P2** NO-GO decode (port NInfer 2,7× lent, régime PCIe/L2) ; Q8-FA-TC + P2-fused en prod | autopsie q8-qpn ; CUPTI 53 % = mmvq (gemv) | **Déjà couvert (fermé)** |
| **B** Alignement dims (×8) | heuristique ; ici c'est le **batch** | ncols=8 paie ×1,5 mais `ne1∈{1,4}` structurellement (MTP `-np 1`) | **P4/P5/P6** (clôturé) | dispatch fingerprint 16/16 ; CUPTI variante ncols=4 | **Déjà couvert (clôturé), non prioritaire** |
| **C** torch.compile / Triton / XLA / fusion | framework PyTorch, **inapplicable** (llama.cpp) | CUDA graphs + kernels ggml déjà fusionnés = équivalents en place ; vrai levier = familles dominantes (mmvq 53 %, KV-q8 ~30 %) | CUDA graphs en prod ; kernels fusionnés (add_rms_norm_mul, unary_gated) | CUPTI 665 ker/rd ; H-SYNC 11 % | **Outils non applicables ; fusion native déjà en place** |
| **D** TensorRT / vLLM / TGI | supportés mais **modèle custom** | **Non** : hybrid SSM+FA + draft-mtp + K-quant + kernels custom → migration = perte de tout | n/a (aucune compat démontrée) | arch `full_attention_interval=4`, MTP n_max=3 | **Non applicable** |
| **E** AWQ / GPTQ / INT4 / FP16 / FP8 | **pas** TC INT4/FP8 sur sm_70 | AWQ/GPTQ = autre famille (perte kernels) ; INT4 = pas de gain TC ; FP16 = perte nette (mem-bound) ; FP8 = non supporté | **H-BW** (requant q4_K, −27,7 % octets, +14,5 % tok/s) déjà appliqué | P3 (84 % DRAM) ; §12 | **Effet utile déjà récolté (H-BW) ; non prioritaire** |
| **F** DataLoader / num_workers / pin_memory / MPS | n/a (entraînement) ; MPS = multi-process | **Non** : serving llama.cpp (pas DataLoader) ; decode mono-process (pas de contention → MPS inopérant) | n/a | CUPTI F2 (1 process CUDA/fenêtre) | **Non applicable / non prioritaire** |

---

## Pistes qui méritent RÉELLEMENT d'être réexaminées (classement intérêt × incertitude)

Aucune des 6 pistes génériques n'ouvre de **nouvelle** direction exploitable pour le **decode** :
A/B/E sont déjà couvertes/fermées, C/D/F sont inapplicables. Ce que le croisement CUPTI met en
avant, c'est le **re-classement interne** du budget GPU du decode (parts en durées, valides) :

1. **[INTÉRÊT élevé, incertitude faible] — Réduire le travail `mul_mat_vec_q` (53 %) par le
   *taux de lances* / le coût par gemv, pas par le choix d'algo.** C'est LA masse. Leviers connus :
   la taille de batch des layers (structurellement limitée, B) et l'amortissement de la relecture
   des poids. **Motivé par** : §15.1 (53 %, confirmé au recalcul) + P3 (mem-bound). **À confirmer** :
   run7 CUPTI v2 (F1) pour voir si le régime est stable / où est la marge réelle. C'est la
   continuité directe de P7 (clôturé, +1,1 %) — **pas une nouvelle branche**, mais la seule famille
   où une marge résiduelle est plausible.
2. **[INTÉRÊT modéré-élevé, incertitude moyenne] — Le poste KV-q8_0 (`ggml_q8v14partial` 29,8 %
   + `quantize_q8_1` 1,8 % ≈ 32 % cumulé), sous-estimé dans le §15.1.** C'est le **2e poste**,
   pas le « second ordre ». Levier : décaler/fusionner le déquant+requant du cache q8_0 (relier à
   H-BW : le cache q8_0 a un coût fixe de requant non négligeable par round, déjà noté en §15.1
   point 2). **Motivé par** : l'incohérence du §3 (29,8 % vs 17,5 %). **À confirmer** : que la
   requant ne soit pas déjà amortie par le checkpoint-KV (le prompt chaud ne requante que la
   diff).
3. **[INTÉRÊT faible, incertitude forte] — Re-démontrer / réfuter « GPU 100 % busy » (F1).**
   N'est pas une piste d'optimisation en soi, mais **bloque la lecture** des points 1–2 (si le GPU
   a des trous, le mur du round n'est pas que des kernels). Exigible avec le **run7 CUPTI v2**
   (gpu_start/gpu_end). **À faire avant** de conclure sur 1–2.

**Explicitement ÉCARTÉS** (pas de nouvelle branche, conforme à la consigne) :
- A (QPN/TC decode) — **fermé P0/P1/P2**, dp4a imbattable T≤9.
- B (ncols=8) — **fermé P4–P6**, structurellement inaccessible (`-np 1`, MTP).
- C (torch.compile/Triton/XLA) — **framework inapplicable** (llama.cpp).
- D (TensorRT/vLLM/TGI) — **migration = perte de l'investissement Volta + modèle custom**.
- E (AWQ/GPTQ/INT4/FP8) — **hors capacité hardware sm_70** (pas de TC INT4/FP8) ou perte nette
  (FP16) ; l'effet utile (octets) est déjà H-BW.
- F (DataLoader/num_workers/pin_memory/MPS) — **entraînement / multi-process**, inapplicable au
  decode mono-modèle.

---

## Limites respectées
- Audit documentaire/analytique uniquement ; **aucun changement de code, aucun benchmark**, aucune
  migration lancée.
- `n_max` conservé à **3** (non rouvert).
- Aucune conclusion de débit théorique substituée aux mesures ; les parts GPU citées sont des
  **parts en durées CUPTI validées** (dur_ns, agrégation par segment), les timestamps/unions/
  chevauchements sont **exclus** (non validés, F1/F16).
- Aucune nouvelle branche ouverte à partir des idées ; les 3 points de la liste sont la
  **continuité** de pistes déjà ouvertes (P7, H-BW, F1/run7).

**Référence croisée** : revue run6 = `h2/cupti129-test/ETAT-cupath-revue-12-10.md` (F1..F16) ;
historique P0–P7 = `autopsy-20261005/q8-qpn/ETAT.md` + `p7/bench/ETAT.md` + `SYNTHSE.md`.
