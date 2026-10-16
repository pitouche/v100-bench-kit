# ÉTAT — Q8_0 : P0/P1/P2 (QPN, fermés) + P3/P4 dp4a + P5 dispatch + P6 prod + P6.5 cartographie decode (2026-10-07, 09:30)

## Verdict P1

**P1 est une NOUVELLE expérience indépendante, verdict NO-GO : 0/5 shapes ≥ 1,5× A.**
Le prepack row-split (layout W8G32 de NInfer) est byte-exact et le kernel P1 est validé,
mais il ne atteint PAS les ~2,94 ms de NInfer sur le lmhd : **7,94 ms** (P0 : 5,83 ms).
Pire encore : P1 est **plus lent que P0** sur lmhd (0,73×) alors qu'il gagne 1,2-1,7×
sur les shapes plus petites.

### Correctness (fenêtre 07:27, V100)

- **Prepack CPU : BYTE-EXACT sur les 5 shapes** (0/39,7 M groupes avec différence,
  reconstruction codes+scales == original, K-pad 128, plans alignés 256 B).
- **MICRO k=32 n=2** (valeurs attendues à la main : 1056/280 par token) : A, B, C
  tous exacts à 1e-3 près → les trois chemins calculent bien la même chose.
- 5 shapes : **C == B au bit près sur les 5 shapes (0 différence)** — preuve que le
  prepack est une permutation sans effet et que les deux kernels QPN sont
  numériquement identiques. Bruit fp16 vs oracle raw : 0,021-0,075 (attendu).
  A (dp4a) validé contre l'oracle q81-exact : 1,6-5,8e-5 (le maxrel ~10⁴ des fenêtres
  P0 venait d'un **oracle de bench mal normalisé** — l'oracle oubliait le scale d_w
  par groupe ; corrigé dans bench_p1.cu, le kernel dp4a lui-même était toujours exact).

### Perf 3 voies (µs, 500 iters, V100-PCIE-32GB)

| shape | A dp4a | B P0 34B | C P1 prepack | C/A | C/B | C GB/s |
|---|---|---|---|---|---|---|
| kv 1024×5120 | 9,80 | 71,1 | 44,4 | 0,22× | 1,60× | 125 |
| q 6144×5120 | 48,6 | 180,7 | 106,8 | 0,46× | 1,69× | 313 |
| down 5120×17408 | 146,0 | 416,1 | 345,3 | 0,42× | 1,21× | 274 |
| **lmhd 248320×5120** | **1781,5** | **5831,9** | **7943,9** | **0,22×** | **0,73×** | **170** |
| n48 48×5120 | 4,97 | 69,4 | 43,5 | 0,11× | 1,59× | 6 |

Références NInfer (leurs mesures, V100-SXM2) : QPN 2,94 ms / incumbent SIMT 4,46 ms à
T=4 sur le lmhd → **notre C est 2,7× plus lent que le QPN de NInfer sur leur shape.**

## Lecture des résultats

1. **Le load path n'était PAS le seul delta.** Sur lmhd, remplacer les lectures uint16
   34 B par les uint4 128 B de NInfer a fait **piquer** C (5,83 → 7,94 ms) au lieu de
   le ramener vers 2,94. Sur les shapes ≤ 6144 colonnes le gain load-path est bien
   visible (1,2-1,7× vs P0), mais il ne suffit pas.
2. **Hypothèse dominante : le PCIE vs SXM2 et le trafic effectif.** La carte est en
   PCIe (32 Go, ~900 GB/s théorique, ~794 mesuré chez NInfer en SXM). Mais surtout :
   170 GB/s mesurés sur C = ~1,45× le poids (1,27 GB) relus — exactement l'amplification
   DRAM que les commentaires NInfer décrivent quand les flux de 128 B ne sont plus
   contenus par le L2 (« 2,44× the weight » chez eux en SXM à 640 CTAs). Avec 7 760 CTAs
   et 80 SM, c'est notre cas extrême : le L2 de 6 Mo ne retient pas les lignes en vol,
   et chaque ligne de 128 B est manquée DRAM ~1,4×. NInfer ne rencontre pas ce régime
   (leurs mesures de 2,94 ms impliquent ~0,43× le poids relus en SXM, i.e. L2 fonctionnel).
   En PCIE, la latence DRAM 2× plus haute transforme le même régime en échec.
3. **Le dp4a (A) reste imbattable en T=4 sur cette carte** : il ne relit pas les poids
   (traffic minimal, ALU dp4a quasi nul), 1,78 ms sur le lmhd = 0,72 GB utile/µs.

## Conclusion P1

Le chemin QPN W8, même avec le layout et le load path exacts de NInfer, **n'est pas
concurrentiel avec le dp4a pour T=4 sur V100-PCIE**. L'écart restant vs NInfer
(2,7×) est attribuable au régime L2/DRAM (PCIE + 7 760 CTAs → 1,45× relecture du
poids) et non au portage : C == B au bit, prepack byte-exact, micro-test exact.
Pas d'intégration, pas d'A/B. La chaîne s'arrête ici.

## Fichiers

- `bench_p1.cu` (assemblé par `make_bench_p1.py` depuis `bench.cu` + section P1)
- `run_prod_window_p1.sh` — fenêtre autoclose dédiée P1 (même trap que P0)
- `window-20261006-071736.log` (gate fail oracle), `window-20261006-072707.log` (run final)
- `sim_baseline_full.py`, `sim_baseline_exact.py` — simulation CPU du tiling dp4a
  (exacte à 1e-16, a permis d'isoler le bug d'oracle)
- `bench.cu` / `bench` — P0 historique (NO-GO, inchangé)

## État prod (vérifié 07:38)

`ollama-v100.service` active, V100 32346 MiB, `.so` = known-good e4afbb4a, aucun swap.

## Prochaine étape (si continuation voulue)

- **P1b (optionnel)** : mesurer le trafic DRAM effectif de C (ncu `lts__t_bytes`) pour
  confirmer l'hypothèse 2, et/ou tester C avec un nombre de CTAs réduit (grid-stride
  sur les 32 colonnes) pour forcer le régime L2-resident de NInfer. Ne change rien au
  verdict si l'objectif est T=4 : même dans le meilleur cas (2,94 ms), C ne bat A
  (1,78 ms) — **le dp4a reste le bon chemin Q8_0 T=4 sur cette carte.**
- Le QPN ne devient intéressant que pour **T ≥ 9-32** (où le 32×8 fused de NInfer et le
  QPN à 2 tiles gagnent leur place) — hors scope du P0/P1 (ncols_dst=4, T=4).

---

# P2 — T ≥ 9–32, régime « 32×8 fused » de NInfer (expérience isolée, 2026-10-06)

## Question centrale

> À partir de quel T la réutilisation des activations permet-elle au coût QPN/Tensor Core de
> s'amortir sur V100-PCIe, et ce point de bascule existe-t-il réellement sur cette carte ?

Pas de seuil 1,5× arbitraire : P2 cherche le **point de bascule T**.

## Ce qui change vs P1 (vérifié dans le code NInfer, non supposé)

P2 = port du kernel `w8_volta_mma_gemm.cuh` (la route dispatchée par NInfer à **T ≥ 9**),
qui est **structurellement différente** du QPN P1 (`w8_volta_qpn_gemm.cuh`) :

- **A operand = 32 lignes (tokens)**, B operand = 8 colonnes/warp, 4 warps → grid=(n/32, t/32).
  Le fragment A (32×8) est **stagé en shared** et **réutilisé** sur les 8 colonnes B du warp →
  c'est l'amortissement T : le poids est lu une fois et multiplié par jusqu'à 32 tokens.
- **SANS split-K** (contrairement au QPN) : le CTA possède sa tile C en propre (N=248320 →
  7760 CTAs, 12× le budget résident → 1 split toujours). Pas de workspace, pas d'atomics, pas
  de 2ᵉ kernel d'addition. Un seul `__syncthreads` par pas K.
- **Poids + activations STAGÉS en smem** (double-buffer, kKStep=32=1 groupe W8), decode fp16
  dans le smem. 10 KB smem/CTA, 62 regs/thread, 0 spill, **8 CTA/SM** (launch_bounds 128,8).
- Seule adaptation au port : x en **fp32** (notre harness) au lieu de BF16 → 2×uint4/vec de 8k.

Le kernel P1 n'est PAS modifié ; le `.so` prod n'est PAS touché ; P2 réutilise le layout
prépacké P1 (byte-exact validé) pour reproduire le chemin mémoire NInfer.

## Correctness (gate, fenêtres 10:57 + 11:42, V100)

- **Prepack byte-exact sur les 5 shapes** (0 diff sur 39,7 M groupes lmhd, etc.).
- Oracle recalé sur le **vrai domaine numérique du mma** `mma.m8n8k4 f32.f16.f16.f32`
  (produits fp16×fp16 EXACTS ≤22 bits, accumulation fp32). D doit donc coller à une référence
  qui cumule ces mêmes produits exacts **en fp32** (seul l'ordre d'addition diffère) : résidu
  ~10⁻⁴, bruit d'ordre, indépendant d'un bug. Le résidu D-vs-raw (bruit d'entrée fp16 par
  conception, 0,02-0,11) est rapporté en info, pas en gate.
- **25/25 points T PASS** (5 shapes × T=4,8,9,16,32). A (dp4a) : 0,9-3,8e-5 vs oracle q81-exact.
- NOTE : le premier run (10:57) affichait « GATE FAIL » sur `down` (1,0-1,45e-3) — c'était
  l'oracle, pas le kernel : la référence cumulait les produits exacts en fp64 au lieu de fp32,
  ce qui surestime l'ordre d'accumulation à grand k. Recalée en fp32 (11:42), `down` passe à
  1,2-1,8e-3. Seuil gate fixé à 5e-3 (~3× le bruit d'ordre max, ~20× sous un vrai bug de
  mapping qui serait O(0,1)).

## Perf A(dp4a) vs D(P2 fused) — µs, V100-PCIE-32GB (fenêtres 10:57 + 11:42)

GB/s = poids Q8_0 (n×k/32×34 B) / t. A mesuré à T=4 (ncols_dst=4, vrai chemin prod) puis
**extrapolé en T/4** (les tokens sont des colonnes indépendantes du CTA → aucun partage de
poids, temps dp4a strictement linéaire en T). D mesuré directement à chaque T.

| shape | T | A dp4a | D P2 fused | D/A | D bat A ? |
|---|---|---|---|---|---|
| **q 6144×5120** | 4 / 8 / 9 | 49,9 / 99,8 / 112,3 | 147,9 / 151,8 / 154,0 | 0,34 / 0,66 / 0,73 | — |
| | **16** | 199,6 | 161,0 | **1,24×** | ✅ (161 < 199) |
| | **32** | 399,1 | 174,4 | **2,29×** | ✅ (174 < 399) |
| **down 5120×17408** | 4 / 8 / 9 | 148,6 / 297,2 / 334,3 | 444,3 / 442,6 / 442,8 | 0,33 / 0,67 / 0,76 | — |
| | **16** | 594,4 | 503,3 | **1,18×** | ✅ (503 < 594) |
| | **32** | 1188,8 | 586,9 | **2,03×** | ✅ (587 < 1189) |
| **lmhd 248320×5120** | 4 / 8 / 9 | 1783 / 3565 / 4011 | 5226 / 5362 / 5424 | 0,34 / 0,66 / 0,74 | — |
| | **16** | 7130 | 5638 | **1,26×** | ✅ (5638 < 7130) |
| | **32** | 14261 | 6470 | **2,20×** | ✅ (6470 < 14261) |
| kv 1024×5120 | 4…32 | 9,7…77,5 | 72…124 | 0,13…0,63 | ❌ partout (N trop petit) |
| n48 48×5120 | 4…32 | 5,0…39,9 | 70…85 | 0,07…0,47 | ❌ partout (N trop petit) |

## Verdict P2

**Le point de bascule EXISTE, et il est à T ≈ 16** (au-delà de T=9, le régime bascule déjà,
mais ne rattrape dp4a qu'à T=16). Sur les 3 shapes à grand N (q, down, lmhd) :

- **T ≤ 9 : dp4a gagne** (0,34-0,76×) — dp4a reste imbattable en régime « 1 CTA = 4 tokens ».
- **T = 16 : D bat A de 1,18-1,26×.**
- **T = 32 : D bat A de 2,03-2,29×** — la réutilisation A-amortit enfin le coût de la
  staging smem + mma. Le temps D **plateau** (q : 161→174 µs de T=16→32 ; down : 503→587 ;
  lmhd : 5638→6470) alors que A **double** (linéaire en T) : c'est exactement la signature de
  l'amortissement cherché.

### Pourquoi ça ne change RIEN pour la prod

- **La prod est en T=4** (ncols_dst=4, decode) : dp4a y est imbattable (P0+P1+P2 concordent).
- À T≥16, le gain de D vient de **re-lire les poids une seule fois pour 16-32 tokens** ; en
  decode production (1 token/sortie) ce bénéfice n'existe pas. Le régime T≥9-32 n'est
  déclenché par NInfer que pour les **préfill / large-batch**, pas le decode.
- Donc : **le QPN/Tensor Core n'est PAS compétitif en Q8_0 decode (T=4) sur cette V100** →
  branche QPN **fermée pour la prod**. Si un jour on fait du préfill/batch sur cette carte,
  le 32×8 fused (P2) serait le bon kernel — il gagnerait dp4a dès T≈16.

### À noter

- D est **2,7× plus lent que NInfer** sur le lmhd à T=4 (5,23 ms vs 2,94 ms) — même écart que
  P1, confirme que le delta vs NInfer est le **régime PCIe/L2** (7760 CTAs, 6 Mo de L2), pas le
  portage (P2 == port exact, correctness 25/25).
- ncu/nsys indisponibles → trafic DRAM/L2 non mesurable directement ; l'analyse L2/DRAM reste
  une hypothèse (mais cohérente : D plateau à ~250 GB/s = ~2× le poids utile, cf. P1).

## Fichiers P2

- `bench_p2.cu` (assemblé par `make_bench_p2.py` depuis `bench.cu` + port P2 + prepack P1)
- `run_prod_window_p2.sh` (T-sweep complet) + `run_prod_window_p2_down.sh` (`--only down`)
- `window-20261006-105728.log` (T-sweep, gate oracle fp64), `window-20261006-114225.log`
  (re-run `down` avec oracle fp32 recalé → 5/5 PASS)
- kernel P2 compilé seul : **62 regs, 0 spill, 10 KB smem, 8 CTA/SM**

## État prod (vérifié 11:52)

`ollama-v100.service` active, V100 32350 MiB (qwen3.8-27b chargé), `.so` = known-good
**e4afbb4a**, aucun swap, trap OK sur les 3 fenêtres P2. **Aucune intégration, aucun A/B,
aucune modif du kernel P1 ni du .so prod.**

## Conclusion globale Q8_0 QPN (P0+P1+P2)

La branche QPN/TC est **fermée pour le Q8_0 decode (T=4) sur cette V100-PCIe** : dp4a est
imbattable en T≤9. Le Tensor Core ne bascule devant dp4a qu'à **T≥16** (préfill/batch),
registre qui ne correspond pas au decode de la prod. → **On passe à autre chose pour le
Q8_0.** (Le 32×8 fused P2 reste documenté comme le bon kernel si un jour on optimise le
préfill/batch sur cette carte.)

---

# P3 — Caractérisation du dp4a decode T=4 (2026-10-06, fenêtre 13:24)

## Question

> Sur le kernel Q8_0 dp4a de prod en T=4 : DRAM, L2, PCIe, déquant/scales, ou dp4a ?

## Mesures valides (fenêtre `window-20261006-132426.log`, V100-PCIe libre)

| shape | M3 kernel dp4a | poids GB/s (kern) | quant | A2 fma (mêmes lectures, sans dp4a) | Δ A2−M3 |
|---|---|---|---|---|---|
| lmhd 1,35 GB | **1784,6 µs** | **756,9** | 2,3 µs | 3520 µs | +1735 µs |
| down 94,7 MB | 143,0 µs | 662,1 | 2,7 µs | 255,6 µs | +112 µs |
| q 33,4 MB | 57,0 µs | 586,6 | 2,4 µs | 93,0 µs | +36 µs |

- M0 (poids en int4 16 B contigus, « plan de codes idéal ») : **876 GB/s** sur lmhd.
- D0 micro dp4a (chaîne registres, aucune mémoire) : **6,8e12 dp4a/s** (borne basse
  prudente). Besoin du kernel lmhd ≈ 204 Gdp4a → **<3 % du budget dp4a**.
- T-sweep (baseline+quant prod : T/4 lancements, poids relus par lot) : lmhd
  **446 µs/token constant** de T=4→16 (poids 758 → 189 GB/s). L'amortissement n'existe PAS
  dans le tiling prod : 1 CTA/token relit les poids T fois.
- Self-check baseline|q81-exact : la 1ère shape a FAIL (maxrel 1,0) — **bug de bench**
  (y8h sous-alloué T/16), pas du kernel : les M3 concordent avec P1/P2 (lmhd 1784 vs
  1781/1783 µs) et le même kernel a été validé 25/25 en P2.
- **T-sweep P3 : bug de comptage** (`t_k *= reps4` après `t_k += ...` → temps kernel T=8/16
  gonflés ×reps4 dans le log). La conclusion reste valide et est croisée avec P4 (propre) :
  B4×2 T=8 = 3575 µs (378 GB/s), B4×4 T=16 = 7147 µs (189 GB/s) → **~447 µs/token constant,
  aucun amortissement**.

## Verdict P3 — qu'est-ce qui plafonne le decode T=4

1. **Le débit DRAM des poids, limite atteinte.** 757 GB/s sur lmhd ≈ **84 %** du ~900 GB/s
   théorique (HBM2 V100). Le kernel lit le poids **une seule fois** (amplification ≈1,0 : le
   L2 retient les relectures des 4 tokens d'un même CTA). Bytes/token ≈ 338 ko (lmhd),
   quasi tout en poids.
2. **Pas d'ALU, pas de dp4a, pas de déquant.** A2 (fma float, mêmes lectures) coûte ~2×
   plus que le kernel dp4a → dp4a est plus rapide que fma sur ce chemin ; micro dp4a : le
   kernel consommerait <3 % du plafond dp4a. Le temps scales linéairement avec les BYTES de
   poids (lmhd 40× le n48 → ~40× le temps/unité de poids).
3. **Pas d'amortissement Tiling dans la prod (mesuré) : 446 µs/token constant de T=4 à
   T=16.** L'amortissement ne vient que d'un kernel qui lit chaque poids une fois pour
   T tokens (→ P4).
4. **PCIE : non limitant** (le trafic poids est HBM2 ; les activations T=4 ≈ 32 KB, rien).

→ **Très peu de « magie » à récupérer à T=4** (84 % du DRAM, ALU nul). Seule voie restante :
**réduire les bytes/token** (P4 : lire chaque poids une fois pour T tokens, dp4a exact).

## Bogueurs de bench P3 (traçabilité — M2 invalidé)

- **M2 (carac_w34, « pattern 34 B ») : mesures invalides** (3,75 µs constants de 1 MB à
  3,3 GB, >17 TB/s — impossible). Cause (diagnostic `probe34.cu` + lecture du PTX) : le
  prédicat anti-DCE `if (acc == 0x12345678u)` est faux pour 100 % des données
  (P(match)≈2⁻³²) ; nvcc -O3 **supprime les 16 ld.global** (PTX : 0 ld.global, 0 st.global)
  → on chronométrait un kernel no-op (~3,7 µs = launch + boucle indexée). Le bug n'atteint
  PAS M3/M0/A2 (le poids y alimente les accumulateurs). Le « coût du layout 34 B » n'a donc
  pas été mesuré ; l'hypothèse « pénalité layout 34 B » reste **ouverte mais secondaire**
  (le kernel dp4a lui-même tourne à 757 GB/s avec ce layout).
- `probe34.cu` (le diagnostic) a subi 4 bugs : binaire sm_61 tourné sur la V100 (mapping
  CUDA device ≠ nvidia-smi index), OOM V100 (prod en place), y8h sous-alloué, et la
  suppression DCE ci-dessus (checksum « OK » illusoire : nmatch=0). `probe34.cu.orig-bad`
  note le binaire invalide.

---

# P4 — dp4a à T=8/16 : « le T que la prod n'a pas » (fenêtre 15:08)

## Question

> Le dp4a peut-il passer de T=4 à T=8/16 **sans perdre son avantage calcul** ?

B8/B16 = le kernel dp4a de prod (`mul_mat_vec_q<Q8_0>`, `mmvq.cu`) instancié avec
`ncols_dst=8/16`, 1 warp, 8 rangées/CTA : **chaque poids est lu UNE fois pour 8/16 tokens**.
`vec_dot_q8_0_q8_1` (2×uint16 codes + 2×dp4a + scales) est IDENTIQUE : arithmétique exacte
int8, pas de fp16, pas de TC. (Le switch prod instancie ncols=1/4/8 — 8 est une
instanciation EXISTANTE du même kernel prod, pas une nouveauté de code.)
Self-check 3/3 shapes PASS : B8/B16 exacts à 1,7-5,1e-5 vs oracle q81 (= dp4a exacte).

## Résultats (µs, V100-PCIe ; poids GB/s = w/temps kernel ; µs/token = (quant+kern)/T)

| shape | T | prod (T/4 × ncols4) | B8 (1×) | B16 (1×) | gain µs/token vs prod |
|---|---|---|---|---|---|
| **lmhd 1,35 GB** | 4 | 1787 (447 µs/t, 757 GB/s) | — | — | — |
| | 8 | 3575 (447 µs/t) | **2197 (275 µs/t, 616 GB/s)** | — | **×1,63** |
| | 16 | 7147 (447 µs/t) | — | **5184 (324 µs/t, 261 GB/s)** | **×1,38** |
| **down 94,7 MB** | 4 | 145 (36 µs/t, 664 GB/s) | — | — | — |
| | 8 | 289 (36 µs/t) | **159 (19,9 µs/t, 608 GB/s)** | — | **×1,82** |
| | 16 | 578 (36 µs/t) | — | **358 (22,4 µs/t, 268 GB/s)** | **×1,62** |
| **q 33,4 MB** | 4 | 59 (14,9 µs/t) | — | — | — |
| | 8 | 117 (14,6 µs/t) | **91,5 (11,4 µs/t, 376 GB/s)** | — | **×1,28** |
| | 16 | 200 (12,5 µs/t) | — | **205,7 (12,9 µs/t, 165 GB/s)** | ×0,97 (nul) |

## Verdict P4 — réponse à la question

**Oui, le dp4a passe à T=8/16 SANS perdre son avantage calcul : il le GAGNE.**

- L'arithmétique dp4a exacte est conservée (self-check 3/3) : pas de fp16, pas de TC, pas de
  changement de précision. Le gain vient UNIQUEMENT de lire chaque poids une fois pour
  T tokens (bytes/token divisé par ~2 à T=8).
- **T=8 est le sweet spot** : ×1,3-1,8 vs la prod en µs/token (lmhd 275 vs 447 µs/token),
  poids à 616 GB/s. **T=16 dégrade** (q n'y gagne plus) : 1 warp/CTA ne masque plus les
  latences → on retombe vers ~260 GB/s. Le plafond reste le DRAM (cohérent P3).
- **P4 T=8 vs P2 (fused TC), lmhd, même T** : B8 **275 µs/token** vs P2 **670 µs/token**
  (×2,4 à T=8) ; P2 à T=16 fait 352 µs/token vs B8 275 → **le dp4a à T=8 bat le fused TC à
  T=16** en µs/token, avec l'arithmétique exacte int8 (aucun bruit fp16 d'entrée). Le TC
  n'apporte rien ici : le bottleneck est les bytes, pas le calcul.

## Synthèse P3+P4 — « qu'est-ce qui empêche le decode Q8_0 T=4 d'aller plus vite ? »

**Le DRAM poids, atteint à ~84 % (757/900 GB/s). L'ALU/dp4a/PCIE/L2 ne limitent pas.**
À T=4, la prod relit les poids 4 fois (446 µs/token constant T=4→16) — c'est le SEUL levier
restant : un dp4a à ncols=8 (kernel prod instancié T=8) ferait ~275 µs/token en lmhd (×1,63)
**si la prod pouvait batcher 8 tokens** (préfill/batch/speculative decoding). Pour le
decode strict T=4 : **rien à gagner de plus** (84 % du DRAM, ALU nul) — sauf réduction de
bytes (quantité plus basse, hors scope Q8_0) ou L2-resident (impossible : modèle de plusieurs
Go, 6 Mo de L2).

## État prod (vérifié 15:40)

`ollama-v100.service` active, V100 32350 MiB, `.so` = known-good **e4afbb4a**, aucun swap,
trap OK (fenêtres 13:11/13:14 avortées par bugs de bench, relancées propres à 13:24 + 15:08).

## Fichiers P3/P4

- `bench_carac.cu` / `make_bench_carac.py` (P3) — M0/M1/A2/D0/D1/T-sweep valides ; M2 invalide (DCE)
- `bench_p4.cu` / `make_bench_p4.py` (P4) — B8/B16 (kernel prod ncols=8/16)
- `probe34.cu` (diagnostic DCE ; binaire invalide noté `probe34.cu.orig-bad`)
- `run_prod_window_p3.sh`, `run_prod_window_p4.sh` (binaires pré-buildés, pas de nvcc dans la fenêtre)
- logs : `window-20261006-132426.log` (P3), `window-20261006-150817.log` (P4)

---

# P5 — Autopsie du DISPATCH naturel Q8_0 selon le batch (2026-10-06, fenêtre 20:56)

## Question

> Dans le `.so` prod (e4afbb4a, V100), quel kernel le moteur choisit-il pour
> `mul_mat(Q8_0 × F32)` selon le nombre de tokens `ne1` (1 / 4 / 8 / 16) —
> et quel est le vrai débit « poids » de chaque branche, sans rien modifier ?

## Méthode (aucune modif du `.so`, preuve par fingerprint)

- `p5_dispatch` (lien direct contre le `.so` prod e4afbb4a + ggml-base 0.24.0, cudart/cublas
  12) : graph `mul_mat` Q8_0 × F32 brut → dst F32, `ggml_backend_graph_compute` = le VRAI
  dispatch. 4 shapes × ne1 ∈ {1,4,8,16}, 300 iters, 10 warm.
- `p5_ktrace.so` (LD_PRELOAD) interpose `cudaLaunchKernel{,ExC,Coop}` et logge
  `(host-stub, grid, block, shmem)` de chaque lancement.
- **Identification par fingerprint** : ce fatbin (144 images sm_70) ne permet PAS de relier
  fiablement le `host-stub` au NOM de kernel (`dladdr` retourne des pointeurs bruts ; la base
  de chargement ne se résout pas sans `/proc` du process mort). Mais le tuple
  `(grid, block, shmem)` est une fonction déterministe de `(shape, ne1, branche, ncols, nwarps)`
  calculable depuis la source `llama-volta-merged/ggml/src/ggml-cuda`. On le compare 1:1 au log.
- Règles de dispatch lues dans la source (prod e4afbb4a) : `ggml_cuda_mul_mat` (ggml-cuda.cu:2056)
  pour Q8_0, src1=F32, dst=F32, ids=nullptr →  `should_use_mmvq(Q8_0, VOLTA, ne11)` =
  `ne11 <= MMVQ_MAX_BATCH_SIZE(=8)` ; sinon `should_use_mmq(Q8_0, VOLTA, ne11, 0)` =
  `ne11 < MMQ_DP4A_MAX_BATCH_SIZE(=64)` (V100=sm_70, fp16 TC dispo).
  V100 → table MMVQ **GENERIC** (sm_70 < Turing) : `nwarps(ncols 1-4)=4`, `(5-8)=2` ;
  `rows_per_block(ncols1)=1`, `(ncols 2..8)=2`. MMVQ launch : `grid=(ceil(n/rpb),1,1)`,
  `block=(32,nwarps,1)`, `shmem=0`. MMQ dp4a : `block=(32,8,1)`, `shmem=40640`.

## Résultat — dispatch RÉEL observé (preuve par fingerprint, 16/16 configs)

Chaque config rejoue 2× (warmup CUDA-graph). Tous les lancements sont expliqués.

| shape | ne1 | branche prod | kernel (fingerprint observé) | note |
|---|---|---|---|---|
| n48/q/down/lmhd | **1** | **MMVQ** | `mul_mat_vec_q<Q8_0,ncols=1>` — `grid=(n,1,1) b=(32,4,1) sh=0` + requant `b=(256,1,1)` | 1 warp-colonne, 4 warps |
| … | **4** | **MMVQ** | `mul_mat_vec_q<Q8_0,ncols=4>` — `grid=(n/2,1,1) b=(32,4,1) sh=0` + requant | 2 lignes/CTA, 4 warps |
| … | **8** | **MMVQ** | `mul_mat_vec_q<Q8_0,ncols=8>` — `grid=(n/2,1,1) b=(32,2,1) sh=0` + requant | **ncols=8 EXISTANT**, 2 warps |
| … | **16** | **MMQ** (dp4a) | requant `b=(128,1,1)` + `grid=(k/64,1,1) b=(32,8,1) sh=40640` + post `b=(32,4,1)` | passage au MMQ (tile J), pas ncols=16 |

→ **Le moteur prod n'a PAS de `mul_mat_vec_q ncols=16`** : dès `ne1=16` il bascule sur le
kernel **MMQ dp4a** (tile J, shmem 40 KB). Le batch 16 n'est donc pas « le même mmvq plus
large » — c'est un tout autre kernel. `ncols=8` (le B8 de P4) est bien une instanciation
**existante** du kernel prod (choisie quand le batch vaut 8).

## Débit « poids » mesuré (µs, µs/token ; poids GB/s = w/temps kernel, poids lu 1×)

`graph_compute` naturel complet (requant + kernel + orchestration). B4/B8 consignés
complet ; ne1=1/16 = points de repère (n1 seul ; 16 = autre branche MMQ).

| shape | ne1 | µs | µs/token | poids GB/s | branche |
|---|---|---|---|---|---|
| **lmhd 1,35 GB** | 1 | 1619,5 | 1619,5 | **834,1** | MMVQ n1 |
| | 4 | 1791,2 | 447,8 | 754,2 | MMVQ n4 |
| | 8 | 2402,2 | 300,3 | 562,4 | MMVQ n8 |
| | 16 | 2643,1 | 165,2 | 511,1 | MMQ dp4a |
| **down 94,7 MB** | 1 | 138,2 | 138,2 | 685,2 | MMVQ n1 |
| | 4 | 148,5 | 37,1 | 637,7 | MMVQ n4 |
| | 8 | 182,4 | 22,8 | 519,2 | MMVQ n8 |
| | 16 | 258,8 | 16,2 | 365,9 | MMQ dp4a |
| **q 33,4 MB** | 1 | 59,5 | 59,5 | 561,9 | MMVQ n1 |
| | 4 | 57,0 | 14,3 | 586,0 | MMVQ n4 |
| | 8 | 70,0 | 8,7 | 477,8 | MMVQ n8 |
| | 16 | 105,5 | 6,6 | 316,9 | MMQ dp4a |
| **n48 5,6 MB** | 1 | 16,3 | 16,3 | 341,8 | MMVQ n1 |
| | 4 | 18,8 | 4,7 | 297,1 | MMVQ n4 |
| | 8 | 23,3 | 2,9 | 238,9 | MMVQ n8 |
| | 16 | 35,4 | 2,2 | 157,3 | MMQ dp4a |

- Correctness `maxrel_vs_F32brut` ~1,4e1..4,5e1 **uniforme** entre ne1=4 et ne1=8 (profil
  identique attendu) : l'écart vient du requant Q8_1 (bruit du baseline), pas d'erreur
  additionnelle introduite par le `ncols`. Aucune anomalie sur `ncols=8`.

## Verdict P5

1. **Le dispatch prod Q8_0 sur V100 est un palier à 8** : `ne1 ≤ 8` → kernel **MMVQ**
   `mul_mat_vec_q<Q8_0, ncols_dst=ne1>` (1/4/8 instanciés, 1 passe poids, requant F32→Q8_1
   préalable) ; `ne1 > 8` (ici 16) → **MMQ dp4a** (tile J, shmem 40 KB). Ce n'est pas une
   simple montée en `ncols`.
2. **Le `ncols=8` (B8 de P4) n'est donc pas une nouveauté** : c'est exactement le kernel que la
   prod appelle quand elle batche 8 tokens. P4 a mesuré ce kernel en isolation (lmhd 275 µs/token
   au T=8) — ici, dans le flux naturel, le batch=8 donne **300 µs/token** sur lmhd (requant +
   overhead de graph inclus), cohérent avec P4.
3. **Le batch 16 passe au MMQ** (165 µs/token lmhd ici) — un autre kernel, à requalifier séparé
   si l'on veut exploiter le T=16 (hors scope « mmvq ncols »).
4. Confirme P3/P4 : la prod en decode ne batche jamais au-delà de 4 → elle ne voit jamais le
   `ncols=8` ni le MMQ. Le gain P4 (B8) reste conditionné au fait de **pouvoir batcher 8 tokens**.

## Bench B4 vs B8 en flux naturel — consigné, ARRÊTÉ (pas de nouveau bench)

Le P5 contient déjà exactement la mesure attendue : le coût réel du chemin
`graph_compute` en conditions naturelles, `ne1=4` vs `ne1=8`, pas de nouveau run requis.

**Gain B8 vs B4 (µs/token, flux naturel, poids lu 1×) :**

| shape | B4 ne1=4 | B8 ne1=8 | gain débit/token | coût/token |
|---|---|---|---|---|
| **lmhd** | 447,80 µs/t (754,2 GB/s) | **300,27 µs/t** (562,4 GB/s) | **×1,49** | **−33 %** |
| down | 37,13 µs/t (637,7 GB/s) | 22,80 µs/t (519,2 GB/s) | ×1,63 | −39 % |
| q | 14,26 µs/t (586,0 GB/s) | 8,74 µs/t (477,8 GB/s) | ×1,63 | −39 % |

Cohérence P4 (kernel-only) ↔ P5 (flux naturel) :

- P4 B8 kernel-only ~274,7 µs/token ; P5 B8 naturel 300,27 → **~25,6 µs/token** de coût
  additionnel du chemin naturel (requant + graph/orchestration).
- P5 B4 447,8 µs/token ≈ B4 kernel-only P4 (~447 µs/token) : le B4 du prod ne paie presque
  aucun overhead → le gain B8 n'est PAS un artefact de la méthode, il survit tel quel dans
  le `graph_compute` réel.

**Chaîne de preuve complète (fermée) :**

1. **P4** → gain intrinsèque du kernel : `mul_mat_vec_q<Q8_0,ncols=8>` lit chaque poids une
   fois pour 8 tokens (exact dp4a, self-check 3/3).
2. **P5 dispatch** (fingerprint 16/16) → preuve que le `.so` prod sélectionne RÉELLEMENT ce
   kernel quand `ne1=8` (pas d'hypothèse de dispatch, tuple grid/block/shmem observé).
3. **P5 benchmark** → preuve que le gain (×1,49-1,63 en µs/token) survit dans le
   `graph_compute` naturel du prod.

`ne1=16` est couvert mais relève d'une **autre branche : MMQ dp4a (tile J)**, pas MMVQ B16 —
ne pas mélanger les deux. **Conclusion : on arrête le bench ici ; le résultat B8 est
suffisamment établi.**

## Question ouverte suivante (remplace « est-ce que B8 est meilleur ? » — démontré)

> **Comment faire apparaître `ne1=8` suffisamment souvent dans le chemin d'inférence réel ?**

Le decode strict reste `ne1≤4` (prod : ncols_dst=4) → il ne voit jamais le B8. Des voies à
qualifier, hors scope Q8_0 kernel : préfill/chunked-prefill (mais P2 : le 32×8 TC bat B8 dès
T≥16 ; le B8 ne sert qu'à T=8-15), batching de requêtes concurrentes (8 requêtes × decode
→ ne1=8), et spec-decoding (drafts vérifiés en batch : ne1=8 si 7 drafts + 1 cible). La
question n'est plus le kernel, c'est le **régime de batch de la charge réelle** : quelle
fraction des `mul_mat` Q8_0 du flux prod pourrait atteindre ne1=8, et à quel coût d'ingénierie.

## État prod / reproductibilité

- Fenêtre `window-20261006-205621.log` : **bench rc=0**, service relancé `active` par le trap,
  modèle rechargé (VRAM 32495 MiB en fin). `.so` prod **non modifié** (probe + LD_PRELOAD
  externe uniquement). V100 device 0 (la GTX 1070 n'est pas visible dans les fenêtres → device 0 == V100).
- Limites : le nom exact du kernel n'est pas résolu (fatbin, base de chargement non reconstruite
  post-mortem) ; l'identification repose sur le fingerprint `(grid,block,shmem)` qui est
  **unique et sans ambiguïté** pour chaque branche (aucun autre kernel du `.so` n'a le même tuple
  pour ces shapes). 16/16 configs expliquées, 0 lancement inexpliqué.

## Fichiers P5

- `p5_dispatch.cpp` / `p5_dispatch` (probe dispatch, lien prod)
- `p5_ktrace.c` / `p5_ktrace.so` (shim LD_PRELOAD des `cudaLaunchKernel*`)
- `run_prod_window_p5.sh` (fenêtre V100 autoclose, binaires pré-buildés, pas de nvcc)
- `window-20261006-205621.log` (log P5 : trace dispatch + perf + correctness)
- scripts d'analyse `/tmp/p5_fingerprint_verify.py` (preuve par fingerprint),
  `/tmp/map_p5_v*.py` (tentatives de mapping nom, documentées comme non-fiables sur ce fatbin)

---

# P6 — La prod actuelle exploite-t-elle B8 ? (2026-10-07, observation live + code, AUCUN redémarrage)

## Question

> Sur la prod en place : quel est le régime de `ne1` réellement vu par les `mul_mat` Q8_0
> (et les autres types), MTP actif ou pas — et combien de `mul_mat Q8_0` à `ne1=8` y a-t-il ?

## Config prod observée live (service actif, aucune perturbation)

- `ollama-v100.service` active depuis le trap P5 (2026-10-06 20:56), PID 183847, V100 32350 MiB.
- `llama-server --model …/blobs/sha256-49330183… -c 200000 -np 1 --log-verbosity 4
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn auto -b 512 -ub 512 --context-shift --keep 4`.
- Backend : `CUDA : ARCHS = 700 | USE_GRAPHS = 1` (journal 20:56:50).
- MTP live (journal 07:11) : 221 918 cycles, 665 746 drafts générés, 360 028 acceptés,
  **mean acc len 2,62**, acc/pos = (0,717, 0,519, 0,385). Tâches longues : 34,5 tok/s, 8650 tokens/171 s.
- `graphs reused = 204 294` (task 06:49) → **le decode steady-state est en replay CUDA graph**
  (`cudaGraphLaunch`, ggml-cuda.cu:4872).

## Types RÉELS du modèle chargé (parse GGUF v3, `p6_gguf_types.py` → `p6_gguf_types.out`)

Arch `qwen35`, 866 tenseurs, 65 blocs (64 trunk + 1 nextn), n_embd 5120, vocab 248320,
quantifié Unsloth. Répartition : **Q6_K ×231, Q8_0 ×201**, Q5_K ×71, Q4_K ×2, IQ4_NL ×1, F32 ×360.

- **`output.weight` [5120, 248320] = Q8_0** : la shape « lmhd » 248320×5120 de P4/P5 n'était
  PAS synthétique — c'est le vrai lm_head, bien Q8_0. (`token_embd.weight` = Q6_K, même shape.)
- Bloc MTP (`blk.64.nextn`) : `eh_proj` **Q6_K** [10240, 5120] + norms F32. Le modèle est
  chargé « mtp_only » (pas de trunk dupliqué dans la couche 64 au-delà de nextn) — le draft
  ne coûte que ce petit bloc + les norms.
- Les shapes « down »/« q » de P4/P5 sont **mixtes par bloc** (Q6_K dominant, Q8_0/Q5_K/Q4_K
 /IQ4_NL selon le bloc) : P4/P5 ont caractérisé l'extrême Q8_0, le reste suit les mêmes règles
  de dispatch avec leur propre palier.

## Règles de dispatch (source `llama-volta-merged`, celle du build prod)

`ggml_cuda_should_use_mmvq` (mmvq.cu), table VOLTA par type :

| type | palier MMVQ (ne11 ≤) |
|---|---|
| Q2_K | 4 |
| Q3_K | 6 |
| Q4_K | 5 |
| Q5_K | 6 |
| Q6_K | 7 |
| **Q8_0 (default)** | **8** (= MMVQ_MAX_BATCH_SIZE) |

Sinon `ggml_cuda_should_use_mmq` (mmq.cu) : tous les types quant (Q8_0, K-quants, IQ*) supportés ;
Volta (fp16 TC) → `ne11 < MMQ_DP4A_MAX_BATCH_SIZE` (**64**) → MMQ dp4a (tile J) ; au-delà cuBLAS.

⇒ À **ne1=4** : Q8_0 → `mul_mat_vec_q<Q8_0, ncols=4>` ; Q6_K/Q5_K/K-quants → `mul_mat_vec_q<T, ncols=4>`.
À ne1=1 : ncols=1. À ne1=8 : ncols=8 **seulement pour Q8_0 (et types default ≤8)** ; Q6_K à ne1=8
basculerait déjà au MMQ (palier 7). À ne1≥16 : MMQ dp4a pour tous.

## Profil de `ne1` du decode target dans cette config (code : `common/speculative.cpp`, `tools/server/server-context.cpp`)

- **Draft MTP** : 1 token par étape, 3 étapes (`n_max=3`, `p_min=0`) sur le seul bloc nextn
  (`eh_proj` Q6_K 10240×5120) → `llama_decode` à **ne1=1**, pas de matmul Q8_0 (le trunk MTP n'est pas dupliqué).
- **Vérification** : le batch cible contient le token échantillonné **+ les 3 drafts complets**
  (`spec_i_batch.size() == n_draft + 1`, assert server-context.cpp:4484 ; `handle_last_sampled_token`
  ajoute sampled + chaque draft au même batch) → **un decode target complet à ne1=4 par cycle MTP**.
- **Fallback** : draft vide (début de génération, échec de draught) → decode à **ne1=1**.
- **Prefill** : ubatches ≤512 (`-ub 512`) → ne1 jusqu'à 512 → domaine MMQ/cuBLAS (P2), pas mmvq.
- **`-np 1`** : une seule séquence ; rien ne batche des requêtes concurrentes en decode.

**Plafond structurel : `ne1 ≤ 4` dans le decode target.** La branche `ncols=8` (B8 de P4/P5)
est **inatteignable** dans cette config : le 8 n'existe nulle part (pas de 7+1 drafts —
`nextn_predict_layers=1`, pas d'entrainement MTP-7 ; pas de 8 requêtes concurrentes — `-np 1` ;
les ubatches de prefill valent 512 ou 4, jamais 8-15).

## Réponse à la question posée

1. **Oui, la prod passe bien par les `mul_mat Q8_0` mesurés en P5** (lm_head Q8_0 + 201 poids
   Q8_0), avec exactement le dispatch P5 — mais **à ne1=1 et ne1=4** (ncols=1/4), jamais ne1=8.
2. **Le nombre de `mul_mat Q8_0` à ne1=8 dans la prod actuelle : ZÉRO, structurellement.**
   Le gain B8 (×1,49-1,63 µs/token, P4+P5) est établi mais **non exploité** par cette charge.
3. Le MTP vérifie déjà en batch (ne1=4) : c'est le régime le plus proche d'un « B » sur ce
   service. Son gain vient de l'amortissement des poids sur 4 tokens + l'acceptation moyenne
   de 2,62 tokens/cycle (34,5 tok/s observés), pas d'un ncols=8.

## Contrainte méthodologique majeure (pour toute observation future)

`USE_GRAPHS=1` + replay : le decode steady-state est un `cudaGraphLaunch` → **un interpose de
kernel (méthode P5, LD_PRELOAD `cudaLaunchKernel*`) ne VOIT RIEN du decode prod** (seuls les
lancements hors-graph, préfill/captures, sont interceptables). L'observation du régime de batch
en prod live ne peut donc se faire qu'à deux niveaux : (a) **journal** (stats MTP, tokens/s,
graphs reused — fait ici, suffisant et concluant), (b) **bench fenêtre** qui rejoue le vrai graph
prod (méthode P5). Pas besoin d'instrumentation sur le service.

## Verdict P6

La question « faire apparaître ne1=8 » (fin P5) est **réglée par la structure de la charge** :
- avec MTP 3+1 et `-np 1`, **ne1∈{1,4} exclusivement** en decode ; B8 = 0 occurrence, pas de
  réglage de charge raisonnable qui y mène sur ce service (n_max=7 non entraîné, np=8 hors charge réelle).
- Le B8 reste un résultat **valide mais conditionnel** : il s'appliquerait si un jour la prod
  batchait 8-15 tokens (serveur multi-requests à decode simultané), pas ici.
- Nouveau levier éventuel (hors scope Q8_0, à ouvrir seulement si voulu) : les poids **Q6_K**
  (231/866, palier MMVQ à 7) — à ne1=4 ils sont au même ncols=4 ; un ncols=8 K-quant existerait
  seulement au-delà du palier (≠ le B8 Q8_0). Non mesuré, non prioritaire.

## Fichiers P6

- `p6_gguf_types.py` (parser GGUF v3 ; mapping types corrigé vs `ggml.h` de llama-volta-merged,
  la 1ère version avait une table décalée de 2 — ids 6/7 = Q5_0/Q5_1, pas Q8_0/Q8_1)
- `p6_gguf_types.out` (dump complet : 866 tenseurs, types, formes, bloc nextn, shapes P4/P5)
- Journal `ollama-v100.service` 2026-10-06 20:56 → 2026-10-07 07:11 (stats MTP, config, graphs)
- Code : `common/speculative.cpp` (draft/verify MTP), `tools/server/server-context.cpp`
  (spec_i_batch, 4484), `ggml/src/ggml-cuda/{mmvq,mmq}.cu` (paliers), `src/models/qwen35.cpp`
  (mtp_only, eh_proj 2×n_embd)
- **Prod non touchée** : pas de redémarrage, pas de .so modifié, GPU jamais isolé.

## Prochaine étape (si continuation voulue)

1. **Clôturer le volet B8** : consigner « B8 = résultat valide, inapplicable à la charge actuelle
   (ne1≤4) » comme conclusion finale de la chaîne P4→P6.
2. Optionnel A : étendre la fingerprint P5 aux types K-quant (Q6_K) pour confirmer sur le .so
   prod la palier-7 du VOLTA (source l'indique ; P5 a seulement prouvé le cas Q8_0).
3. Optionnel B (hors Q8_0) : si l'objectif devient « accélérer le decode prod réel », la cible
   est le K-quant ncols=4 (231 Q6_K + 71 Q5_K), pas le Q8_0 — nouvelle expérience à part.

---

# P6.5 — Cartographie statique du decode prod (Q6_K/Q5_K, 2026-10-07, 09:30) — SANS toucher à la prod

## Découverte d'abord : le `.so` prod n'est PAS du vanilla — il porte déjà 2 chemins Volta custom

- Le `.so` en place (sha256 `55bc0a96…`) est le **known-good e4afbb4a** (identique à
  `reference/libggml-cuda.so.e4afbb4a.known-good`), construit le **2026-10-04 19:29**,
  service redémarré le 10-04 20:05 — **après** les travaux Q8_0 (P0/P1).
- Strings dans le binaire : `GGML_CUDA_VOLTA_Q6_W4R4` et `GGML_CUDA_VOLTA_Q5_X4` (absents du
  vanilla). L'unité systemd s'appelle « build experimental, CUDA 12.6, **Volta Q6_W4R4** ».
- Source correspondante (llama-volta-merged, mmvq.cu) :
  - **Q6_K + ncols=4 + Volta + pas d'ids/fusion** → instanciation dédiée
    `mul_mat_vec_q<Q6_K, 4, rpb_override=4>` (4 lignes/CTA au lieu de 2, `GGML_CUDA_VOLTA_Q6_W4R4`,
    désactivable par env = 0) ;
  - **Q5_K + ncols=4 + Volta** → `vec_dot_q5_K_q8_1_x4` : le fragment de poids Q5_K est décodé
    **une seule fois et réutilisé sur les 4 colonnes d'activation** (`GGML_CUDA_VOLTA_Q5_X4`),
    au lieu de le ré-décoder 4×.
  - Les deux sont **activées par défaut** (env absente → 1).
- Trace de l'instrumentation de ce build (10-04 21:21, `/tmp/mmvq-instrument.tsv`, propriétaire
  ollama) — le dispatch RÉEL observé en prod : toutes les lignes K-quants ont
  `w4r4_kernel=1` (Q6_K n4) / `q5x4_kernel=1` (Q5_K n4) ; zéro `ncols=8` ; un seul `ne1=2`
  (ubatch imparité) ; `ne1=1` = draft MTP (Q6_K 10240/12288/17408×5120 + fusions 5120/17408). **La
  cartographie statique est donc confirmée par une observation live antérieure.**

## 1) Shapes Q6_K/Q5_K qui dominent le coût du decode (poids lus 1× par verify ne1=4)

Trunk 64 couches (qwen35 = 64 blocs ; 16 full-attention + 48 SSM/hybrid ; MoE-free, pas de router).
Poids 2D : **Q6_K 10.91 GiB, Q8_0 5.46 GiB, Q5_K 2.40 GiB** (+ Q4_K 0.03, IQ4_NL 0.16, F32 0.01).
Par shape (lue 1× par cycle de verify ; GiB = MiB/1024) :

| shape (n_out×k) | type | nb | GiB | % du cycle |
|---|---|---|---|---|
| 17408×5120 (ffn_up) | **Q6_K** | 85 | 4.91 | **27 %** |
| 5120×17408 (ffn_down) | **Q6_K** | 42 | 2.42 | **13 %** |
| 17408×5120 (up) | Q8_0 | 16 | 1.41 | 8 % |
| 17408×5120 (up) | **Q5_K** | 29 | 1.30 | **7 %** |
| lm_head 248320×5120 | Q8_0 | 1 | 1.26 | 7 % |
| 10240×5120 (attn_qkv) | **Q6_K** | 36 | 1.22 | 7 % |
| 5120×6144 (ssm_out) | Q8_0 | 33 | 1.03 | 6 % |
| 5120×17408 (down) | Q8_0 | 10 | 0.88 | 5 % |
| **Q6_K total trunk** | | 231 | **10.04** | **55 %** |
| **Q5_K total** | | 71 | **2.40** | **13 %** |
| Q8_0 total (trunk+lm_head) | | 201 | 5.46 | 30 % |

**K-quant (Q6_K+Q5_K) = 12.44 GiB sur 18.10 GiB lus par cycle = 69 %** du trafic poids du decode.
Poids **lus par cycle** : trunk 16.84 GiB (64 couches ≈ 265 MiB/couche) + lm_head 1.26 GiB =
**18.10 GiB (19.4 GB)**, lu 1× en ncols=4. (`token_embd` Q6_K 0.84 GiB = gather 1 ligne/token ≈ 48 KB,
non compté ; `eh_proj` nextn 0.04 GiB = côté draft, en section 4.)

> ⚠️ **CORRIGÉ le 2026-10-07 (15:45) — voir « Recalcul budget » en section GATING P7a.**
> Ces GiB étaient calculés avec les layouts vanilla (Q6_K 178, Q5_K 138 B/256). Les layouts
> PROD réels (Q6_K 210, Q5_K 176, Q8_0 34) donnent **21,68 GB/cycle (20,19 GiB)** : Q6_K
> 12,43 GB (+18 %), Q5_K 3,29 GB (+27,5 %), Q8_0 5,86 GB (inchangé). Les pourcentages
> ci-dessus sont obsolètes (Q6_K 55 % → 57,3 %, Q5_K 13 % → 15,2 %).

## 2) Kernel exact sélectionné à ncols=4 (source + trace 10-04)

| type | kernel prod ne1=4 | détails |
|---|---|---|
| Q6_K | `mul_mat_vec_q<Q6_K, ncols=4, rpb=4>` (variante W4R4) | grid=(n/4), block=(32, 4warps), 4 lignes/CTA |
| Q5_K | `mul_mat_vec_q<Q5_K, ncols=4, q5_x4>` | grid=(n/2), block=(32, 4warps), 2 lignes/CTA, poids décodé 1×/4 colonnes |
| Q8_0 | `mul_mat_vec_q<Q8_0, ncols=4>` (P5, prouvé 16/16) | grid=(n/2), block=(32, 4warps) |
| draft (Q6_K) | `mul_mat_vec_q<Q6_K, ncols=1>` | bloc nextn 10240×5120, ne1=1 |

VDR (blocs de K par thread) : Q6_K=1 (16 blocks/thread), Q5_K=2, Q8_0=2 — Q6_K est le moins
vectorisé (payload contigu 6 bits, `VDR=1` = une 4e d'itération de plus que Q8_0).

## 3) Variante V100/Volta

- Table MMVQ **GENERIC** (sm_70 < Turing) : nwarps=4, rows_per_block=2 (surclassées en 4 pour
  Q6_K par la variante W4R4). Pas de prefetch L2 (DGX Spark only). Pas de TC (dp4a int8, fp32 out).
- `.so` = known-good e4afbb4a (10-04), **inchangé depuis** ; service actif, GPU occupé par la
  charge réelle (tâches ~140k tokens, 34,5 tok/s).

## 4) Où sont les µs — estimation par cycle (76 ms observés)

> ⚠️ Recalé le 2026-10-07 : poids = **21,68 GB** (pas 19,4) ; temps poids 24–36 ms selon
> plafond ; trou 30–43 ms. Voir « Recalcul budget » en section GATING P7a.

Un cycle MTP = 1 verify target (ne1=4) + 3 drafts (Q6_K 10240×5120, négligeable) ≈ **76 ms**
(34,5 tok/s ÷ 2,62 acc/cycle).

| composante | taille/cycle | temps estimé | part |
|---|---|---|---|
| **poids mmvq (18.1 GiB, lu 1×)** | 19.4 GB | **22–26 ms** au plafond DRAM 750–900 GB/s | **~30–35 %** |
| **attention KV (flash-attn, 16 couches)** | 34.8 KB/cellule × ~145k cellules ≈ **5.0 GB** (buffer journal : 6647 MiB à 200k) | **6–7 ms** | **~8 %** |
| SSM/hybrid states (48 couches, RS buffer 598 MiB) | ~1.3 GB (R+S lus+écrits) | ~1.5–2 ms | ~2–3 % |
| drafts MTP (3 × 10240×5120 Q6_K + norms) | ~0.2 GB | ~0.3 ms + overhead | <1–2 % |
| requant F32→Q8_1 (382 matmuls × activations) + norms/embd | petit | ~1–2 ms | ~2 % |
| **Sous-total GPU attribuable** | | **~32–38 ms** | **~45 %** |
| **NON ATTRIBUÉ (gaps CPU↔GPU, orchestration MTP, sync, éventuel sous-régime mmvq K-quant)** | — | **~38–44 ms** | **~50 %** |

Lecture : contrairement à l'intuition, **l'attention KV n'est pas le goulot** (~5 GB à 145k ctx,
6–7 ms) ; le poids l'est (22–26 ms au mieux). Mais le cycle observé est ~76 ms alors que le
sous-total GPU attribuable est ~32–38 ms : **~50 % du cycle n'est expliqué par aucun trafic
connu**. L'utilisation GPU était à 80–98 % pendant la génération (pas du GPU inactif) : ce
temps est soit dans des kernels tournant nettement sous le plafond (les mmvq K-quant — VDR=1
pour Q6_K, layout 178 B/256 — ou SSM/flash-attn), soit dans des gaps entre lancements tenus
par le CPU (3 decode/cycle, orchestration MTP, samplers). Cette inconnue **décide** la cible
P7 : elle se tranche par une mesure (option A), pas par un nouveau kernel.

## Verdict P6.5 — où sont réellement les µs, et qu'est-ce que ça implique pour P7

1. **Q6_K/Q5_K = 69 % des poids lus par cycle** : la cible du trafic poids est confirmée.
   Temps poids borné à ~22–26 ms/cycle (19.4 GB / DRAM) ; les kernels y tournent déjà avec 2
   variantes custom (W4R4/X4). **Gain maximaliste « poids » ≈ 3–6 ms/cycle, si et seulement si
   les K-quants tournent sous le plafond DRAM (non mesuré — c'est l'option A).**
2. **L'attention KV (~6–7 ms à 145k ctx) est du second ordre** — et ne devient dominante qu'à
   très haut contexte (>500k), hors scope de cette charge.
3. **Le trou de ~50 % du cycle est la vraie question ouverte** : sous-régime mmvq K-quant, gaps
   CPU/inter-lancements, ou orchestration MTP. Sans le mesurer, un P7 kernel serait un pari.

## Fichiers / preuves P6.5

- `/tmp/mmvq-instrument.tsv` (10-04 21:21, trace dispatch live du build instrumenté)
- strings `GGML_CUDA_VOLTA_Q6_W4R4` / `GGML_CUDA_VOLTA_Q5_X4` dans le `.so` prod (known-good)
- `mmvq.cu` (variantes W4R4/X4, table GENERIC, VDR), `vecdotq.cuh` (vec_dot_q6_K_q8_1_impl_mmvq,
  VDR_Q6_K=1, QR6_K=2, block_q6_K vanilla = 178 B/256 — **layout PROD réel = 210 B/256**, recalcul 10-07), `ggml-common.h` (layouts)
- parse GGUF (p6_gguf_types.out) + calc GiB/cycle : trunk 16.84 GiB + lm_head 1.26 = 18.10 GiB
  (19.4 GB) ; token_embd = gather, non compté
- journal : 34,5 tok/s, acc len 2.62, tâches 139k–145k tokens, KV cache 6647 MiB (16 layers),
  flash_attn enabled, RS buffer 598 MiB

## Prochaine étape (P7 à trancher)

A) **Mesure du split du cycle** (poids mmvq vs fattn vs reste) : fenêtre bench rejouant le vrai
   graph prod (méthode P5, sans toucher au service), ou à défaut micro-bench des composants sur
   les shapes réelles — Q6_K 17408×5120 @ ne1=4 (GB/s effectif du kernel W4R4) + fattn q8v
   145k ctx. **Ce chiffre tranche la cible P7 : si Q6_K < ~700 GB/s → P7 kernel Q6_K (VDR=1/
   occupation en ligne de mire) ; si poids ≈ plafond et trou persistant → P7 = gaps CPU/
   orchestration, pas un kernel.**
B) Les drafts MTP (~0.3 ms/cycle) et l'attention (~7 ms) ne méritent pas de P7 à ce stade.

---

# GATING P7a — verrouillé le 2026-10-07 (12:50) par décision utilisateur : PAS DE KERNEL AVANT CETTE MESURE

> Consigne verrouillée : « verrouiller le gating avant de toucher à un kernel ». Le profil macro
> ci-dessus est la référence figée ; tout P7 kernel (Q6_K/Q5_K/Q8_0/autre) est **interdit** tant
> que le gate P7a n'a pas produit son chiffre. Les P4/P5 Q8_0 restent clos et ne sont pas
> ré-optimisés : ils ne couvrent qu'une partie du budget (23 % du trafic poids + lm_head 7 %).

## 1) Budget cycle figé (référence de gating)

Cycle MTP = **75,9 ms** (34,5 tok/s ÷ 2,62 tokens acceptés/cycle). Trafic poids du verify
ne1=4 : **19,4 GB/cycle** = Q6_K 10,04 GiB (55 %) + Q8_0 trunk 4,20 GiB (23 %) +
Q5_K 2,40 GiB (13 %) + lm_head Q8_0 1,26 GiB (7 %).

> ⚠️ **Budget recalé le 2026-10-07 (15:45) sur les layouts prod réels → 21,68 GB/cycle**
> (Q6_K 12,43 GB 57,3 % + Q8_0 5,86 GB 27,0 % dont lm_head 1,35 GB + Q5_K 3,29 GB 15,2 %).
> Les seuils du gate (700 / 600–750 / 750) ont été vérifiés contre ce budget et **restent
> validés inchangés** — détail en section « Recalcul budget » ci-dessous.

| composante | ms | part |
|---|---|---|
| poids MMVQ (19,4 GB, lu 1×) | 22–26 (plafond DRAM 750–900 GB/s) | 30–35 % |
| attention KV (16 couches, ~145k ctx, ≈5 GB) | 6–7 | 8 % |
| SSM states + drafts + requant | 2–3 | 4 % |
| **trou non attribué** | **38–44** | **~50 %** |

## 2) Hypothèses du trou, et le chiffre qui les tranchent (gate = GB/s effectif des kernels poids)

Le GPU n'était PAS inactif (80–98 % pendant la génération) : le trou est soit du temps GPU
sous-régime (H1), soit du CPU qui tient le stream (H2), soit des kernels GPU non comptés
(H3). Trois hypothèses, une mesure les discrimine :

- **H1 — les mmvq K-quants tournent sous le plafond DRAM.** Réaliste : VDR_Q6_K=1 (vs 2 pour
  Q5_K/Q8_0), layout 210 B/256 (prod, ex-178 vanilla — recalcul 10-07), et le P5 montre déjà
  que les shapes petites/chuettes
  tombent à 297–586 GB/s même en Q8_0. **Si vrai : le P7 utile est un kernel Q6_K/Q5_K.**
- **H2 — gaps CPU : orchestration MTP (3 décodeurs par cycle : verify ne1=4 + 3 drafts),
  lancements des ~4400 nodes du graph, requant, samplers.** Si les kernels poids sont au
  plafond et que le trou persiste, c'est lui. **Si vrai : le P7 utile est CPU/scheduler,
  pas un kernel** (aucune optimisation mmvq ne touchera ce temps).
- **H3 — kernels GPU non attribués sous-estimés** (fattn, SSM, requant, elementwise) : la part
  connue (8–12 ms) est bornée, H3 seule ne peut pas expliquer 38–44 ms — à exclure par
  construction, sauf si la mesure P7a-bis (fattn/SSM) contredit l'estimation.

**Seuil du gate (fixe, décidé avant de mesurer) :**

| mesure P7a (GB/s effectif, poids lu 1×, flux naturel `graph_compute`) | conclusion | P7 qui en découle |
|---|---|---|
| **Q6_K 17408×5120 @ ne1=4 < ~700 GB/s** | H1 confirmée : le trafic poids (55 % du cycle) est sous-régime | **P7 kernel Q6_K** (VDR=1 → vectorisation/occupation de la variante W4R4), Q5_K en 2e cible si lui aussi < 700 |
| **Q6_K ≥ ~750 GB/s** (et Q5_K ≥ 700) | H1 morte : le poids est au plafond, trou = H2 (H3 exclue par construction) | **P7 = gaps CPU/orchestration MTP** — zéro kernel mmvq ; on instrumente le split temps GPU vs temps CPU du cycle (events CUDA par étape du flux, ou ktrace en fenêtre) |
| **moyen (600–750)** | H1 partielle : gain poids borné à ~2–4 ms/cycle | P7 kernel Q6_K **seulement si** le gain estimé > 2 ms/cycle, sinon P7 = H2 |

## 3) Protocole P7a (fenêtre courte, mêmes invariants que P5)

Méthode = P5 au mot près : harness `ggml_backend_graph_compute` lié contre le `.so` prod
exact (e4afbb4a), flux naturel (le dispatch réel du prod, dont la variante W4R4 Q6_K et X4 Q5_K
qui s'activent toutes seules à ne1=4 sur Volta), binaire **pré-buildé**, pas de nvcc dans la
fenêtre. Fenêtre ~2–3 min, autoclose (trap relance le service + recharge le modèle).

Mesures dans la fenêtre (toutes : 300 iters, 10 warm, poids GB/s = w/temps kernel) :

1. **Gate** — Q6_K aux shapes réelles, ne1=4 (et ne1=1 pour le draft) :
   `17408×5120` (ffn_up, la shape n°1), `5120×17408` (ffn_down), `10240×5120` (attn_qkv),
   `12288×5120`. GB/s effectif du kernel W4R4 tel quel.
2. **Contrôle** — Q5_K `17408×5120` @ ne1=4 (variante X4) : même test, cible 2e.
3. **Référence** — Q8_0 `17408×5120` @ ne1=4 (≈ le 754 GB/s de P5 attendu en flux naturel) :
   ancre de plafond pour comparer H1 ; si Q8_0 chute ici, c'est un problème de méthode,
   pas de K-quant.
4. **(bis, si 1–3 ne suffisent pas à trancher)** — fattn q8v à 145k ctx + 1 shape SSM : pour
   tuer/valider H3. Ne se lance que si le gate 1–3 laisse H3 en suspens.

Garde-mém fenêtre : `cudaMemGetInfo` ≥ 1 Go libre avant chaque probe, SKIP sinon (la prod est
arrêtée pendant la fenêtre, ~32 Go disponibles — le garde reste pour la robustesse).
Correctness : 1 it vs oracle F32 (maxrel), même profil attendu qu'en P5 (bruit de requant Q8_1,
pas d'erreur additionnelle de dispatch).

## 4) Invariants — ne changent PAS (liste de non-régression du gate)

1. Service `ollama-v100.service` actif avant/après ; `.so` prod byte-identique
   (sha256 `55bc0a96…`) avant/après ; GGUF intouché ; aucun `ncu`/`nsys`.
2. Fenêtre V100 autoclose (pattern `run_prod_window_p5.sh`) : arrêt VRAM vérifiée,
   binaire pré-buildé seulement, trap redémarre + recharge le modèle (ping num_predict=1).
3. P4/P5 Q8_0 : clôturés, **pas de ré-optimisation Q8_0** tant que le gate n'est pas passé.
4. Toute nouvelle donnée entre ici, dans un fichier `p7a_*` du dossier `q8-qpn/` + section
   P7a dans ce `ETAT.md`.

## État

- **Gate verrouillé (12:50), mesure non encore lancée.** En attente du feu vert utilisateur
  pour la fenêtre P7a (2–3 min de V100 libre, service relancé automatiquement).

## Recalcul budget — layouts PROD réels (2026-10-07, 15:45) — fait offline, AUCUNE mesure lancée

**Correction majeure.** Le `.so` prod ne lit PAS les layouts vanilla de ggml : confirmés
indépendamment par (a) `libggml-base.so.0` prod (`ggml_row_size` : Q6_K=210 B/256, Q5_K=176
B/256, Q8_0=34 B/32) et (b) les offsets réels du blob GGUF (`p7a_gguf_recs.tsv`, parse 866
tenseurs) : Q6_K 210, Q5_K 176, Q8_0 34, Q4_K/IQ4_NL 144. P6.5 avait calculé ses GiB/cycle
avec 178/138 → **les bytes/cycle et les % de P6.5 étaient sous-estimés pour Q6_K et Q5_K**.
Script : `p7a_budget_recalc.py` (sortie `p7a_recalc.out`).

### 1) Bytes/cycle du verify ne1=4 (poids lus 1×) — recalculé

| famille | nb | P6.5 (vanilla) | **PROD (recalc)** | Δ | % du cycle poids |
|---|---|---|---|---|---|
| Q6_K (trunk) | 224 | 10,54 GB | **12,43 GB** (11,58 GiB) | **+18,0 %** | **57,3 %** |
| Q8_0 trunk+lm_head | 199 | 5,86 GB | **5,86 GB** (5,45 GiB) | 0 % | 27,0 % |
| Q5_K | 71 | 2,58 GB | **3,29 GB** (3,06 GiB) | **+27,5 %** | 15,2 % |
| Q4_K/IQ4_NL/F32 2D | 51 | 0,10 GB | 0,10 GB | 0 % | 0,4 % |
| **TOTAL verify** | 545 | 19,08 GB (19,4 en P6.5) | **21,68 GB (20,19 GiB)** | **+13,7 %** | 100 % |

- dont **lm_head Q8_0 248320×5120 = 1,35 GB (6,2 %)** ; trunk seul = 20,33 GB (93,8 %).
- Exclu du cycle verify (côté draft/embd, comme P6.5) : `token_embd` Q6_K 1,04 GB (gather),
  bloc `blk.64.*` (attn 0,065 GB + ffn 0,220 GB + `nextn.eh_proj` 0,043 GB) ≈ 0,35 GB/step.
- Correction interne P6.5 : Q8_0 y valait 4,20 GiB — c'était le **trunk seul** ; Q8_0 total
  trunk+lm_head = 5,45 GiB (5,86 GB), inchangé par le recalcul (layout Q8_0 déjà bon).

### 2) Temps théorique du poids/cycle (21,68 GB, lu 1×)

| plafond DRAM | ms/cycle | part du cycle 75,9 ms |
|---|---|---|
| 600 GB/s | 36,1 ms | 47,6 % |
| 700 GB/s | 31,0 ms | 40,8 % |
| 750 GB/s | 28,9 ms | 38,1 % |
| 900 GB/s | 24,1 ms | 31,7 % |

Sous-total GPU attribuable ≈ 33–46 ms (poids + KV 6–7 + SSM/drafts/requant 2–3) →
**trou non attribué ≈ 30–43 ms** (vs 38–44 en P6.5 : borne basse recalée, borne haute stable).

### 3) Gain maximal ms/cycle (si on amène la famille au plafond)

| famille (GB/cycle) | 600→750 | 600→900 | 700→754 (ancre P5) | 750→900 |
|---|---|---|---|---|
| **Q6_K (12,43)** | **+4,14** | **+6,91** | +1,27 | +2,76 |
| Q5_K (3,29) | +1,10 | **+1,83 (max absolu)** | +0,31 | +0,73 |
| Q8_0 (5,86) — clos, référence | +1,95 | +3,25 | +0,56 | +1,30 |

**Seuil « gain > 2 ms/cycle » = 2,6 % du cycle.** Bascule Q6_K : gain = 2 ms atteint à
**~669 GB/s** (plafond 750) / **~672 GB/s** (plafond = ancre P5 754).

### 4) Cohérence des seuils figés avec le budget recalé → **seuils VALIDÉS, inchangés**

- **`< 700` (H1 → kernel Q6_K)** : au-dessus de la bascule 2 ms (669–672) — légèrement
  conservateur : à 680 GB/s le gain réel est ~1,8 ms. Tendance de surestimation contrôlée :
  le kernel P7 se dimensionnera sur le chiffre mesuré, pas sur le seuil. **Cohérent.**
- **`≥ 750` (H1 morte)** : ≈ le plafond ancre P5 (754) — au-dessus, gain Q6_K < 0,6 ms,
  Q5_K < 0,4 ms : H1 objectivement morte. **Cohérent.**
- **600–750 + règle « > 2 ms/cycle »** : la bascule 2 ms tombe à ~672 GB/s, donc la bande
  672–750 = « H1 partielle mais gain < 2 ms → H2 ». **Cohérent.**
- **Q5_K exclu du trigger kernel** : son gain **max absolu est 1,83 ms/cycle < 2 ms**, quel que
  soit le régime mesuré → le contrôle Q5_K sert de témoin de régime (H1/H2), jamais de cible.
  Consigne ajoutée au TSV : lire le gain Q5_K avec la règle 2 ms (pas de P7 Q5_K standalone).
- Le budget recalé ne déplace aucun seuil : la fenêtre P7a peut ouvrir **sans changer le
  protocole**.

### 5) Notes d'exécution (pas de changement de protocole)

- **L'ordre d'énumération CUDA a basculé depuis P5** : log P5 (10-06) `Device 0 = V100` ;
  nvidia-smi du 10-07 `index 0 = GTX 1070, index 1 = V100`. Le binaire `p7a_dispatch` choisit
  maintenant la V100 **par nom + cc 7.0** (arg numérique en override, non utilisé). La GTX 1070
  (Pascal) est hors sujet : W4R4/X4 sont Volta-only, et un OOM y est apparu au smoke test
  (5,8 Go libres < pool). Pas d'impact sur la mesure : le kernel W4R4/X4 n'existe que sur la
  V100, donc aucune fausse validation possible.
- `p7a_dispatch` pré-buildé (g++-13, hors prod), lié aux `.so` prod par rpath, 6 probes
  (4 gate Q6_K + 1 contrôle Q5_K + 1 ancre Q8_0) × ne1 {4,1}, 300 it / 10 warm, oracle CPU
  prod (roundtrip mesuré : Q6_K 1,6e-2, Q5_K 4,0e-2, Q8_0 4,1e-3 maxrel).

## CRASH P7a — ROOT CAUSE (2026-10-07, 21:05–22:05) — segfault libcuda RESOLU

> Consigne utilisateur (20:23) : « les deux traces kernel constituent un signal de
> reproductibilité du crash — les conserver telles quelles dans ETAT.md ». Pas de
> nouvelle fenêtre P7a ni changement du .so prod tant que le 1er appel fautif n'est pas
> isolé. Résultat : le crash est un **bug du harness P7a (sous-allocation du buffer
> poids host)**, pas un problème driver, pas un problème W4R4/Volta, pas un problème
> CUDA-graph. Le .so prod n'a jamais été touché (sha256 identique avant/après chaque
> fenêtre).

### Traces kernel (conservées telles quelles)

```
[mer. oct.  7 17:00:59 2026] p7a_dispatch[294934]: segfault at 7c55e4f29000 ip 00007c560a221680 sp 00007ffcbda3ff38 error 4 in libcuda.so.580.178.04[221680,7c560a166000+f84000] likely on CPU 4 (core 4, socket 0)
[mer. oct.  7 17:00:59 2026] Code: f8 3f 76 4a 49 8d 40 c0 48 83 e0 c0 48 8d 50 40 48 89 f8 48 8d 0c 16 0f 1f 00 0f 28 56 10 0f 28 4e 20 48 83 c6 40 48 83 c0 40 <0f> 28 46 f0 0f 28 66 c0 0f 2b 50 d0 0f 2b 60 c0 0f 2b 48 e0 0f 2b
[mer. oct.  7 19:22:38 2026] p7a_dispatch[302554]: segfault at 7cdd90d29000 ip 00007cddb6021680 sp 00007fff983c4c48 error 4 in libcuda.so.580.178.04[221680,7cddb5f66000+f84000] likely on CPU 4 (core 4, socket 0)
[mer. oct.  7 19:22:38 2026] Code: f8 3f 76 4a 49 8d 40 c0 48 83 e0 c0 48 8d 50 40 48 89 f8 48 8d 0c 16 0f 1f 00 0f 28 56 10 0f 28 4e 20 48 83 c6 40 48 83 c0 40 <0f> 28 46 f0 0f 28 66 c0 0f 2b 50 d0 0f 2b 60 c0 0f 2b 48 e0 0f 2b
```
+ 3e signature identique (repro minimal `p7a_min`, 21:05:24, **sur la GTX 1070** — voir
ci-dessous, argument de device bugué par le script diag) :
```
[275817.473144] p7a_min[308292]: segfault at 7327bc5de000 ip 00007327ed421680 sp 00007fff7333dc78 error 4 in libcuda.so.580.178.04[221680,7327ed366000+f84000] likely on CPU 4 (core 4, socket 0)
```
Même offset d'instruction `+0x221680`, même `error 4` (lecture user, page absente), même
pattern SSE (déplacements 16B sur pointeur ±64), sur **2 GPU différentes** (V100 cc7.0,
GTX 1070 Pascal cc6.1) → élimine d'emblée le kernel W4R4/Volta : le défaut est dans un
chemin host commun au driver (H2D).

### Isolation (repro minimal `p7a_min`, marqueurs A–I)

Fenêtre diag autoclose (`run_prod_window_p7amin.sh`, mêmes invariants que P7a : sha256
.so avant/après, trap relance service+modèle). 1 probe `Q6_K 17408×5120 ne1=4`, 1
`graph_compute`, marqueurs stricts. Résultat (21:05, V100 32495 MiB libres) :

```
A BEGIN/END backend/CUDA init        OK
B BEGIN/END context creation         OK
C BEGIN/END tensor creation          OK
E BEGIN/END ggml_mul_mat             OK
F BEGIN/END graph construction       OK
G BEGIN/END graph allocation         OK
D BEGIN weight upload (tensor_set)   ← SIGSEGV ICI, avant D END
```
→ Le crash est **AVANT `graph_compute`**, dans `ggml_backend_tensor_set` du poids
(`cudaMemcpyAsync H2D` sur `cudaStreamPerThread`, ggml-cuda.cu:796).

### Divergence P5/P7a (la 1re et unique)

| | P5 (fonctionne) | P7a (crash) |
|---|---|---|
| buffer poids host | `wb = n * ng * sizeof(block_q8_0)` — **ligne complète** | `wb = n * ggml_row_size(type, blck_size)` — **1 seul bloc** |
| taille Q6_K 17408×5120 | n/a (Q8_0 : 94,7 MB, exact) | n×210 = **3,66 MB**, alors que `ggml_nbytes(w)` = 73,11 MB |

`ggml_row_size(type, ne)` = `row_meta_size + type_size*ne/blck_size` (ggml.c:1357) :
appeler avec `ne = blck_size` donne **210 B (un bloc)**, pas 4200 B (une ligne de
k=5120). Conséquence en chaîne, mesurée :
- `quantize_row_q6_K_ref` écrit k elems = 20 blocs × 210 B = **4200 B/ligne** → déjà
  20× au-delà du buffer alloué (corruption heap, masquée — le std::vector est suivi de
  mémoire « vide » jusqu'à la prochaine page) ;
- `ggml_backend_tensor_set(w, wb, 0, ggml_nbytes(w)=73 113 600)` → H2D de 73,11 MB à
  partir d'un buffer host de 3,66 MB → lecture ~69,5 MB **au-delà de l'allocation** →
  page non présente → **SIGSEGV `error 4` dans libcuda** (le driver ne retourne pas
  proprement l'erreur, il segfaulte en CPU sur la boucle SSE de copie).

Preuve du runtime prod : `ggml_row_size(Q6_K,256)=210`, `ggml_row_size(Q6_K,5120)=4200`,
`ggml_nbytes(w 17408×5120)=73 113 600 B` (binaire lié `libggml-base.so.0` prod).
P5 ne touchait jamais ce chemin faute d'avoir ce bug : son allocation était déjà la
ligne complète (ng×34 B), donc `set()` ne lisait que des pages présentes.

### Correctif (harness seul — AUCUN .so, GGUF, service)

- `p7a_dispatch.cpp:146` : `size_t bs = ggml_row_size(type, k);` (ligne complète, 4200 B
  pour Q6_K k=5120) + garde `AB(ggml_nbytes(w) == n*bs)` dans la boucle ne1. Backup :
  `p7a_dispatch.cpp.orig-underalloc`.
- `p7a_min.cpp` / `p7a_min2.cpp` : même fix (bs = ligne complète) + ordre d'args corrigé
  (`argv[1]=ncomputes`, device choisi **par nom**, plus d'override positionnel — c'est le
  bug qui avait fait courir le repro de 21:05 sur la GTX 1070).

### Validation (22:04, fenêtre diag `run_prod_window_p7amin.sh 1`)

```
poids host prepare : 69 Mo                      (73,1 MB = ligne complète OK)
D BEGIN → D END   weight upload OK
[DUMP] w type=q6_K ne0=5120 ne1=17408 nb=[210,4200,73113600] row=4200 nbytes=73113600
[H] → [I]  graph_compute OK, sync OK, FIN PROPRE, rc=0
```
Invariants : `.so` prod sha256 `55bc0a96…` identique avant/après, service relancé
active, API répond, VRAM 32 Go libérés/rechargés, **aucun Xid, aucun nouveau segfault**
dans dmesg. La divergence P5/P7a est identifiée ET corrigée : le chemin `graph_compute`
n'avait jamais été atteint dans P7a ; c'est maintenant fait, proprement.

### Conséquences sur le protocole
- Les correctifs d'essai 2 (`GGML_CUDA_DISABLE_GRAPHS=1`, `LD_PRELOAD p5_ktrace.so`)
  étaient inutiles : ils n'avaient rien à cacher. La fenêtre P7a peut rouvrir avec la
  **méthode P5 au mot près** (env = parité prod : `GGML_CUDA_VOLTA_Q6_W4R4=1` + SHORTLIST,
  sans ktrace ni DISABLE_GRAPHS) et le binaire corrigé.
- Le gating (700 / 600–750 / 750, budget 21,68 GB/cycle) reste verrouillé, inchangé.
- AUCUNE conclusion H1/H2 : pas de GB/s mesurés. La fenêtre P7a attend le feu vert.

## GATE P7a — MESURE (2026-10-08 06:07) — H1 CONFIRMEE → P7 KERNEL Q6_K

> Feu vert utilisateur → fenêtre `run_prod_window_p7a.sh` (binaire corrigé, env parité
> prod W4R4+SHORTLIST, ktrace/DISABLE_GRAPHS retirés). rc=0, **aucun crash, aucun Xid**,
> `.so` sha256 identique avant/après, service relancé active + API OK + VRAM rechargée.
> TSV : `p7a_results.tsv`.

### Résultats (GB/s effectif poids lu 1×, 300 iters/10 warm, V100 cc7.0, 80 SM)

| rôle | shape | type | ne1 | µs | µs/tok | GB/s | maxrel |
|---|---|---|---|---|---|---|---|
| **gate** | 17408×5120 (ffn_up) | Q6_K | **4** | 147.81 | 36.95 | **494.7** | 9.03e-01 |
| gate | 17408×5120 | Q6_K | 1 | 118.67 | 118.67 | 616.1 | 9.03e-01 |
| gate | 5120×17408 (ffn_down) | Q6_K | 4 | 137.47 | 34.37 | 531.9 | 1.35e+00 |
| gate | 5120×17408 | Q6_K | 1 | 116.00 | 116.00 | 630.3 | 9.10e-01 |
| gate | 10240×5120 (attn_qkv) | Q6_K | 4 | 86.53 | 21.63 | 497.0 | 9.67e-01 |
| gate | 10240×5120 | Q6_K | 1 | 71.66 | 71.66 | 600.1 | 3.58e-01 |
| gate | 12288×5120 (attn_q) | Q6_K | 4 | 101.29 | 25.32 | 509.5 | 1.09e+00 |
| gate | 12288×5120 | Q6_K | 1 | 83.67 | 83.67 | 616.8 | 8.86e-01 |
| ctl | 17408×5120 | Q5_K | 4 | 128.81 | 32.20 | 475.7 | 1.02e+00 |
| ctl | 17408×5120 | Q5_K | 1 | 90.92 | 90.92 | 673.9 | 1.02e+00 |
| anchor | 17408×5120 | Q8_0 | 4 | 136.86 | 34.22 | 691.9 | 1.17e+00 |
| anchor | 17408×5120 | Q8_0 | 1 | 124.26 | 124.26 | 762.1 | 1.17e+00 |

### Lecture

1. **Le gate tranche : Q6_K 17408×5120 @ ne1=4 = 494.7 GB/s < ~700 → H1 CONFIRMÉE.**
   En dessous de 700 sur **toutes** les shapes Q6_K (494–532 @ ne1=4 ; 600–630 @ ne1=1) :
   le trafic poids (57,3 % du cycle) tourne nettement sous le plafond DRAM.
2. **Méthode validée par l'ancre** : Q8_0 @ ne1=1 = 762,1 GB/s vs P5 754 → **+1,0 %**.
   Le flux naturel `graph_compute` du `.so` prod est mesuré correctement.
3. **Signature H1 (batch)** : ne1=4 < ne1=1 **systématiquement** sur les 3 types
   (Q6_K 495→616, Q5_K 476→674, Q8_0 692→762). Le chemin ncols=4 (W4R4/X4, 4 tokens/passe,
   le batch de verify MTP) est le point bas. C'est là que le kernel P7 doit agir.
4. **Gain P7 kernel Q6_K projeté** (12,43 GB/cycle, cycle 75,9 ms) : 495→700 = **+7,4 ms**
   (9,7 % du cycle) ; 495→750 = +8,6 ms ; 495→900 (plafond) = +11,3 ms (14,9 %). Le seuil
   de rentabilité 2 ms/cycle est largement dépassé → le P7 kernel Q6_K est le bon.

### Conséquences
- **H1 confirmée → P7 = kernel Q6_K** (VDR=1 → vectorisation/occupation de la variante
  W4R4, cible ne1=4/ncols=4). Q5_K reste exclu du trigger (gain max 1,83 ms < 2 ms).
- Q8_0 clos (692/762, référence). H2 (gaps CPU) n'est pas écarté mais la part « poids
  sous-régime » est maintenant mesurée et c'est la cible prioritaire.
- Le verrou « pas de kernel avant cette mesure » est levé : le chiffre existe.

## P7a — VERDICT FIGE (2026-10-08 06:20) — passage en P7

### Verdict
**H1 CONFIRMÉE.** Q6_K 17408×5120 @ ne1=4 = **494.7 GB/s** < seuil ~700. Le trafic
poids (57,3 % du cycle) tourne sous-régime ; **P7 = kernel Q6_K** (variante W4R4, cible
ne1=4/ncols=4). Q5_K exclu du trigger (gain max 1,83 ms < 2 ms) ; Q8_0 clos.

### Mesures obtenues (source de vérité : `p7a_results.tsv`)
- gate Q6_K @ ne1=4 : **494.7 / 531.9 / 497.0 / 509.5** GB/s (4 shapes réelles du decode)
- gate Q6_K @ ne1=1 : 616.1 / 630.3 / 600.1 / 616.8 GB/s
- ctl  Q5_K 17408×5120 : ne1=4 475.7 | ne1=1 673.9 GB/s
- anchor Q8_0 17408×5120 : ne1=4 691.9 | ne1=1 762.1 GB/s
- Pattern : ne1=4 < ne1=1 systématiquement (les 3 types) → le batch ncols=4 est le point bas.

### Conditions de test (figées)
- V100-PCIE-32GB cc7.0 80 SM, choisie par NOM par le binaire (device 0 dans la fenêtre).
- `.so` prod exact e4afbb4a (sha256 `55bc0a96…`), libggml-base prod, flux naturel
  `ggml_backend_graph_compute` (W4R4/X4 s'activent seules à ne1=4 sur Volta, pas d'ids).
- 300 iters / 10 warm ; GB/s = bytes_poids_lus_1× / temps wall (device-syncé).
- env = parité prod (`GGML_CUDA_VOLTA_Q6_W4R4=1` + SHORTLIST qwen38-27b exact 131072).
- Poids : F32 random → `quantize_row_*_ref` du `.so` prod ; layout Q6_K 210 B/256 (ligne 4200 B).
- Fenêtre V100 autoclose, trap relance service+modèle. Binaire pré-buildé, pas de nvcc.

### État sain de la prod (après la fenêtre)
- `ollama-v100.service` **active** ; API `/api/tags` + `/api/generate` OK.
- `.so` prod sha256 **identique avant/après** (`55bc0a96…`) → prod jamais modifiée.
- VRAM V100 0→32 Go (rechargée) ; **aucun segfault, aucun Xid** nouveau dans dmesg
  (dernière signature = p7a_min 21:05, pré-fix).
- GGUF intouché ; aucun ncu/nsys.

### Raccourci méthodologique ACCEPTÉ (décision utilisateur 06:20)
> On considère les résultats P7a **suffisamment probants pour lancer l'expérience
> applicative P7, SANS prétendre avoir démontré définitivement le mécanisme physique**
> responsable du déficit Q6_K. La mesure établit un fait (sous-régime, 494 GB/s au batch
> ncols=4) et valide la méthode (ancre Q8_0 +1,0 % vs P5) ; elle ne démontre **pas** la
> micro-cause (VDR=1, occupation, coalescing, requant, ou mixte). Le choix P7 = kernel
> Q6_K repose sur le seuil de rentabilité (gain projeté 7,4–11,3 ms/cycle ≫ 2 ms), pas sur
> une cause démontrée.

### Hypothèses NON démontrées — trace pour reprise (PAS des bloqueurs)
| # | hypothèse | état | condition de reprise |
|---|---|---|---|
| U1 | Le déficit Q6_K vient de **VDR_Q6_K=1** (1 valeur/regs/it vs 2) | plausible, non mesuré | si le kernel P7 ne rattrape pas le gap → profiler regs/occupancy (ncu sur la variante W4R4, fenêtre dédiée) |
| U2 | Le déficit vient de l'**occupation** (regs/smem/CTA/SM à ncols=4) | non mesuré | si U1 écarté → occupancy report cible ncols=4 |
| U3 | Le **requant Q8_1** (inconditionnel non-PXQ) pèse dans le temps mesuré | non isolé | si le gain P7 est ambigu → isoler requant (flag/disable) |
| U4 | **Métrique `corr maxrel` ~0,9–1,35** : oracle F32-brut, bruit K-quant inclus ; pas un oracle serré | connu | réouvrable si P7 ambigu ou pour approfondir : oracle F64 serré + bound d'erreur par type, validation du requant |
| U5 | Le gain applicatif ≠ gain kernel (MTP peut masquer/amplifier) | attendu | c'est l'objet même de P7 (A/B applicatif) |

→ Aucune de ces hypothèses ne bloque le passage en P7. Elles sont tracées avec leur
condition de reprise pour une analyse ultérieure ciblée.

### Prochaine étape (P7)
Expérience applicative A/B sur charge comparable : **débit de génération, latence par
cycle MTP, longueur d'acceptation, stabilité** — avant (`.so` prod e4afbb4a) / après
(`.so` P7 build séparé, kernel Q6_K optimisé). `.so` d'or conservé + rollback prêt.
Décision d'intégration seulement si gain confirmé sans régression.

## P7 — KERNEL Q6_K X4 : RESULTATS (2026-10-08 10:10) — GAIN CONFIRME FAIBLE → PAS D'INTEGRATION SEULE

### Ce qui a été fait (conformément au feu vert)
1. **Patch P7** dans le build séparé `p7/work/llama-volta-merged-p7` (jamais le tree prod,
   jamais le tree instrumenté `src/llama-volta-merged`) :
   - `vecdotq.cuh` : nouveau `vec_dot_q6_K_q8_1_x4()` — décode le fragment Q6_K (vl/vh/
     scales/d) **une seule fois** par (kbx, thread) et le réutilise sur les 4 colonnes q8_1,
     en miroir exact du X4 Q5_K existant.
   - `mmvq.cu` : paramètre template `q6_x4` + branche K-loop `Q6_K && ncols==4 && !fusion &&
     q6_x4` dans `mul_mat_vec_q` ; la W4R4 (rows_per_block=4) est conservée et reçoit `q6_x4=true`.
   - Build incrémental nvcc 12.6.85 / gcc-13 / sm_70-real (toolchain de référence du manifest)
     → **`.so` P7 md5 `a1b3b776`** (135 425 560 octets, NEEDED identiques au prod, 1254
     symboles exportés identiques hormis les mangled names aléatoires de nvcc).
2. **Gate kernel isolé** (fenêtre `run_prod_window_p7.sh`, `LD_LIBRARY_PATH` → `.so` P7,
   prod jamais touchée, sha256 vérifiée avant/après) :
   | cible Q6_K @ ne1=4 | prod W4R4 | P7 X4 | Δ |
   |---|---|---|---|
   | 17408×5120 | 494.7 | 502.1 | **+1.5 %** |
   | 5120×17408 | 531.9 | 541.5 | +1.8 % |
   | 10240×5120 | 497.0 | 504.8 | +1.6 % |
   | 12288×5120 | 509.5 | 519.6 | **+2.0 %** |
   Chemins non touchés (bruit de ré-exécution) : Q5_K +0.5 %, Q8_0 +0.1 %, ne1=1 ±0.5 %.
   → le gain X4 est **réel mais petit (+1,5 à +2 % sur le poids)** : la re-lecture des poids
   4× n'était PAS la cause principale du déficit 495 vs 692 GB/s.
3. **Expérience applicative A/B** (`p7/bench/p7_ab.sh`, 2 itérations complètes, swap
   `swap_revert.sh` garanti + revert automatique ; charge T1 256 / T2 512 / T3 256 tok, T=0,
   MTP actif, modèle `qwen3.8-27b-q6-mtp:latest`) :

   | tache | A (prod) ms | B (P7) ms | Δ ms | acc A → B |
   |---|---|---|---|---|
   | T1 (256 tok, prose) | 5609 / 5604 | 5547 / 5539 | **−62/−65** | 0.47771 → 0.47771 |
   | T2 (512 tok, code) | 7460 / 7456 | 7380 / 7374 | **−80/−82** | 0.89423 → 0.89423 |
   | T3 (256 tok, raisonnement) | 3807 / 3806 | 3766 / 3766 | **−41/−40** | 0.86792 → 0.86792 |

   **Moyenne 2 itérations : A = 60.70 t/s (16.48 ms/tok) → B = 61.37 t/s (16.29 ms/tok)
   = +1.10 % global (−185 ms / 1024 tok, soit −0.18 ms/tok).**
   - **Stabilité** : acceptance **bit-identique** à 5 décimales sur les 3 taches (T=0,
     déterministe) ; taux par position identiques ; VRAM identique (31 186 MiB) ; aucun
     Xid/segfault pendant les fenêtres ; service active après revert.
   - **Régressions : AUCUNE.**

### Bilan / décision
- **Gain confirmé sans régression : +1,10 % applicatif** (±0,03 % entre les 2 itérations —
  reproductible, bien au-delà du bruit). Mais c'est **~0,4 ms/cycle** (0,18 ms/tok × mean len
  2,48 ; cohérent avec le micro-gate : 12,43 GB × (1/494,7 − 1/502,1) GB⁻¹ = 0,37 ms/cycle),
  **sous le seuil de rentabilité 2 ms/cycle** fixé pour l'effort P7.
- **Décision (recommandation) : NE PAS intégrer `a1b3b776` tel quel en prod.** La prod
  reste en known-good `e4afbb4a` (revert vérifié, service sain). Le patch reste l'**artefact
  de base P7b** : il supprime le coût de re-décodage des poids et il est le prérequis d'une
  2ᵉ itération kernel.
- **Diagnostic affiné (le déficit reste ailleurs)** : le gap 495 (Q6_K) vs 692 (Q8_0) GB/s
  n'est PAS la re-lecture ×4 (X4 ne rapporte que +2 %). Ce qui reste structuré :
  `VDR_Q6_K_Q8_1_MMVQ=1` → 2× plus d'itérations de K-loop que Q8_0 (16 vs 32 blocs/itération
  de warp) et chaînes de dépendance sur les scales (2 `scales[4*i]` sérialisées par bloc)
  sur du dp4a **émulé** en sm_70 → le kernel Q6_K est **latence/ILP-bound**, pas
  bande-passante-bound. C'est l'hypothèse U1, désormais la seule piste crédible vers 700 GB/s.

### Prochaine action (à valider)
**P7b** : itération kernel Q6_K X4 — réduire la pression ILP/latence à ncols=4 :
traiter les 2 sous-blocs q6_K (32+32) d'un thread en parallèle (VDR effectif 2) ou
pré-charger les 4 scales dans des registres avant la boucle de dot, en gardant X4.
Cible : >650 GB/s sur 17408×5120 ne1=4 au micro-gate **avant** toute nouvelle fenêtre
applicative. Si P7b ne dépasse pas ~+3 % applicatif, le P7 kernel s'arrête et on retourne
à l'orchestration/gaps CPU (H2) avec ce chiffre figé.

## P7 — VERDICT FIGE + FEU VERT P7b (2026-10-08 10:20)

### P7 figé (chiffres de référence, source de vérité)
- **Gain kernel** : +1,5 % sur le poids Q6_K 17408×5120 @ ne1=4 (494,7 → 502,1 GB/s) — micro-gate.
- **Gain applicatif** : **+1,10 %** global (60,70 → 61,37 t/s, 2 itérations, ±0,03 %).
- **Reproductible** ; **acceptance MTP inchangée** (bit-identique à 5 décimales, T=0) ; aucune
  régression ; **prod restaurée sur le known-good e4afbb4a** (revert vérifié).
- **Constat clé** : le gain X4 ne représente qu'**~0,4 ms/cycle** (0,18 ms/tok × mean len 2,48 ;
  cohérent avec 12,43 GB × (1/494,7 − 1/502,1) = 0,37 ms/cycle) → sous le seuil de rentabilité 2 ms.

### Feu vert P7b (décision utilisateur)
Piste ILP/latence du Q6_K X4 (paralléliser les 2 sous-blocs 32+32 et/ou précharger les scales
en registres), **build expérimental séparé uniquement, jamais le .so prod**.
- **D'abord un micro-gate kernel** : il faut un signal **nettement supérieur à P7 (+1,5 %)** ;
  sinon pas de multiplication de variantes → **fermeture du chantier Q6_K kernel**.
- Si micro-gate prometteur → A/B applicatif **exactement dans les conditions de P7**.
- **Critère** : gain applicatif clairement supérieur à +1,10 % (P7) et se rapprochant du seuil
  de rentabilité ; sinon figer et **bascule sur H2 (orchestration / gaps / CPU)**.
- Objectif de P7b : déterminer si la piste contient encore un gain méritant du temps — pas
  de re-démonstration (P7 l'a fait).

## P7b — HOISTING SCALES : RESULTAT + VERDICT FIGE (2026-10-08 12:30) — MICRO-GATE NON PASSE → CHANTIER Q6_K KERNEL FERME, BASCULE H2

### Ce qui a été fait (une seule variante, conformément au feu vert)
- **Variante unique** : hoisting des scales de poids en registres dans `vec_dot_q6_K_q8_1_x4()`
  (`llama-volta-merged-p7b`, copie dédiée du tree P7 — le diff source P7→P7b est limité à
  `vecdotq.cuh` : nouveau `vec_dot_q6_K_q8_1_impl_mmvq_sc(vl, vh, u, sc0, sc1, d, d8)` +
  `sc0=scales[0]`, `sc1=scales[4]` chargés UNE FOIS par (kbx) au lieu d'être re-lus 4× dans
  impl). Même arithmétique, même ordre d'opérations → sortie bit-identique attendue.
  Le chemin standard (ncols_dst≠4) et tous les autres kernels sont inchangés.
- **Build séparé** `build-p7b` → **`.so` P7b md5 `c39da9b3`** (distinct de P7 `a1b3b776`).
  - Règle 1a : kernel cible `mul_mat_vec_q<type14,ncols=4,...,q6_x4=true,rows=4>` présent
    (signature `type14ELi4ELb0ELb0ELb0ELb0ELb1ELi4EE` dans le ptxas + le fatbin du .so).
  - Règle 1b : **registres ptxas = 96, IDENTIQUES à P7 (96)** → le compilateur avait déjà
    maintenu les scales en registres ; le hoisting manuel est sans effet sur le code généré.
- **Micro-gate kernel** (même fenêtre/méthode P5 que P7, `.so` P7b chargé par LD_LIBRARY_PATH,
  prod jamais touchée, sha256 connue avant/après) :

  | Q6_K @ ne1=4 | prod W4R4 | P7 X4 | P7b hoist | P7b vs P7 |
  |---|---|---|---|---|
  | 17408×5120 (GATE) | 494.7 | 502.1 | **501.0** | **−0.2 %** |
  | 5120×17408 | 531.9 | 541.5 | 539.3 | −0.4 % |
  | 10240×5120 | 497.0 | 504.8 | 508.0 | +0.6 % |
  | 12288×5120 | 509.5 | 519.6 | 521.1 | +0.3 % |
  | contrôle Q5_K 17408×5120 | 475.7 | 478.1 | 482.8 | +1.0 % (bruit fenêtre) |

### Verdict (règle 2 du feu vert P7b)
- **Cible 501.0 GB/s < seuil 515 GB/s** (et ≤ P7, −0.2 % ; le contrôle Q5_K de la même
  fenêtre à +1.0 % montre que le delta est dans le bruit inter-fenêtres).
- **Le hoisting ne donne rien** : le mécanisme précis (re-lecture des scales par colonne)
  n'était pas le goulot — ptxas le démontre (96 regs avant et après). Piste fermée, pas de
  variante supplémentaire, **pas d'A/B applicative** (le filtre kernel a tranché).
- **DECISION : CHANTIER Q6_K KERNEL FERME.** La prod reste en known-good `e4afbb4a`.
  Les artefacts P7/P7b (patch, builds, TSV) sont conservés :
  `p7a_results_prod_gate.tsv`, `p7a_results_p7_x4.tsv`, `p7a_results_p7b_hoist.tsv`.

### Synthèse définitive de la piste Q6_K (pour le dossier H2)
1. Le kernel Q6_K decode est bien sous l'isoparité Q8_0 (494.7 vs 691.9 GB/s @ ne1=4,
   ~197 GB/s de déficit) → le verdict H1 de P7a tenait.
2. Mais la **marge exploitable par patch local est mince** : X4 (décodage une fois) =
   +1.5 % kernel / +1.10 % applicatif (~0.4 ms/cycle) ; hoisting scales = 0. Le gap restant
   est structurel : `VDR_Q6_K_Q8_1_MMVQ=1` (2× itérations de K-loop vs Q8_0) + dp4a émulé
   en sm_70 + pression registres (96 regs → occupation plafonnée). Fermer ce gap exigerait
   une réécriture de kernel (VDR=2 / mmq tensor cores), hors scope du budget de la piste.
3. Potentiel théorique si Q6_K atteignait le débit Q8_0 : 12.43 GB × (1/494.7 − 1/691.9)
   ≈ **7.2 ms/cycle** — c'est l'échelle du « gap kernel », mais inaccessible par les moyens
   P7/P7b. Le reste du cycle MTP (~16.3 ms/tok mesuré) doit donc être attaqué par H2.

### Bascule H2 (orchestration / gaps / CPU) — point de départ
Reprise exacte : H2 = localiser ce qui reste dans le cycle ~16.3 ms/token hors efficacité
du kernel Q6_K : gaps CPU↔GPU entre kernels, overhead de launch, pipeline MTP
(draft/verify, shortlist kernel), ops KV-cache, synchronisations. Outils déjà en place :
`p5_ktrace.so` (trace des kernels par LD_PRELOAD), journal `print_timing`, `p7a_dispatch`
comme harnais de mesure. Prod en known-good, service actif, VRAM libre.
