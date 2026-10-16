# run7 CUPTI v2 — verdict instrumentale (13/10)

**Question** (formulation exacte du F1) : *« Les durées CUPTI des kernels du decode
task 236 (run6, segment pid 387290, seq 661583..705293) couvrent-elles le temps mural
W = 1576,52 ms, et si oui, quel niveau de simultanéité réelle existe entre les streams 16/17 ? »*
En pratique : déterminer **U (union des intervalles GPU)**, **O (chevauchement)**, **idle = W − U**,
sans assimiler Σ dur ≈ W à « GPU 100 % occupé ».

**Décision : run7 v2 NÉCESSAIRE (recapture, pas re-analyse)** — les données existantes
permettent de borner mais pas de trancher (ci-dessous). Protocole de validation explicite à la §6.

---

## 1. Ce que les données EXISTANTES permettent (bornes strictes)

Décomposition exacte du decode (bin/names alignés 1:1, 43 711 records, seq contigus) :

| grand | valeur |
|---|---|
| S16 = Σ KER stream 16 (40 122 ker) | 1406,56 ms |
| S17 = Σ KER stream 17 (2 496 ker) | 169,25 ms |
| CPY stream 0 (1 093) | 34,42 ms |
| S_all | 1610,23 ms |
| W (mural, wall clock §15) | 1576,52 ms |

Bornes mathématiques (aucune hypothèse au-delà du nombre de streams) :
- **U ∈ [1406,56 ; 1576,52] ms** → **U/W ∈ [89,22 % ; 100 %]**
- **idle = W − U ∈ [0 ; 169,96] ms** → **idle ≤ 10,78 %** de W
- **O = S_all − U ∈ [33,71 ; 203,71] ms** → O/W ∈ [2,1 % ; 12,9 %]

L'ancien « 99,95 % » = SK/W (Σ KER seuls / W) — **c'est un rapport de somme, pas une mesure
d'occupation**. Il est *compatible* avec U/W = 89–100 % mais ne le démontre pas.

**Ce que la borne inférieure dit déjà** : le GPU ne peut PAS être « 100 % idle » ni « largement
sous-utilisé » ; au pire il y a 10,8 % d'idle. Mais la question « 95 % ou 100 % ? » **reste
indéterminée** → seule une mesure de positions (gpu_start/gpu_end) tranche.

## 2. Ce que les données EXISTANTES ne peuvent PAS dire (F1 confirmé)

- `t` du `.names` = **temps d'émission** (horloge du thread drain, ~CLOCK_REALTIME ns) :
  - prefill 387221 : tspan d'émission 197 s (vs Σ dur ≈ 194 s → cohérent avec un drain ~temps réel,
    PAS avec une horloge GPU) ;
  - decode : tspan d'émission 73,7 ms (vs Σ dur 1610 ms) → le drain n'est pas à l'échelle du GPU.
  - **Conclusion : `t` ne donne ni start ni end GPU. Aucune union/chevauchement n'est calculable
    sur run6.** C'est le verrou exact du F1.
- Horloges des headers : une par fork (CLOCK_MONOTONIC, origin décalé par fork) → non
  corrélables entre segments (F16).

## 3. Domaine temporel de chaque champ (vérifié)

| champ | domaine | corrélable à ? |
|---|---|---|
| `dur_ns` (bin+names) | durée GPU CUPTI, ns | **oui** (unités cohérentes, max 105,9 ms plausibles) |
| `stream` | identifiant stream CUDA | oui |
| `seq` (bin) | compteur global 0..705 293, **contigu** (tiling exact, 0 gap) | oui — **ordre total des events**, exploitable |
| `t` (names) | ns, horloge CPU thread drain à l'émission | **non** (ni GPU, ni wall cohérent) |
| header | horloge CPU du fork à l'init | non entre segments (F16) |

## 4. Cohérence compteurs/unités vs format émis (vérifié)

- Format bin : header texte `[cupath] pid=N ...\n` + records 24 B `<QQII>`
  (seq u64, dur u64, stream u32, op u32 ; op 0=KER 1=CPY 2=RESET) — 89 Mo = 62 headers + 705 356 records.
- `.names` : 705 294 lignes `KER|CPY t= stream= dur_ns= name=` (RESET exclus) — **alignement 1:1**.
- **decode : 0 mismatch** dur_ns (bin vs names) et stream (bin vs names) sur 43 711 records.
- seq global contigu 0..705 293 ; 7 segments « atypiques » = 1 RESET à seq 0 (fin de vie) avant
  le header suivant, pattern régulier — pas de perte de record.
- 8 segments portent des records (403/462/2 428/612 133/21 862/3 696/20 599/**43 711**) ;
  les 54 headers vides = forks sans GPU (spawn, warmup, …).
- **Correction F2 : ce ne sont PAS 62 segments de données mais 62 headers dont 8 porteurs.**
  Le decode = le 8e porteur (pid 387290, seq 661583..705293) = **déterminé par seq, plus « par chance »
  (F5 levé pour cette fenêtre : dernier bloc = dernière tâche decode)**.
- Unités ns confirmées (dur max 105,9 ms ; un decode mmvq mean 112,8 µs — plausible V100).

## 5. Réponses aux questions du brief (synthèse)

| question | réponse |
|---|---|
| Quelle affirmation vérifier ? | U/W réel (busy) et O (chevauchement 16/17) du decode task 236 |
| Timestamps fiables ? | `dur_ns` OUI ; `t` NON (émission CPU) ; headers NON entre segments |
| Méthode durée cumulée vs activity vs overlap vs wall | `S_stream = Σ dur` (par stream) → **borne** U ≥ S − min(S16,S17) − CPY ; **mesure** U = |∪[gpu_start,gpu_end]| et O = S − U — nécessite start/end GPU |
| Compteurs/unités cohérents ? | OUI (0 mismatch decode, seq contigu, ns plausibles) |
| run7 nécessaire ? | **OUI** — la borne [89,22 ; 100] % ne tranche pas la question posée |

## 6. Protocole run7 v2 (à exécuter seulement si la borne 89,2–100 % ne suffit pas)

**Changements shim v2** (un seul, minimal) :
1. Pour chaque record : **`gpu_start_ns` / `gpu_end_ns`** CUPTI (activity buffer `start`/`end`,
   même domaine horloger pour tous les records d'un même processus — CUPTI garantit une horloge
   GPU cohérente par run ; **ne pas** les comparer entre runs/segments).
2. Header : `t_real_us` = **CLOCK_REALTIME** à l'init (remplace l'origine CLOCK_MONOTONIC du F16)
   → permet d'ancrer le run7 au wall (début/fin de la fenêtre decode) sans corréler les segments.
3. **Silencing stderr** du shim (F6) et nom de fichier de sortie correct (F7) — hygiène, pas science.
4. Critère de validation explicite (avant d'analyser) :
   - **V1** : 0 record sans start/end ; **V2** : `dur_ns == end − start` ∀ record ;
   - **V3** : seq contigu et alignement bin/names 1:1 (reprise du check §4) ;
   - **V4** : au moins une fenêtre decode (64 tok, `finish_reason=length`) identifiable par
     profil de kernels (mmvq + q8v14partial, sans coïncidence) — cf. F15 ;
   - **V5** : `|W_cupti|` (end−start du run, horloge GPU) vs W wall : écart < 2 % (sanity).
**Résultat attendu par hypothèse** :
- H_a « GPU saturé » : U/W ≥ 97 % (idle < 3 %).
- H_b « léger idle structurel » (sync par round) : U/W ∈ [89 ; 97] %, idle ≤ 169 ms.
- H_c « forte sous-occupation » (réfute l'ancien 100 %) : U/W < 89 % → **impossible** vu la borne,
  à moins d'un bug de capture (V1–V5 le détecteraient).
**Lecture post-capture** : U = union des intervalles (sweep-line), O = S_all − U,
idle par round (découpes aux syncs), et **réconciliation** : si U/W ∈ [89,2 ; 100] % → cohérent
avec run6 ; sinon → le run6 est fautif, on refait toute l'attribution.

## 7. Verdict

- **F1 : confirmé** — `t` est un temps d'émission CPU, non exploitable pour U/O.
  Les données run6 **bornent** U/W à **[89,22 % ; 100 %]** (idle ≤ 10,78 %) mais ne le tranchent pas.
- **F2 : corrigé** — 62 headers, 8 porteurs de records ; le decode = 8e porteur (déterminé par seq,
  pas par chance → F5 levé pour cette fenêtre).
- **Limite de la mesure** : sans gpu_start/gpu_end, U, O et l'idle par round **restent indéterminés**
  à l'intérieur des bornes ci-dessus. Les répartitions par famille/dimensions/fréquence (§8 du brief)
  **sont en revanche valides et utilisables maintenant** (dur_ns fiables).
- **Décision** : lancer le **run7 v2** uniquement pour trancher U/W et O ; tout le reste
  (décomposition `mul_mat_vec_q`, poste KV Q8_0) se fait sur les données existantes.

---
## Annexes — chiffres dérivés (déjà exploitables)

**`mul_mat_vec_q` par (type, ncols) — decode task 236, 9 408 lancements, 840,27 ms (53,3 % KER)**
| type | ncols | n | ms | mean |
|---|---|---|---|---|
| q6_K (t14) | 4 | 4 050 | 456,83 | 112,8 µs |
| q8_0 (t8) | 4 | 3 617 | 168,25 | 46,5 µs |
| q5_K (t13) | 4 | 1 278 | 132,78 | 103,9 µs |
| *mapped* (mangling différent, non décomposable) | – | 51 | 46,40 | 910 µs |
| q6_K (t14) | 1 | 255 | 28,97 | 113,6 µs |
| q8_0 (t8) | 1 | 103 | 2,86 | 27,8 µs |
| t12? | 4 | 36 | 2,52 | 70,0 µs |
| t20? | 4 | 18 | 1,66 | 92,0 µs |

**Lecture** : le poids dominant est q6_K (60,5 % du mmvq) malgré un mean 112,8 µs ≈ 2,4× plus
lente que q8_0 (46,5 µs) à ncols=4 — la couche q6_K/nc4 est à elle seule 28,9 % du decode.
ncols=4 = batch 4 (T=4), cols=1 = T=1 (probablement attention/draft). Les 51 `*mapped*` (46,4 ms,
mean 910 µs) restent à identifier (nom manglé différent — à traiter comme piste, pas comme un
chiffre final).

**KV Q8_0 (KER, sans recouvrement — 33,3 % KER)** : `ggml_q8v14partial` 288× 469,59 ms (29,8 %)
+ `quantize_q8_1` 9 408× 28,24 ms (1,8 %) + `q8v13reduce` 288× 26,77 ms (1,7 %).
Signature binaires confirmées (`libggml-cuda.so` Ollama) :
`ggml_q8v::partial_kernel(const float*, const block_q8_0*, const block_q8_0*, const __half*, int,int,int, float, long×8, __nv_bfloat16*, float*, float*)`
et `ggml_q8v::reduce_kernel(const __nv_bfloat16*, const float*, const float*, int,int, float*, long,long)`.
→ `q8v14partial` lit q8 KV (block_q8_0) + proj half, partiel ; `q8v13reduce` réduit les partiels bf16.
**Le lien `quantize_q8_1` ↔ `q8v14partial` est séquentiel (quantifier puis lire), non recouvrant** :
ils s'additionnent légitimement (28,24 + 469,59 + 26,77). Le « 32 % cumulé KV » de la revue tient
en 33,3 % KER une fois le double-comptage exclu.
