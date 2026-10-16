# H2 — Protocole de forage nsys (PRÉPARATION — à exécuter après validation)

> Statut : **PRÊT, NE PAS LANCER.** Capture validée sur les éléments de la machine.
> Mandat : déterminer où se trouvent les ~30–43 ms/cycle MTP non attribuées (~50 % du cycle ~76 ms).
> Discipline : **prod jamais modifiée** (pas de swap .so, pas de patch, pas de redémarrage hors fenêtre) — même invariants que P0→P7a.

---

## 1. Invariants (ne changent pas)

| # | Invariant | Vérification |
|---|---|---|
| 1 | `.so` prod byte-identique avant/après (md5 `e4afbb4a`) | md5sum au début ET à la fin, dans le log |
| 2 | Unité systemd sauvegardée avant, service restauré après (trap `die` → restore) | `cp` du unité dans le dir de run ; `systemctl is-active` + API check après |
| 3 | Gate V100 avant le stop : `run/v100probe` refuse si le V100 n'est pas énuméré (cas du reboot 10-04 16:29) | rc=0 obligatoire |
| 4 | Aucun `p5_ktrace.so`, aucun DISABLE_GRAPHS, aucun code modifié | le serveur direct tourne sur le `.so` prod tel quel (LD_LIBRARY_PATH du payload prod) |
| 5 | Port dédié 44901 (ne touche pas 11434) | — |
| 6 | Restauration **obligatoire** même en erreur (trap + `die()`) | `restore()` dans tous les chemins |

**Durée totale d'arrêt du service : ~15–25 min** (startup + prefill ~100k hors capture ~4–6 min + capture ~60–90 s + export/stats + restauration).

## 2. Ce qui est déjà vérifié sur la machine (8 oct)

- `nsys` 2025.5.2 présent (`/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys`) ; `sqlite3` + python3-sqlite OK.
- GGUF prod présent : `…/blobs/sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb` (23 088 409 504 octets = modèle `qwen3.8-27b-q6-mtp` ; **le même blob** est partagé avec `:vision` — vérifié par les manifests).
- **Bug 1er essai (20:09)** : le hash GGUF du script contenait une typo (62 caractères, `…80bb578…` au lieu de `…80bbcb578…`) → `GGUF introuvable`, le script est sorti proprement **sans rien toucher** (service intact, md5 vérifié au restore). Corrigé + contrôle de taille ajouté au preflight. **Preflight hors-service 6/6 OK** (hash, md5 `.so`, gate V100, filler, shortlist, binaire) à 20:27.
- Corpus long disponible : `v100-repro/work/filler-200k.txt` (1,1 Mo ≈ 200k tok) — on prend les premiers ~550 k chars ≈ **100k tokens** de contexte.
- `run/v100probe` présent (gate post-reboot, compilé).
- Unité prod lue : env de parité = `GGML_CUDA_VOLTA_Q6_W4R4=1`, `GGML_CUDA_QWEN35_MTP_SHORTLIST=…/qwen38-27b-exact-131072.i32`, `LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1`, KV q8_0, ctx 200000, host 11434.
- Les prompts du bench shortlist dans /tmp sont **perdus** (reboot) → le prompt est reconstruit depuis le filler (aucune dépendance au /tmp).
- Ancien script `night-20261004/nsys-t4-profile.sh` : base connue — **mais son pattern `nsys launch` (sans `-o`) + `nsys stop --output=` est exactement ce qui a raté le 8 oct** (nsys 2025.5.2 : `stop` n'accepte que `--session`/`--keep`).

### Echec du 3e essai (21:23) et ROOT CAUSE definitif : l'ordre CUDA != l'ordre nvidia-smi
- La gate fit a fait son boulot : le serveur n'a PAS charge le modele (breakdown Host seulement, aucun device CUDA). Mais le vrai bug, trouve ensuite : **`CUDA_VISIBLE_DEVICES=1` pointait vers la GTX 1070, pas la V100.**
- Ordre reel (verifie avec un probe `cuDeviceGet*` dedie, 9 oct, sans arreter le service) : **cuda:0 = Tesla V100, cuda:1 = GTX 1070** — l'inverse de `nvidia-smi -L`. Sans nsys, sous nsys, et avec l'`LD_LIBRARY_PATH` du toolkit : stable. C'est pour ca que `v100probe` affichait toujours `free=0/total=0` (le .so prod ne rapporte la memoire que sur les devices initiales ; et que tous les essais « sans GPU » n'etaient en fait que… la 1070).
- **La prod elle-meme utilise `CUDA_VISIBLE_DEVICES=0`** (lu dans `/proc/<llama-server prod>/environ` : `CUDA_VISIBLE_DEVICES=0` + `GGML_BACKEND_PATH=…/libggml-cuda.so` + `LLAMA_ARG_FIT_TARGET=1024` — poses par l'ollama scheduler, pas par l'unite systemd). Le journal prod confirme : `using device CUDA0 (Tesla V100-PCIE-32GB) (0000:2b:00.0)`.
- Le run de 20:31 (fit abort) s'explique donc aussi : le serveur ne voyait que la 1070 (8 Go) → fit impossible ; pas une question de « 1 GPU vs 2 GPU ».
- **Correctifs du script (parite prod stricte)** : `CUDA_VISIBLE_DEVICES=0` ; `GGML_BACKEND_PATH` ajoute ; ordre `LD_LIBRARY_PATH` aligne sur le prod (`lib/ollama` avant `cuda_v12`) ; **attente liberation V100 (< 1 Go, 120 s) apres le stop** — le fit est serre a 58 MiB, tout residu VRAM est fatal ; gate `v100probe` **post-stop** (`free=32xxx` = V100 visible ET libre) ; gate `using device CUDA…Tesla V100` apres le fit (protection anti-inversion d'ordre).
- **Note OOM prod (9 oct 07:15–07:52)** : la prod a fait 4 OOM CUDA de suite (fit « need to reduce by 490 MiB » — la V100 etait a 30.8 Go libres au lieu de 31.5, un residu d'un de mes serveurs de test de la veille ; le fit a survécu en reduisant le ctx a ~105k). Ollama a auto-recover (reloads a chaque OOM), le service tourne depuis le restart de 07:54 et le md5 est sain (`e4afbb4a`). Conséquence : **le preflight d'arret doit garantir une V100 vierge** — c'est ce que fait la nouvelle attente de liberation.
- **Leçon** : `v100probe` est une gate FAIBLE (il ne detecte que l'absence totale d'init CUDA). La verite vient de `using device CUDA…` dans le log serveur + d'un probe qui compare les **noms** de devices, pas un simple rc.

### Ce qui reste a voir (a verifier a la prochaine capture, sans prejuger)
- Le serveur direct (hors ollama) avec le petit modele nomic abort au warmup sur `n_outputs_max` avec `-c 2048` (incoherence interne) : le run H2 utilise le vrai modele (ctx 200000), pas ce cas. Si ca refait, on bascule la capture sur l'API `/api/generate` du server direct — mais seulement apres avoir confirme que le server direct du fork expose la meme API ollama-compatible qu'utilise le prod (le fork implemente `context checkpoint` et le journal prod montre que le scheduler est du prod, pas du fork).
### Résumé des 3 échecs (état corrigé au 9 oct)
1. **20:09 — GGUF introuvable** : typo de hash (62 car.). Corrigé + contrôle de taille. Rien touché.
2. **20:31 — fit abort + `nsys stop --output=` + shortlist** : les correctifs restent valides (contrôles de contenu curl, stop sans `--output` + rattrapage du rapport, gate shortlist post-warmup, retrait `--trace=syscall`). **Mais la cause du fit abort est révisée** : ce n'était pas « 1 GPU vs 2 GPU », c'était `CUDA_VISIBLE_DEVICES=1` = la GTX 1070 (voir root cause ci-dessus). Le run de 21:23 (gate fit OK, serveur parti sur rien) confirme le même mécanisme : V100 pas vue → fit « OK » sur Host seul.
3. **21:23 — ordre CUDA** : root cause definitive + correctifs parite prod (ci-dessus).

## 3. Méthode de capture (premier tir)

### Principe
`llama-server` **direct** (payload prod, `.so` prod, mêmes flags/env) lancé **sous `nsys launch`**, préfill du contexte ~100k **hors capture**, puis fenêtre de capture **60–90 s de decode steady-state** à ~100k+ n_past, MTP actif.

```
nsys launch --session=h2 --trace=cuda,osrt,syscall --cuda-graph-trace=graph \
  <llama-server prod> --model <GGUF> -c 200000 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn auto -b 512 -ub 512
```

- **`--cuda-graph-trace=graph`** : chaque `cudaGraphLaunch` est tracé (graphId) → **marqueur de phase MTP natif** (1 decode step = 1 graph launch : verify = gros graph, drafts = petits graphs). C'est la clé pour H2c sans toucher au code.
- **`osrt`** : activité OS du CPU (futex, usleep, clock_nanosleep…) → permet de classifier les gaps GPU (CPU bloqué vs CPU en cours de calcul).
- **`syscall`** : récap syscalls (complément CPU).
- Pas de NVTX dans le code → pas de trace nvtx (inutile au premier tir, et pas de modif de code au premier tir).

### Séquence
1. Gate `v100probe` + md5 `.so` + backup unité.
2. `systemctl stop ollama-v100` (service arrêté ≈ 15–25 min au total).
3. `nsys launch` du serveur direct (port 44901, `CUDA_VISIBLE_DEVICES=1` = V100).
4. Warmup court (16 tok, « Say OK ») — chauffe CUDA graphs + vérification shortlist chargé.
5. **Prefill du contexte ~100k HORS capture** (+32 tok) — ~4–6 min.
6. `nsys start --session=h2`.
7. **Decode : ~1600 tokens sur le même prompt** (le checkpoint LCP restaure les ~100k → le decode part de n_past≈100k+ ; fenêtre ≈ 60–90 s à ~25–30 t/s long contexte).
8. `nsys stop` → export `sqlite` + stats CSV (`cuda_api_sum`, `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, `osrt_sum`).
9. Kill serveur → **restauration du service** + vérifications post (md5, API, journal : modèle rechargé, shortlist actif).

### Représentativité
- Contenu = filler répéteur → **acceptance MTP élevée** (≈0,8–1,0) → mean len ~3,3 (vs 2,62 charge réelle) : le temps par cycle wall est un peu plus long qu'en prod (à renormaliser : `cycle_ms = 1000 × mean_len / tps`). La **structure** du cycle (phases, gaps, familles de kernels) est celle du régime long contexte, qui est l'objet de H2.
- Le régime de batch est exactement celui de la prod (MTP 3+1, `-np 1` → ne1∈{1,4}) : même graphs, même kernels.
- Le chiffre « 76 ms » sert d'ordre de grandeur, pas de référence absolue (acceptance différente) ; l'analyse rapporte le cycle mesuré et sa renormalisation.

## 4. Mesures produites (script `h2_analyze.py`)

Source : export sqlite de l'nsys-rep (+ CSV nsys en cross-check).

### GPU (obligatoires)
1. Durée totale de la fenêtre observable (premier→dernier intervalle GPU).
2. Somme des durées kernels + memcpy + memset (temps GPU occupé, **fusionné** multi-streams).
3. Temps GPU **par famille de kernels** (regex sur les noms : `mul_mat_vec_q`/`vec_dot` = poids mmvq ; `mmq`/`mul_mat` ; `flash_attn`/`fattn` ; `gated_delta_net` ; `quantize`/requant ; `sample`/`top_k`/shortlist ; memcpy H2D/DtoH ; elementwise/norm/rope ; autres).
4. Nombre de kernels (total + par famille).
5. **Gaps entre activités GPU** (complémentaire du temps occupé).
6. Distribution des gaps : **médiane, p90, p95, max**.
7. Top-20 plus gros gaps individuels (avec horodatage).
8. **Occupation temporelle GPU** = occupé / fenêtre.

### CPU (minimum)
1. Activités OSRT par type (futex, usleep, sleep, …) : temps total, count.
2. **Classification de chaque gap GPU** par recouvrement OSRT :
   - `CPU-bloqué` (futex/attente dans le gap → thread qui attend, typiquement sync GPU ou lock) ;
   - `CPU-cours` (aucun OSRT dans le gap → le CPU exécute du code entre lancements : orchestration) ;
   - `idle` (le reste).
3. Temps d'API CUDA côté CPU : count + temps de `cudaGraphLaunch`, `cudaStreamSynchronize`/`cudaDeviceSynchronize`/`cudaEventSynchronize` (depuis `cuda_api_sum`).
4. **Corrélation CPU ↔ gaps GPU** : fraction du temps de gap attribuable à (2)+(3).

### MTP (sans modif de code)
- Via `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` + `graphId` des kernels : chaque cudaGraphLaunch = 1 step ; regroupement en cycles (1 verify = le step le plus long + 3 drafts).
- Si le graphId n'est pas relié aux kernels dans cet export → **repli heuristique** : bursts d'activité (gaps intra-burst < 2 ms), cycle = 4 bursts (le max = verify).
- Par cycle : durée wall (verify→verify), temps GPU par phase (verify / drafts), gaps **intra-phase** vs **inter-phase** (la signature H2c).

### Décomposition finale (la figure du rapport)
```
Cycle MTP (mesuré, renormalisé aux ~76 ms prod)
├── kernels GPU            B ms   (par famille)
├── gaps GPU — CPU bloqué  Gw ms  (attentes sync)
├── gaps GPU — CPU en cours Gc ms (orchestration)
└── (reste)
```
+ réponse chiffrée : **quelle fraction des ms est GPU utile, quelle fraction gaps, quelle fraction CPU/orchestration.**

## 5. Critères de décision (chiffrés, figés avant la capture)

| Hypothèse | Signature numérique attendue | Verdict |
|---|---|---|
| **H2a — gaps CPU↔GPU** | occupation GPU < ~80 % du cycle **ET** (Gw+Gc) > ~20 % du temps de cycle **ET** corrélation CPU (OSRT futex/sync ou Gc) significative sur les top gaps | renforcée → next : réduire l'orchestration (CPU) : batch de lancements, fusion des ops MTP, scheduler — pas de kernel |
| **H2b — sous-régime GPU** | occupation GPU > ~90 % **ET** une famille « non comptée » ou sous-optimale importante (ex. mmvq K-quant globalement < 700 GB/s, fattn, SSM) | renforcée → next : re-qualifier cette famille (kernel ou config) |
| **H2c — overhead MTP structurel** | temps systématiquement regroupé **entre phases** (inter-phase >> intra-phase), coûts identifiés par cycle (verify→draft, draft→verify, refresh KV, transitions de graph) | renforcée → next : attaquer la mécanique MTP (graphes, refresh, transitions) |
| **Les trois faibles** | B élevé et bien attribué, gaps faibles, pas de structure de phase | → remonter la décomposition initiale (P6.5) : re-estimer le budget poids (21,68 GB, layouts prod) et rouvrir le gate P7a avec une ancre plus serrée |

## 6. Risques & mitigations

| Risque | Mitigation |
|---|---|
| Reboot en cours pendant le run (cas 10-04) | gate `v100probe` au démarrage ; si le driver n'est pas prêt, on sort sans rien toucher |
| OOM V100 (modèle 31,5 Go sur 32 Go) | service stoppé = VRAM libre ; garde `cudaMemGetInfo` implicite via le warmup qui échoue proprement → `die` → restore |
| nsys export trop lourd (100k+ kernels) | fenêtre bornée 60–90 s ; export sqlite local ; si l'export échoue, les stats CSV (kern_sum/api_sum/osrt_sum) sont déjà sorties et suffisent pour H2a/H2b (gaps fin → 2e tir) |
| EOS précoce avant 1600 tok (fin de tâche dans la fenêtre) | le rapport indique `gen_count` ; l'analyse borne la fenêtre au régime cadence-stable (écart-type du cycle < 15 %) et exclut la queue ; sinon rejouer avec un prompt plus « continue » |
| Le service ne remonte pas | `die()` → restore dans tous les chemins ; `Restart=always` sur l'unité ; vérification post (md5 + API + journal) consignée |
| Biais filler (acceptance élevée) | renormalisation du cycle par mean_len/tps ; la structure (phases/gaps/familles) est indépendante du contenu |
| L'interpose CUDA graph (le piège P6) | **non utilisé** : CUPTI natif via nsys voit le contenu du replay ; `p5_ktrace.so` volontairement écarté |

## 7. Runbook (lors du feu vert)

```bash
# 1) lancer la capture (stoppe le service ~15-25 min, se restaure seul)
bash /home/guile/llama-experiments/h2/h2_capture.sh
#    -> /home/guile/llama-experiments/h2/runs/run-<stamp>/  (nsys-rep, sqlite, stats, console.log)

# 2) analyser
python3 /home/guile/llama-experiments/h2/h2_analyze.py /home/guile/llama-experiments/h2/runs/run-<stamp>
#    -> rapport-h2.md dans le même dir + tables (csv)

# 3) vérifier l'état prod (déjà fait par le script, à confirmer)
systemctl is-active ollama-v100
md5sum /home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so
```

## 8. Livrables attendus

1. `h2/runs/run-<stamp>/` : `prof.nsys-rep`, `prof.sqlite`, `stats-*.csv`, `console.log`, `server.out`, unité backupée.
2. `rapport-h2.md` : décomposition du cycle + tables GPU/CPU/MTP + **verdict H2a/H2b/H2c** avec les chiffres.
3. Mise à jour de `SYNTHSE.md` (section H2) + de l'ETAT de reprise.

## 9. Ce que ce premier tir NE fait PAS (2e tir si besoin)

- Aucun NVTX/instrumentation de code (interdit au premier tir) → si la phase MTP n'est pas identifiable par graphId, 2e tir = ajout NVTX dans un **build séparé** (jamais le .so prod) rejoué en fenêtre.
- Pas de verrouillage de clocks GPU (nsys ne le fait pas ; à documenter dans le rapport comme limite de reproductibilité).
- Pas de 2e contexte (160k) au premier tir ; si le 100k est propre et que la question se pose de la position (cf. bench 02/10 : la position domine), un 2e run 160k est prévu en option (env `H2_CTX_CHARS`).
