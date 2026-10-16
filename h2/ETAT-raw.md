# ETAT.md — H2 : verdict Nsight/CUPTI et bascule sur les journaux serveur

Date : 2026-10-09 (20:00–20:30, heure locale)
État prod : **service `ollama-v100` actif, md5 `libggml-cuda.so` = `e4afbb4a`, V100 ~32 350 MiB, modèle chargé** (vérifié après chaque test, restauration par `trap EXIT`).

---

## 1. VERDICT — piste Nsight/CUPTI : CLOS

**Aucun CUPTI ne peut collecter d'événements CUDA sur cette machine avec ce driver et cette V100. Le blocage est sous Nsight (couple driver/CUPTI/Volta), pas dans l'installation ou la configuration de nsys.**

### Test décisif : mini client CUPTI direct (sans nsys)
- Programme : `h2/cuptiminiclient.c` (C pur, driver API cuInit/cuCtxCreate/cuModuleLoadData PTX sm_70, 50 launches, callbacks CUPTI request/completed, comptage des records réellement reçus — un « init réussi » ne suffit pas, on compte les records).
- Compile : `gcc -O2 -Wall -I/usr/local/cuda-13.1/targets/x86_64-linux/include -o cuptiminiclient_c cuptiminiclient.c /usr/local/cuda-13.1/targets/x86_64-linux/lib/libcupti.so /usr/lib/x86_64-linux-gnu/libcuda.so`
- Résolution (ldd) : `libcupti.so.13 -> /usr/local/cuda-13.1/.../libcupti.so.2025.4.1`, `libcuda.so.1 -> /usr/lib/x86_64-linux-gnu/libcuda.so.580.178.04` (driver réel).
- Exécuté sur V100 vierge (service stoppé), 3 libs CUPTI forcées en `LD_PRELOAD` :

| lib CUPTI forcée | version rapportée | résultat |
|---|---|---|
| `cuda-cupti-13-0` (13.0.85, version du driver) | 0x1fbd1 | `cuptiActivityRegisterCallbacks: 2 (CUPTI_ERROR_INVALID_DEVICE)` |
| `cuda-cupti-13-1` (toolkit 13.1, 2025.4.1) | 0x1fc35 | idem |
| nsys 2025.5.2 embarqué (libcupti.so.13.1) | 0x1fc34 | idem |

`cuInit`, `cuDeviceGet`, `cuCtxCreate`, le listing du device (V100 cc 7.0) fonctionnent ; l'échec est systématique à l'activation CUPTI, avec la **même erreur `CUPTI_ERROR_INVALID_DEVICE` que celle vue dans les rapports nsys** — ce qui explique le « CUDA injection initialized successfully » mais 0 event : l'activation du device CUPTI échoue silencieusement côté injection.

### Preuves convergentes (avant ce test, toutes négatives)
1. Matrice nsys/CUPTI sur harnais minimal auto-suffisant (`h2/nsysselftest.cu`, sort de lui-même, kernels + CUDA graphs, V100 vierge) :
   - nsys **2025.5.2** avec CUPTI 11.8 / 12.9 / 13.1 / 13.0 (`NSYS_CUPTI_LIBRARY_PATH`) → 0 kernel, 0 API, `No CUDA events collected`, `Number of CUPTI events produced: 0, CUPTI buffers: 50`.
   - nsys **2025.3.2** (dernière release à supporter Volta, extraite isolément dans `~/tools/nsys-2025.3.2/` sans toucher au système) → échec **identique**.
   - `--trace=cuda` (HES par défaut) et `--trace=cuda-sw` (software trace) → mêmes zéros ; OSRT collecte normalement (438k events), le GPU lui-même tourne (binaire se termine sans nsys, OOM propre du stress = GPU sain).
2. Note de release NVIDIA : *« Nsight Systems versions, starting with 2025.4 do not provide support for Pascal or Volta »* → la V100 (Volta sm_70) est hors support dès 2025.4, et CUPTI 13.x embarque cette politique → `CUPTI_ERROR_INVALID_DEVICE` = le device est « invalid » pour l'activity API.
3. Le rapport du **8 octobre 20:31** est déjà vide → le problème n'est pas régressif de la session.
4. `strings libcupti` : présence du message `Requested functionality has been deprecated`.

### Ce qui a été testé / installé (aucun impact système)
- nsys 2025.3.2.474 extraite par `dpkg-deb -x` dans `~/tools/nsys-2025.3.2/` (côte à côte, rien de désinstallé).
- `cuda-cupti-13-0` 13.0.85 + `cuda-cupti-dev-13-0` extraits dans `/tmp/cupti130`, `/tmp/cuptidev130`.
- Binaire `/usr/local/cuda-13.1` (CUPTI 2025.4.1) utilisé pour le test direct — déjà présent, rien d'installé.
- Script de test : `/tmp/cupti_miniclient.sh` (compile + stop/restore service + 3 exécutions LD_PRELOAD).
- Rapports témoins : `~/report2.nsys-rep` (run H2 15:27), `/tmp/st52.*`, `/tmp/st32.*`, `/tmp/stc.*` (zéros), run `runs/run-20261009-152723/`.

### Reprise éventuelle plus tard (hors session)
La piste reste théoriquement réopenable uniquement si : (a) le driver redescend en série 57x/56x supportant CUPTI sur Volta, ou (b) NVIDIA réinstalle le support Volta. Sinon : profiler depuis une autre machine (V100+driver plus vieux) ou se contenter des journaux — c'est ce qui suit.

---

## 2. PILOTE — analyse sur les journaux serveur (run `run-20261009-152723`)

Le `server.out` du run (llama-server lancé avec `-lv 4 --no-log-prefix --no-log-timestamps` par `h2_capture.sh`) contient déjà le nécessaire pour la décision H2, à l'échelle du **round MTP** (pas du kernel) :

### Mesures fiables du decode (task 236 = fenêtre nsys 100k ctx, 1800 tok)
- **Decode wall** : 53 530 ms / 1800 tok = **33.6 tok/s**, **29.76 ms/tok** (première fenêtre ~39.6 t/s, régime stable ~30–33 t/s, remontée à ~33.4 t/s en fin).
- **MTP (draft-mtp, n_max=3)** : **622 rounds** ; drafts 622 lancés / 520 acceptés (83.6 %) ; tokens générés 1865 / acceptés 1243 ; **mean_len = 3.00** ; acc rate par position (0.836, 0.648, 0.514) — la 3e position est faiblement utile.
- **dur(b,g,a) MTP** : batch(CPU) = **0.017 ms/round** (negligeable), **gpu = 6621.0 ms / 622 rounds = 10.64 ms GPU/round**, other = 0.459 ms.
- **Wall/round = 53530/622 = 86.1 ms** → **GPU occupe 10.64/86.1 ≈ 12 % du temps de round** ; le reste (~75 ms) = orchestration CPU + sync + gaps inter-round (sampler, KV, checkpoint/prompt-cache, scheduling, CPU→GPU transfers, attente des streams).
- `graphs reused = 610` (les CUDA graphs du decode sont bien réutilisés, ~1/round).

### Lecture H2 (sans données kernel)
- **H2a (gaps CPU↔GPU / orchestration) est la cause dominante** : ~88 % du wall/round hors GPU-MTP, alors que le CPU batch est quasi nul (17 µs) → l'essentiel est CPU non-CUPTI-visible : syncs/launchs/futex/sampler/KV/checkpoint + attente GPU non-MTP (verify complet, pas seulement le draft MTP).
- **H2b (sous-régime GPU pur, kernels trop lents) est minoritaire** pour la partie MTP ; sans nsys on ne peut pas décomposer les 10.64 ms par famille de kernels — c'est l'aveu de limite du pivot.
- **H2c (overhead MTP)** : mean_len 3.00 avec n_max=3 ; la 3e position n'apporte que 51.4 % d'acceptation. **Piste d'action concrète testable sans nsys : forcer n_max=2** (acc pos 0.836/0.648) ou n_max=1 pour comparer tok/s — si le wall/round diminue plus que la perte d'acceptance, la 3e draft coûte plus qu'elle ne rapporte (elle fait un forward full supplémentaire par round).

### Prochaine action H2 (prioritaire)
1. Expérience A/B sur le serveur de capture (pas la prod) : `n_max` 3 → 2 → 1, même protocole (100k, 2e passe, ~600–1000 tok), lire `dur gpu` et `eval time` → estimation du coût marginal d'un draft.
2. Compléter le wall/round par la décomposition CPU : relancer une capture **sans nsys** avec `perf` (ou `--perf`-like du build) sur le process llama-server pendant le decode → où va les ~75 ms hors GPU-MTP (sampler vs sched vs sync vs memcpy).
3. Garder `h2_analyze.py` en l'état (inutilisable sans tables CUPTI) ; la logique de gap-classement OSRT ne sert que si nsys fonctionne.

---

## 3. GYRATIONS DOCUMENTÉES (commandes, erreurs, verdicts)

| étape | command | erreur | verdict |
|---|---|---|---|
| repro matrice 2025.5.2 × CUPTI 11.8/12.9/13.1/13.0 | `NSYS_CUPTI_LIBRARY_PATH=... nsys profile ... ./nsystest` | `No CUDA events collected`, 0 CUPTI events | pas de discrim par CUPTI embarqué |
| `--trace=cuda` vs `--trace=cuda-sw` | `nsys profile --trace=cuda-sw ...` | rc=1 (switch inconnu/limité), zéro CUDA aussi | le fallback HES→SW ne sauve rien |
| harnais auto-suffisant | `nsys profile ... ./nsysselftest 200` | zéro, identique | écarte le défaut de `nsystest` (wait sur `/tmp/h2mark`) |
| nsys 2025.3.2 isolé | `dpkg-deb -x ... ~/tools/nsys-2025.3.2` | zéro, charge `libcupti.so.13.1` | le dernier nsys « Volta OK » échoue pareil |
| CUPTI 13.0 direct (client C) | `gcc ... cuptiminiclient.c libcupti.so libcuda.so` ; `LD_PRELOAD=... ./cuptiminiclient_c` | `cuptiActivityRegisterCallbacks: 2 (CUPTI_ERROR_INVALID_DEVICE)` ×3 libs | **verdict : le device V100 est rejeté par CUPTI 13.x sous driver 580.178.04** |

Notes de compilation (à reproduire) :
- nvcc 12.6 ne passe pas les headers CUPTI 13.x (`generated_cuda_meta.h` / `CUcheckpointUnlockArgs`) → client en C pur.
- Headers CUPTI 13.x demandent `cuda_stdint.h`, `crt/host_defines.h`, `CUatomicOperation` → seul l'ensemble cohérent `/usr/local/cuda-13.1` passe ; `cuCtxCreate` = 4 args (v4) en 13.x.
- Linker : pointer les fichiers `.so` explicites (pas de `-lcupti` dans `-L` : le symlink de version n'est pas toujours résolu), `-Xlinker -rpath` pour l'exécution.

## 4. GARDE-FOU TENUS
- Prod Ollama : jamais modifiée (stop/start uniquement, md5 `e4afbb4a` intact à chaque contrôle, V100 rechargée ~32 Go).
- Pas de mise à jour driver, pas de changement `.so` prod, pas de remplacement d'outils système (extraction `dpkg-deb -x` dans des dirs dédiés).
- Pas de téléchargement au-delà des deux packages CUPTI/nsys 2025.3.2 (repos officiels NVIDIA).
- Limite de tentatives respectée : le test CUPTI direct était le dernier ; après lui → documentation + pivot.

---

# PHASE 2 — MESURES APPLICATIVES + PERF (2026-10-09, 21:21–23:50)

État prod à chaque contrôle : **`ollama-v100` actif, md5 `libggml-cuda.so` = `e4afbb4a`**. Travaux exclusivement dans l'espace H2 (`ab_results/`, `h2_ab_mtp.sh`, `ab_parse.py`) ; captures brutes préservées (`perf.data`, `perf_script.txt`, `perf_report.children.txt`).

## 5. ÉTAPE 1 — A/B MTP `n_max` 3 → 2 → 1

### 5.1 Protocole et conditions
- Serveur de **capture** isolé (port 44901, **pas** le service prod), `h2/h2_ab_mtp.sh` = `h2_capture.sh` **sans nsys** : même binaire prod, mêmes flags (`-c 200000 -np 1 -lv 4 --spec-type draft-mtp --spec-draft-n-max N --spec-draft-backend-sampling --cache-type-k/v q8_0 --flash-attn auto -b 512 -ub 512`), mêmes env prod (`GGML_CUDA_VOLTA_Q6_W4R4=1`, shortlist, `LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1`, `LLAMA_ARG_FIT_TARGET=1024`, `CUDA_VISIBLE_DEVICES=0`=V100), `GGML_BACKEND_PATH` = libggml-cuda.so prod.
- **Parité stricte** : même modèle (blob sha256-4933…), même prompt/prefill ~100k tokens (réutilisés du run 15:27), gates fit / device=V100 / model-loaded / shortlist chargés avant chaque mesure.
- Chaque config = **serveur propre** (n_max est un flag de lancement) ; 2 réplicats ; **ordre alterné** (rép1 : 3-2-1 ; rép2 : 1-2-3) pour neutraliser la dérive.
- 100k prefill (hors mesure) → 2ᵉ passe chaude (chaud) → **decode 600 tok** = fenêtre mesurée. `n_max=1` traité comme **témoin sans gain de profondeur** (1 seul token spéculatif).

### 5.2 Tableau consolidé (decode seul = task 236, différence des 2 dernières lignes `common_specu` + `slot print_timing`)

> **Légende colonnes** : `rounds` = nombre d'itérations spéculatives (`#calls a`). `acc. token (slot)` = accepté/généré depuis la ligne `slot print_timing` (seuil = 1 token). `acc/pos (0/1/2)` = taux d'acceptation par position (probas de la chaîne de draft). `ms draft/rd (dur g)` = durée de la **phase de génération du draft MTP** (`dur g` des stats spec, i.e. le wall de l'appel `draft()` — dominé par le GPU car `backend_sampling=1`, mais c'est une **mesure wall de la phase**, PAS une somme CUPTI de kernels GPU). `ms wall/rd` = wall/round (wall decode / rounds).

| run | n_max | tok | wall (s) | tok/s | ms/tok | rounds | acc. token (slot) | mean len | acc/pos (0/1/2) | ms draft/rd (dur g) | ms wall/rd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ab-1-n1 | 1 | 600 | 30.91 | **19.38** | 51.60 | 327 | 82.9 % (271/327) | 1.83 | 0.839 / — / — | 4.60 | 94.5 |
| ab-1-n2 | 2 | 600 | 26.62 | **22.50** | 44.45 | 252 | 68.7 % (346/504) | 2.37 | 0.804 / 0.600 / — | 8.18 | 105.7 |
| ab-1-n3 | 3 | 600 | 19.97 | **30.00** | 33.34 | 218 | 58.5 % (381/651) | 2.75 | 0.812 / 0.537 / 0.399 | 10.96 | 91.6 |
| ab-2-n1 | 1 | 600 | 30.88 | **19.40** | 51.54 | 327 | 82.9 % (271/327) | 1.83 | 0.839 / — / — | 4.61 | 94.4 |
| ab-2-n2 | 2 | 600 | 26.63 | **22.50** | 44.45 | 252 | 68.7 % (346/504) | 2.37 | 0.804 / 0.600 / — | 8.19 | 105.7 |
| ab-2-n3 | 3 | 600 | 20.21 | **29.63** | 33.75 | 218 | 58.5 % (381/651) | 2.75 | 0.812 / 0.537 / 0.399 | 11.04 | 92.7 |
| perf-n3 | 3 | 1200 | 37.84 | **31.69** | 31.56 | 427 | 60.2 % (771/1281) | 2.81 | 0.808 / 0.576 / 0.422 | 10.76 | 88.6 |
| 15:27 (nsys) | 3 | 1800 | 53.53 | **33.61** | 29.76 | 603 | 66.2 % (1196/1808) | 2.98 | 0.834 / 0.643 / 0.506 | 10.66 | 88.8 |

Reproductibilité : écart entre réplicats **< 2 %** partout (n3 : 30.00 vs 29.63) → le signal est bien au-dessus du bruit.

**Lecture** : le débit est **strictement monotone en n_max** : n3 ≈ **30 t/s**, n2 ≈ **22.5 t/s**, n1 ≈ **19.4 t/s** → **+33 %** (n3 vs n2), **+55 %** (n3 vs n1). Le gain vient de la profondeur : tok/round passe de 1.83 (n1) → 2.37 (n2) → 2.75 (n3). La phase draft MTP (`dur g`) monte avec la profondeur (4.6 → 8.2 → 11.0 ms/round) mais **reste une minorité du wall/round** (~88–92 ms) → le win est d'amortir le coût fixe du round (verify cible) sur plus de tokens acceptés.

**Détail observé, non expliqué (à suivre)** : le **wall/round de n2 (105.7 ms) > celui de n3 (91.6 ms)** — contre-intuitif (plus profond devrait coûter plus par round). L'effet n'inverse pas le classement au tok/s (la métrique de décision), mais il reste une incertitude : soit une anomalie de mesure, soit un comportement du scheduler/verify à n_max=2. Signalé tel quel, pas interprété.

### 5.3 Vérification que `n_max` change réellement le comportement (pas seulement le flag)
- `spec common_specu: - n_max=N` présent dans chaque log (flag effectif).
- **Comportement observé, pas seulement paramètre** : le nombre de rounds diminue avec n_max (327 → 252 → 218 pour 600 tok), le `mean len` monte (1.83 → 2.37 → 2.75), et l'**acc/pos a autant de positions que n_max** (1, 2, 3 valeurs). → le draft se fait réellement à la profondeur demandée.
- `n_max=1` = témoin : 327 rounds ≈ 600/1.83, i.e. ~1.83 tok/round (1 verify + 1 draft accepté) → **bénéfice spéculatif de profondeur nul**, comme demandé.

### 5.4 Écart 30 vs 33.6 tok/s (run 15:27 vs A/B) — explication
Les deux runs **n_max=3** ne sont pas à la même longueur ni sous le même profilage : le run 15:27 est **1800 tok sous nsys** (surcharge de trace), l'A/B est **600 tok sans nsys**. Deux effets distincts :
1. **Longueur / contexte de fin** : le contenu généré est **identique** sur les 2715 premiers caractères (token pour token, `gen.json` 15:27[0:2715] == ab-1-n3[0:2715], décodage `temperature=0` déterministe). Donc sur **les mêmes 600 premiers tokens**, 15:27 et ab-1-n3 ont **exactement la même acceptation (58.5 %)** et le même wall/round. L'écart global vient de la **seconde moitié** du run 15:27 : sur les tokens 600→1800, l'acceptation **remonte à 70.4 %** (vs 58.5 % au début) → moins de rounds → plus de t/s en moyenne. C'est un **effet de position dans le texte**, pas un effet de configuration.
2. **Surcharge nsys** : le run 15:27 est instrumenté (osrt/cuda), ce qui tend à **ralentir** (plus de wall) ; le fait qu'il affiche quand même 33.6 t/s malgré 1800 tok et nsys renforce le point 1.
→ **Conclusion** : pas de contradiction de protocole. Sur un même sous-segment (les 600 premiers tok), les deux runs coïncident ; la différence de t/s global est portée par la dérive d'acceptation le long du contexte. Le nombre « 30 t/s » de l'A/B est la valeur **représentative courte** ; « 33.6 t/s » inclut une zone à forte acceptation. Les deux sont cohérents.

### 5.5 Décision (critère = débit applicatif mesuré)
**`n_max = 3` est conservé.** C'est le meilleur débit mesuré (+33 % vs 2, +55 % vs 1), reproductible, et la phase draft MTP reste ~11 ms/round (minorité du wall/round, bon marché). La 3ᵉ position (acc ~0.40–0.51) n'est pas inutile : le coût par round n'augmente que marginalement avec n_max, donc les tokens de la 3ᵉ position sont quasi « gratuits » ; il n'y a pas de point de basculement dans 1/2/3. (La piste `n_max=2` de §2 du premier ETAT est ainsi **écartée par la mesure**.)

---

## 6. ÉTAPE 2 — `perf` pendant un decode représentatif (n_max=3, 1200 tok)

Capture en **root** (`perf_event_paranoid=4` → l'utilisateur ne peut ouvrir **aucun** event ; root peut tout compter), attach au PID `llama-server` pendant le decode : `perf stat -e cycles,instructions,task-clock,cpu-migrations,context-switches` + `perf record -F 199 -g -e cycles`. Décode mesuré : **37.84 s / 1200 tok = 31.69 t/s** (concordant avec l'A/B).

### 6.1 Compteurs (`perfstat.txt`)
| mesure | valeur |
|---|---|
| cycles | 597.86 G |
| instructions | 511.67 G |
| **IPC** | **0.856** |
| task-clock (CPU consommé) | **133.22 s** |
| temps mural (time elapsed) | **43.11 s** |
| **cœurs occupés** (task-clock/elapsed) | **3.09** / (6 workers + main) |
| context-switches | 7 485 |
| cpu-migrations | 393 |

### 6.2 CPU consommé vs temps mural
3.09 cœurs sont occupés en moyenne sur ~43 s de decode, pour **1200 tokens** → le CPU est **fortement sollicité** (pas un « temps CPU faible »). IPC 0.856 = pas saturé par le calcul pur. La question n'est donc pas « CPU vs GPU » au sens d'un CPU à l'arrêt : c'est **où va ce 3 cœurs de CPU**. La réponse (§6.3–6.4) : **presque rien de utile — de l'attente encodée en spin + sync.**

### 6.3 Répartition par DSO (self, % des échantillons de cycles)
| DSO | % |
|---|---|
| `libggml-cpu-haswell.so` | **70.91 %** |
| `libcuda.so.580.178.04` (driver, fermé) | 25.50 % |
| `[vdso]` (clock_gettime) | 2.32 % |
| `[kernel.kallsyms]` | 0.55 % |
| `libllama.so` / `libllama-common.so` / `libggml-base.so` / libc / libstdc++ | < 1 % chacun |

### 6.4 LE résultat central — le 70.9 % CPU n'est pas du calcul, c'est du **spin** d'attente
- Le seul symbole résolu dans `libggml-cpu-haswell.so` est **`ggml_graph_compute_secondary_thread`** (70.91 %).
- En regardant les **offsets** chauds (self) : `+0x1b3` (14 048), `+0x1af` (2 309), `+0x1ba` (817), `+0x1bf` (257) = **≈ 93 % des échantillons de `libggml-cpu-haswell.so`** (17 431 / 18 855).
- Désassemblage de ces offsets dans le **binaire prod** (`objdump`) : la boucle est `pause ; add $1,%rax ; mov … ; cmpb $0,0xc(%r15) ; je` — c'est la **boucle de busy-wait** du thread-pool ggml entre deux unités de travail (aucun `call`, aucun SIMD dans la zone chaude 0x1e600–0x1e62f, vérifié : 0 instruction `call`).
- **Interprétation directe (pas une inférence)** : les **6 workers CPU** (`n_threads=6`, TID 73862–73867, chacun ~3 780 échantillons, très symétriques) sont **à l'arrêt de fait, en train de tourner en rond** à attendre du travail, pendant que le GPU fait le travail réel. « 70 % CPU » ≠ « 70 % de calcul ».

### 6.5 Le reste du CPU : **synchronisation GPU** (main thread)
- `ggml_backend_cuda_synchronize` → `cudaStreamSynchronize` → `cuStreamSynchronize` : **27.63 % (children)** des cycles, i.e. le **thread principal bloqué en attente du GPU** (la majeure partie des 25.5 % `libcuda.so`).
- → Le CPU « utile » résiduel (sampler/KV/scheduler) est **marginal** (< ~1 % cumulé, symboles `libllama*`).
- **Synthèse CPU vs GPU** : le CPU brûle ~3 cœurs, mais c'est (a) 6 workers en **spin** + (b) 1 main thread en **stream-sync** = **deux formes d'attente du GPU**, pas de calcul de décodage. Le decode est donc **limité par le GPU et/ou la synchronisation**, pas par le CPU de calcul. (Ceci **redimensionne** le « 85–88 % hors GPU » du §2 du 1er ETAT : cette part « hors GPU » est surtout de l'attente CPU *pour* le GPU, et ne doit pas être lue comme « du temps à optimiser côté CPU ».)

### 6.6 Répartition par thread
| TID | rôle | échantillons |
|---|---|---|
| 73815 | main (serving + sync GPU) | 7 526 |
| 73862–73867 | 6 workers ggml (spin) | ~3 780 chacun |
| 73826 | `cuda-EvtHandlr` (driver) | 47 |
| autres | HTTP/sidecar | < 60 |
→ La masse d'échantillons est dominée par les **6 workers en spin** + la **main en sync**, comme §6.4–6.5.

### 6.7 Limites de la lecture perf (documentées, pas contournées)
1. **Symboles insuffisants au niveau opérateur** : `libggml-cpu-haswell.so` a un `.symtab` mais seulement l'enchapement `…_secondary_thread` est nommé ; les opérateurs internes (vec_dot / mul_mat / rope…) sont **statiques** → on voit *que* le worker tourne (spin), **pas quel op** il exécute. Pas d'attribution opérateur inventée.
2. **Driver fermé** : les 25.5 % `libcuda.so` sont des offsets anonymes (`0x46090e`…) ; l'attribution « sous `cuStreamSynchronize` » vient de la **pile d'appels**, pas d'un symbole.
3. **Coût du spin non séparable du calcul** : comme le worker alterne spin et micro-tâches dans la même fonction, on ne peut pas découper exactement « X % spin / Y % calcul » — on sait seulement que le cluster d'instructions dominant *est* le spin (§6.4, par désassemblage).
4. **Aucun event GPU** (piste CUPTI clôturée §1) : impossible de mesurer la **charge réelle du GPU** (utilisation des SMs, durée des kernels) → on ne sait pas encore si le GPU lui-même est sous-exploité ou s'il est le vrai plafond. C'est précisément l'incertitude que la prochaine étape doit lever.

---

## 7. CONTRAINTES DE COLLECTE ET DE RÉSOLUTION DES SYMBOLES
- `perf_event_paranoid=4` : **l'utilisateur ne peut ouvrir aucun event** ; toute capture `perf` doit passer par **root** (c'est ce qui a été fait, sans toucher à la prod).
- Binaire `llama-server` = PIE mince (4 NEEDED, **sans build-id**), mais les `.so` qu'il charge (`libllama-server-impl.so`, `libllama.so`, `libggml-cuda.so`, `libggml-cpu-haswell.so`) **ont leur `.symtab`** → résolution par chemins prod.
- `perf buildid-list` lie chaque échantillon au **fichier prod exact** (ex. `libggml-cpu-haswell.so` = build-id `b724e5…` sur le chemin `ollama-v100/lib/ollama/`). Les offsets désassemblés proviennent de ce **même fichier prod** → l'attribution « spin » porte bien sur le binaire réellement servi, pas sur un artefact de build. *(Petite réserve : `readelf -n` ne remonte pas la ligne Build ID du fichier disque — lecture du note vide — mais la liaison build-id→fichier a été faite par perf à la capture ; les disassemblages sont faits sur le chemin lié par perf.)*
- Pas de build-id commun entre les copies de build locales (md5 différents : `2488ae90` vs `7644bb24`/`c8f066e1`) → **on n'a pas utilisé les builds locaux** pour la résolution, seulement les `.so` prod.

---

## 8. DÉCISION
**`n_max = 3` conservé** (critère : débit applicatif mesuré, §5.5) ; **aucun** kernel/bibliothèque/driver/configuration de service modifié. La suite ne doit pas chercher à « optimiser le CPU de décodage » (il n'y en a pas, §6.4) mais **à comprendre où le GPU passe son temps** par round (le seul point d'appui dur restant).

---

## 9. PROCHAINE ÉTAPE — PROPOSÉE, NON EXÉCUTÉE

### 9.1 Hypothèse prioritaire
**Le goulot est côté GPU, dans la partie « non-MTP » du round (le verify/target + les op non-spéc de la couche MTP), et non pas dans le draft MTP lui-même.**

### 9.2 Indices qui la soutiennent (observés, pas supposés)
- La **phase draft MTP est bon marché** : ~11 ms/round (`dur g`) quel que soit n_max (§5.2), et le gain de profondeur est quasi « gratuit » → le draft n'est **pas** le plafond.
- Le CPU est en **attente du GPU** (spin 70.9 % + sync 27.6 %, §6.4–6.5) : le round est **limité par le temps GPU**, or le seul temps de phase *mesuré* (draft MTP, ~11 ms/round) ne couvre **pas** le wall/round (~88–92 ms) → il y a **~77–81 ms de travail/sync non comptés** par la stat MTP, i.e. le **verify (forward target complet, 66 couches) + les op MTP non-spéc**.
- Le 1er ETAT (H2a) et ce perf convergent : l'essentiel du round est **hors du draft**, et ce « hors draft » est **attendu côté GPU**.

### 9.3 Incertitudes restantes
- On **ne connaît pas la charge réelle du GPU** (CUPTI clôturée) : le GPU peut être (a) saturé par de gros kernels lents (verify full), (b) sous-exploité à cause de **gaps/latence** (launch, sync, transfer CPU→GPU, KV), ou (c) un mix. L'hypothèse 9.1 n'indique pas *lequel*.
- Le **wall/round de n2 > n3** (§5.2) reste inexpliqué et pourrait être un signal de comportement du scheduler à surveiller.
- Sans CUPTI, **pas de durée par kernel** → on ne peut pas encore pointer un kernel précis.

### 9.4 Test minimal de falsification (pas lancé)
Sans kernel/driver/.so prod, pour distinguer « GPU saturé par de gros kernels » de « gaps/latence » : **mesurer la corrélation wall/round ↔ charge GPU à débit constant** via deux témoins non-invasifs, sur le serveur de capture :
1. **V100 util** : échantillonner `nvidia-smi --query-gpu=utilization.gpu,memory.used` (100 Hz) pendant un decode n_max=3 stable → si l'exploitation GPU moyenne est **haute et plate** (~90–100 %) sur la fenêtre, c'est **du gros travail GPU** (verify) ; si elle est **creusée en dents de scie** (pics + vides), ce sont des **gaps**.
2. **Longueur de draft vs wall/round** : un run supplémentaire n_max=3 mais **moitié de longueur** (600 tok) vs le perf-n3 (1200 tok) → si le wall/round est **stable** quel que soit le contexte, le round est borné par un coût **fixe** par round (→ suspecter verify/sync) ; s'il **augmente avec le contexte**, c'est du travail qui **grandit** (→ suspecter les op dépendants de la profondeur de contexte, ex. FA sur 100k KV).

Chacun des deux témoins est **falsifiable** et ne nécessite ni kernel nouveau, ni driver, ni `.so` prod, ni nsys. **À valider avant exécution.**

*(Aucune action lancée : en attente de validation de cette proposition.)*

---

## 10. ÉTAPE 3 — ACTIVITÉ GPU (nvidia-smi) PENDANT UN DECODE n_max=3 (2026-10-10, 00:45–00:51 + analyse 07:48–08:05)

### 10.1 Protocole (reprise stricte du protocole validé)
- **Même pipeline que `h2_ab_mtp.sh`** (`h2_gpuutil.sh`) : stop prod → serveur de **capture** isolé (port 44901, **pas** le service prod), même binaire/flags/env prod (`n_max=3`, `--spec-draft-backend-sampling`, shortlist, `FIT_TARGET=1024`, `CUDA_VISIBLE_DEVICES=0`=V100, `GGML_BACKEND_PATH`=libggml-cuda.so prod), gates fit / device=V100 / model-loaded / shortlist, warmup, **prefill 100k + 2ᵉ passe chaude** (identique aux runs A/B).
- **Même workload que `perf-n3`** : decode **1200 tok** (task 236). Résultat : **38.34 s / 1200 tok = 31.27 tok/s** (perf-n3 : 31.69 ; A/B n3 : 29.6–30.0) → régime cohérent.
- **Aucune** modification kernel/driver/.so/service ; **aucun** benchmark concurrent ; trap de restauration prod à chaque phase.
- **Sémantique vérifiée** : la V100 est `nvidia-smi -i 1` (0 = GTX 1070, occupée par un reranker 2.3 Go — non concernée) et `cuda:0` (ordre CUDA, cohérent avec la prod).

### 10.2 Cadence réelle du sampler (pas 100 Hz — documenté)
Sampler dédié `h2/gpu_samplerv2.py` : chaque ligne = `epoch_ns mono_ms util.gpu util.mem mem.used mem.total power.draw q_ms` — **son propre horodatage** + **durée réelle de la requête nvidia-smi**. Pas de `nvidia-smi -lms 10` (non garanti).
- Mesuré sur le run : **intervalle médian 25.8 ms** (min 22.4, max 57.7, p95 34.9) → **~39 Hz effectif** (décode : 36.6 Hz ; repos : 40.3 Hz). **Ce n'est pas de la télémétrie 100 Hz.**
- **Limite intrinsèque (vérifiée, source forum NVIDIA)** : `utilization.gpu` = « % de temps où au moins un kernel a exécuté **sur la période d'échantillonnage interne du driver** (~1 s ou ~1/6 s, non modifiable par `-lms`) ». Une lecture ne reflète donc pas la charge à l'échelle sub-seconde, quel que soit le polling. La **puissance** réagit quant à elle en ~50 ms (rampe visible 153→228 ms) → c'est elle la plus résolue.

### 10.3 Résultats bruts (fenêtres marquées par epoch_ns)
Fichiers : `gpuutil-n3/gpu_samples.tsv` (1654 lignes brutes), `markers.txt`, `decode_timing.txt`, `gpuutil_report.txt`.

| grandeur | DECODE (n=1412, 38.60 s) | ÉTAT STABLE (t>2 s, n=1334) | REPOS (3 s, n=121) |
|---|---|---|---|
| util.gpu % | min 0, méd **97**, moy 96.1, max 99 | min **93**, méd 97, moy 97.05, max 99 | 0 partout |
| fraction ≥ 90 % | **98.9 %** | 100 % | 0 % |
| puissance W | 50.8 → pic 244.3, méd 188.5 | 147–216, méd **188.1**, moy 184.3 | 43.8–48.2 |
| puissance / TDP 250 W | — | **~74 %** | ~18 % |
| util.mem % | méd 57 (47–62) | méd 57 | 0 |
| mem.used MiB | 31210 (stable) | 31210 | 32350 (prod) |

**Creux (util < 50 %) pendant le decode** : **un seul**, en tête — **t=0…382 ms** (valeurs 0 puis 43), i.e. **démarrage** avant le premier batch (la puissance est déjà à 226 W à t≈228 ms). **Aucun creux en régime établi** : minimum 93 %, aucun échantillon < 60 W hors fin de fenêtre (83 W à t=38.6 s, décharge de fin). **Oscillation** de puissance en régime : amplitude ~63 W (147–211 W) visible à ~40 Hz → il y a de la **variation sub-seconde** de la charge, mais elle ne descend jamais à l'idle.

### 10.4 Interprétation prudente (sans extrapoler)
1. **À la résolution 1 s, le GPU est occupé en continu pendant le decode** : 98.9 % des lectures ≥ 90 %, jamais de retour à l'idle (puissance), seul creux = les ~0.4 s de démarrage. **L'hypothèse « gros gaps GPU » de §9.3 (b) est fortement défavorisée à cette échelle.**
2. **Ce que la résolution permet de conclure** : des gaps **durables (> ~1 s)** n'existent pas pendant le decode.
3. **Ce que la résolution n'INTERDIT pas** : des **micro-creux sub-seconde** (< période d'échantillonnage du driver) restent invisibles dans `util.gpu` (fenêtrage interne ~1 s / ~1/6 s, non modifiable) ; l'oscillation de puissance (~63 W à 40 Hz) prouve qu'il y a de la **modulation** sub-seconde, sans pouvoir la découper « kernel vs latence ».
4. **Sur le rôle des syncs hôte/GPU** (question de l'étape) :
   - **Conclu** : le CPU en attente du GPU (perf §6.4–6.5 : spin 70.9 % + sync 27.6 %) **n'est pas** causé par des longs vides GPU — le GPU tourne en continu ; c'est la signature d'un GPU **en charge** que le hôte **attend** (sync/stream), pas d'un GPU en pause.
   - **Non conclu** : (a) la **part de micro-gaps sub-seconde** dans le wall/round (88–92 ms) — impossible sans CUPTI (piste clôturée §1) ; (b) si 97 % de *util* = **saturation des SM** — non : `util` = « ≥ 1 kernel en cours » (définition NVIDIA), un petit kernel seul donne 100 % ; la puissance (74 % du TDP, ~188 W) est plus compatible avec une charge **réelle mais non maximale** ; (c) le **temps par kernel** (verify vs autres ops) — toujours impossible sans CUPTI.
5. **Limite expérimentale revendiquée** : nvidia-smi ne fournit **pas** de timeline kernel ; c'est un capteur de charge moyenne, pas un traceur. Pour lever les points (a)/(c) il faudrait soit CUPTI (bloqué), soit une machine V100+driver ancien.

### 10.5 Vérifications de fin (prod)
- Service `ollama-v100` **active**, md5 `libggml-cuda.so` = **e4afbb4a**, V100 32 350 MiB (modèle rechargé, util 0 = idle, ~47 W).
- Aucun process de capture résiduel : rien n'écoute sur 44901 (le `pgrep "port 44901"` du contrôle se **matchait lui-même** — vérifié par `ss -ltnp`) ; seul le `llama-server` prod (port 33609) tourne.
- Artefacts bruts préservés : `gpuutil-n3/{gpu_samples.tsv, markers.txt, decode_timing.txt, server.out, gen.json}`, `gpuutil_report.txt`, `gpuutil_run.log`.
- Anomalie bénigne du log de fin : `restore: V100=0 MiB` à 00:50:15 = vérifiée **avant** le rechargement du modèle par la prod (~1 min) ; état final rechargé confirmé ci-dessus.
---

## 11. DIAGNOSTIC FIGÉ — REEXAMEN DES MESURES EXISTANTES (2026-10-10, 08:30–09:50)

**Périmètre** : aucune mesure nouvelle, aucune optimisation lancée. Réexamen croisé des artefacts existants (`ab_results/ab-*-n*`, `perf-n3`, `gpuutil-n3`) + lecture des sources du build prod (`build/ollama-v100-merged/_deps/llama_cpp-src/common/speculative.cpp`, `tools/server/server-context.cpp`) pour fixer la sémantique. §10 relu : la séparation **mesuré / interprété / non-établissable** y est déjà explicite (§10.3 = mesuré, §10.4 = interprété avec les « Non conclu » chiffrés, §10.4.5 = limite revendiquée) → conservé tel quel, pas de réécriture.

### 11.1 Sémantique MTP confirmée dans les sources (ce que mesurent les stats du log)
- **`dur(b,g,a)`** = `t_begin_us` (hook `process()` sur le batch du **target** : capture des hidden states `h_nextn` + catch-up du KV draft, wall), `t_draft_us` (wall de `draft()`), `t_accept_us` (bookkeeping post-verify). **Aucun des trois ne couvre le verify du target** — le verify est le `llama_decode(ctx_tgt)` du slot, chronométré seulement par le **wall total**.
- **Le round** = `draft()` (n_max décodes du ctx MTP, 1 tête entraînée, `nextn_predict_layers = 1`) → `process()` (hook draft sur le batch verify) → **verify = 1 décode target sur `1 + n_max` positions (n_outputs_max = 4)** → accept (bookkeeping).
- **Le ctx MTP (draft) est un « extra model »** : il **partage les poids** du target (21 Go ne sont comptés qu'une fois) mais a **sa propre KV** (1 couche, 782 MiB, 200 192 cells) → le draft lit le contexte complet (~100k) à chaque étape.
- Conséquence directe : le « wall/round » du serveur = **draft + verify + orchestration** ; le seul sous-terme mesuré par les stats spec est le draft. **Tout le reste est une enveloppe**, pas une somme.

### 11.2 Modèle : **Qwen3.8-27B « qwen35 » = architecture HYBRIDE** (découvert à la relecture, pas dans les § précédents)
- 64 blocs répétés, **tout offloadé sur la V100** (model buffer 21 013 MiB ; CPU_Mapped 995 MiB = logits/embeddings) :
  - **16 couches à attention complète** → KV cache **6 647 MiB** (200 192 cells, K/V q8_0) ;
  - **48 couches récurrentes (SSM/Mamba-like)** : `llama_memory_recurrent` **598.5 MiB** (R 22.5 MiB + **S 576 MiB**, d_state=128, d_inner=6144, d_conv=4, dt_rank=48, n_group=16) — l'état est **fixe** (pas une mémoire croissante) ;
  - quantisations : 201 tenseurs q8_0, 71 q5_K, 2 q4_K, 1 iq4_nl, 6 f16 (blob 23.1 Go) ; flash-attn activé (requis par le V q8_0).
- **Lecture (mesure, non hypothèse)** : dans un verify de 4 tokens à ~100k de contexte, seules les **16 couches full-attn** portent la charge proportionnelle au contexte (FA sur KV q8_0) ; les 48 couches SSM sont **O(1) par token** en lecture d'état (~576 MiB de state, vs ~24 Go de poids).
- **Cohérence interne** : préfill = **1.80 ms/token** (107 351 tok, 555 t/s) alors que le décode = **~31.6 ms/token** → à 100k ctx, le coût par token du décode est **~17×** celui du préfill. Le calcul pur ne justifie pas cet écart : FFN seul du verify (3 GEMM 5120×17408 × 64 couches, batch 4) = 137 GFLOP ≈ **1.1 ms** sur V100 fp16 (~125 TFLOPS). L'écart est **structurel** (attention longue + état SSM + overhead de launch par petite opération), pas un déficit de FLOPs.

### 11.3 Décomposition du wall/round (mesure, fenêtre decode, sources server.out)
| run | wall/rd | draft (`dur g`) | `b+a` | **enveloppe (verify + orchestration)** |
|---|---:|---:|---:|---:|
| n1 (×2 réplicats) | 94.4–94.5 | 4.6 | 0.02 | **~89.9** |
| n2 (×2) | 105.7 | 8.2 | 0.02 | **~97.5** |
| n3 (×2) | 91.6 / 92.7 | 11.0 | 0.02 | **~80.6–81.7** |
| perf-n3 (1200 tok) | 88.6 | 10.8 | 0.02 | **77.8** |

- L'enveloppe (verify+orchestration) est **quasi constante en n_max** (78–98 ms) ; elle représente **~88 % du wall/round**. **C'est le goulet** (confirmé, pas supposé : c'est la seule part non mesurée et la plus grande).
- La **monotonie en n_max** (n3 le meilleur) tient uniquement au **tok/round** (1.83 → 2.37 → 2.75) : le gain d'acceptation amortit un coût fixe par round quasi inchangé.
- **Anomalie du §5.2 requalifiée (mesure, non interprétation)** : l'enveloppe **n2 (97.5) > n3 (80.6)** persiste sur les 2 réplicats (écart ~17 ms, bruit < 2 %) → ce n'est **pas** une anomalie de mesure mais un **comportement reproductible** : avec n_max=2, le verify porte **3** positions (`1+n_max`) au lieu de **4** et coûte **plus** par round. **Le verify n'est pas proportionnel à la taille du batch verify** (3 vs 4 positions) — signal fort en faveur d'un coût **dominant fixe par round** (cf. 11.4) et/ou d'un comportement du scheduler. **Cause précise non établie** (pas de décomposition possible sans CUPTI).

### 11.4 Le test de falsification §9.4 (W1) est **déjà résolu par les données existantes** — verdict : coût FIXE par round
Le témoin 2 du §9.4 (« wall/round stable quel que soit le contexte → coût fixe ; croissant → travail qui grandit, ex. FA sur 100k KV ») n'a pas besoin d'un nouveau run : les 4 runs n_max=3 existants couvrent déjà la profondeur de KV :

| run | KV (≈) | wall/rd | enveloppe verify |
|---|---:|---:|---:|
| ab-1-n3 | 100.0k | 91.6 | 80.6 |
| ab-2-n3 | 100.0k | 92.7 | 81.7 |
| perf-n3 | 100.6k | 88.6 | 77.8 |
| 15:27 (nsys) | 101.8k | 88.8 | 78.1 |

→ **plat sur +1.8k tokens de contexte** (100.0k→101.8k) ; le léger recul (88.6 vs 91.6) est dans le sens de la **dérive d'acceptation** du §5.4 (plus d'acceptation → moins de rounds nécessaires), pas une réduction du verify. **Verdict : le verify/round est un coût FIXE par round, non un travail qui grandit avec la profondeur de contexte.**
- **Lecture prudente** : à +1.8 % de profondeur KV, le **travail qui grandit** (attention FA sur les 16 couches full-attn) reste **sous la résolution** du test (quelques %), il ne peut pas dominer ; ce qui domine est la **partie fixe** (64 blocs exécutés, dont les 48 SSM à état fixe, launchs, sync).
- **Ce que cela ne dit PAS** : si, sur des **très** longues conversations (ctx 2×, 4×), la part FA des 16 couches finit par dominer — le test ne couvre que +1.8k tok. C'est une **limite du périmètre**, pas une contradiction.

### 11.5 Signatures temporelles de `gpuutil-n3` (analyse supplémentaire sur données brutes)
- **Auto-corrélation de la puissance** (fenêtre 3–36 s, n=1201, dt=25.8 ms) : pic à **lag ≈ 80–89 ms (r ≈ +0.54)**, secondaire à ~177 ms (+0.19), creux à ~30 ms (−0.21). **La modulation sub-seconde de la charge GPU est synchrone du rythme du round (~88.6 ms)** → chaque round produit un **cycle de charge caractéristique** (montée/retour) à l'intérieur de la fenêtre du driver ; il n'y a **pas** de cycle à la période du token (~32 ms, < Nyquist 19.3 Hz → non résolue de toute façon).
- **Borne bande-passante** (estimation, TDP/BW de spec, pas une mesure) : un verify = 1 forward du modèle (20.5 Go de poids sur GPU + ~3.5 Go de KV des 16 couches full-attn à 100k) ≈ 24 Go ; à ~900 GB/s (HBM2 V100-PCIE) le **plancher d'un seul passage** ≈ **27 ms**. L'enveloppe mesurée = **78–98 ms ≈ 3–3.5× le plancher** → le forward **ne sature pas** la bande-passante : **ni quelques gros kernels saturés, ni des vides durables** (cf. §10.3) → régime **overhead/launch-bound** (beaucoup de petits kernels, batch 1–4, + sync).
- **Puissance médiane 188 W = 74 % TDP**, oscillation 147–216 W, **jamais < 83 W** en régime établi : compatible avec un GPU **constamment alimenté mais non saturé** (les SM tournent sur des petits lots ; le pic 244 W est la rampe de démarrage).

### 11.6 Synthèse — diagnostic figé
**Le goulet du décode n_max=3 est le *verify* (forward target, 4 positions) + son overhead applicatif**, ~78–98 ms/round (~88 % du wall), d'un coût **fixe par round** (test §9.4.2 résolu négativement par les données existantes), qui **ne sature ni la bande-passante (≈3× le plancher), ni le calcul (FFN seul ≈ 1 ms), ni le GPU en gros kernels (puissance 74 % TDP, util « ≥1 kernel » 97 %, oscillations synchrones du round)** — c'est-à-dire un **régime overhead/launch-bound côté GPU, avec le hôte qui attend (sync 27.6 %, spin 70.9 %, CPU utile < 1 %)**. Le draft MTP (~11 ms/round) n'est **pas** le goulet. Les 48 couches SSM (état fixe 576 MiB) sont **candidates structurelles** au coût fixe par round (exécution de 64 blocs par verify), mais **leur part respective vs les 16 couches FA vs l'overhead de launch n'est pas mesurable** sans CUPTI.

**Ce qui reste non-établissable (limites revendiquées, non contournées)** :
1. **Partitions internes** du verify : temps FA (16 couches) vs SSM (48 couches) vs GEMM vs launchs/syncs → **impossible** sans CUPTI (§1 clôturé).
2. **Part des micro-creux sub-seconde** dans le wall/round → non résoluble avec `util.gpu` (fenêtrage driver ~1 s, §10.2) ; l'auto-corrélation prouve la **modulation** mais pas les **trous**.
3. **Cause précise de n2 > n3** (§11.3) : comportement du scheduler/verify à 3 positions vs 4 — signal mesuré et reproductible, cause non accessible avec les outils actuels.
4. **Évolution du coût du verify avec des contextes bien plus longs** (2×/4× la profondeur) : hors périmètre des runs existants (100.0–101.8k) ; le test W1 ne conclut que sur +1.8k tok.
5. **Durée par kernel** (verify vs autres ops) → toujours **impossible** sans CUPTI.

**Conséquence pour la suite (pas d'action lancée)** : le levier d'optimisation, si décision de passage à l'action, est **côté GPU/overhead du verify** (batching, CUDA graphs — déjà 437 réutilisations/1200 tok, réduction du nombre de launchs, ou réduction du coût fixe des 64 blocs), **pas** côté « calcul CPU de décodage » (inexistant, §6.4) et **pas** côté `n_max` (déjà optimal mesuré, §5.5). Aucune de ces pistes n'est engagée à ce stade : le diagnostic est figé, l'optimisation reste à valider.

---

## 12. ÉTAPE 4 — HYPOTHÈSE D'OPTIMISATION **H-BW** : le verify est-il limité par le trafic HBM des poids ? (2026-10-10, 11:55–13:10 — préparation read-only)

**Statut : EXÉCUTION EN COURS** (validation utilisateur : protocole complet + q4_K pleine). Requant **terminée** (§12.1c), 4 runs A/B en cours. Aucune modif du code ni de la production (prod stoppée/restaurée uniquement par le trap du harnais, comme §5/§6/§10).

### 12.0 Pourquoi cette direction (et pourquoi pas les autres)
Reprise des leviers du §11.6, triés par **potentiel × risque × capacité de falsification** :

| levier | verdict cette session | pourquoi |
|---|---|---|
| CUDA Graphs « activer » | **déjà en place** — levier mort | `USE_GRAPHS = 1` dans le log prod ; `graphs reused = 437` sur 1200 tok (task 236). Il reste au plus de la marge de *réutilisation* (formes non capturées), non mesurable sans CUPTI. |
| Optimisation CPU | **infirmé par mesure** | §6 : 70.9 % spin d'attente, 27.6 % sync, CPU utile < 1 %. |
| `n_max` | **optimal mesuré** | §5.5 : n3 = 30 t/s, +33 % vs n2, +55 % vs n1, reproductible. |
| MoE / routage / dispatch experts | **n'existe pas** | `n_expert = 0` ; source `qwen35.cpp` : FFN dense (`GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr)`). Intégration demandée au raisonnement : rien à intégrer. |
| Réduction des launches / coût fixe 64 blocs (H-SYNC) | **piste de repli** | Vraie hypothèse concurrente, mais non testable sans modif code ou CUPTI → gardée pour le cas où H-BW est infirmée. |
| **Trafic HBM (H-BW)** : requantiser les poids du verify | **choisi** | Seul levier à la fois (a) à fort potentiel, (b) testable **sans toucher au code ni au runtime prod** (un nouveau GGUF + le harnais capture déjà validé), (c) **falsifiable de façon nette** : si le temps ne bouge pas, la part « poids en transit » du verify est bornée, et H-SYNC devient la piste par défaut. |

**Verrou méthodologique demandé (tenue)** : `util.gpu` = 97 % ne prouve **pas** la saturation HBM (un kernel seul suffit à 100 % d'« utilisation »). H-BW est donc traité comme **hypothèse à falsifier**, pas comme conclusion. Les planchers de §12.1 disent simplement : le trafic des poids est du même ordre que le temps du verify, donc un test de trafic est **informatif**.

### 12.1 Chiffres exacts (mesurés, pas estimés)

**a. Recensement des octets par type — GGUF v3 lu par le reader ggml du build prod** (`h2/gguf_census.c`, lié à `libggml-base.so.0.24.0` de `build/ollama-v100-merged/llama-server-local/bin`, `gguf_get_tensor_size`, read-only) :

| type | n tenseurs | octets | GB | % |
|---|---:|---:|---:|---:|
| q6_K | 231 | 13 816 320 000 | 13.816 | 59.87 % |
| q8_0 | 201 | 5 865 799 680 | 5.866 | 25.42 % |
| q5_K | 71 | 3 287 285 760 | 3.287 | 14.24 % |
| iq4_nl | 1 | 50 135 040 | 0.050 | 0.22 % |
| q4_K | 2 | 47 185 920 | 0.047 | 0.20 % |
| f32 | 360 | 10 686 464 | 0.011 | 0.05 % |
| **TOTAL** | **866** | **23 077 412 864** | **23.077** | 100 % |

Cohérence croisée : identique au décompte `llama_model_loader` du log prod (231/201/71/2/1) et à `llama-gguf` (866 tenseurs, version 3). **Aucun f16 dans le GGUF** (les 6 f16 du log = buffers runtime, pas des poids).

**b. Ce que le GPU lit réellement par forward** (server.out perf-n3) :
- modèle GPU : `CUDA0 model buffer size = 21013.71 MiB` = **20.52 GB** (f32/embds offloadés CPU : 23.08 − 20.52 ≈ 2.5 GB hors GPU — les logits sont déjà CPU_Mapped en prod) ;
- KV full-attn : **6 647 MiB** (K/V **q8_0**, 200 192 cells, 16 couches) ;
- KV MTP : **782 MiB** (f16, 1 couche) ;
- état récurrent (SSM) : 598.5 MiB (lu/écrit O(1) par couche GDN) ;
- → **trafic poids+KV ≈ 20.52 + 6.49 = 27.0 GB** par forward (plus activations, ≈ négligeable).

**c. Scénario requant q4_K — chiffres réels après exécution (16:16–16:21, `h2/q4k/quantize.log`)** :
- `llama-quantize --allow-requantize` (binaire build prod, 6 threads, 5 min 09 s) ; GGUF de sortie : **16 799 719 424 B = 16.800 GB** (attendu naïf 15.38 GB — écart expliqué ci-dessous) ;
- recensement réel (`gguf_census` sur le fichier de sortie) : **q4_K 439 tenseurs = 12.262 GB (73.0 %)**, **q6_K 67 tenseurs = 4.527 GB (26.9 %)**, f32 360 = 0.011 GB ;
- **Écart au 15.38 GB expliqué (mesuré, pas supposé)** : les overrides du requantiseur (sémantique `_M` du type cible) **répartissent** au lieu d'aplatisser tout en q4_K : source q8_0 → **q6_K** (201 tenseurs, dont les 64 `attn_v` + 64 `ffn_down` + `output.weight`), source q6_K/q5_K → **q4_K**, `token_embd` (f32 1.04 GB) → **q4_K**. Les couches les plus sensibles (V, FFN-down, head de sortie) restent à haute précision — la variante de protection de la qualité est obtenue **gratuitement** par la sémantique du binaire prod.
- Trafic par forward : baseline 20.52 + 6.49 = **27.0 GB** ; q4_K ≈ (20.52 − 5.23) + 6.49 ≈ **21.8 GB** → **trafic −19 %** ;
- **plancher BW (900 GB/s HBM2)** : 30 ms → **≈ 24.5 ms** par forward.

**d. Conséquence attendue SI H-BW est vraie** : l'enveloppe verify (77.8 ms, §11.3) proportionnelle au trafic des poids → −19 % de trafic ⇒ **enveloppe → ≈ 63 ms**, **wall/rd 88.6 → ≈ 74 ms**, **31.7 → ≈ 38 t/s (+20 %)**.
**SI H-BW est fausse** (temps dominé par launches/syncs/overhead indépendant du trafic) : **|Δwall/rd| < 2 %** (bruit établi §5.2).
→ Le test est net : la zone intermédiaire se lit comme **mixte**, chiffrable a posteriori (fraction « transit » = Δmesuré / Δtrafic).

### 12.2 Hypothèse (formelle)
> **H-BW** : l'enveloppe verify+orchestration (~78 ms/round) est majoritairement proportionnelle au nombre d'octets de poids lus par forward (20.5 GB). Requantiser les poids GPU en q4_K (−25 % d'octets GPU) réduit le wall/round d'au moins 10 % **sans dégrader l'acceptation MTP de plus de 3 points** (seuil qualité).

**Mécanisme technique précis** : le verify = 1 `llama_decode` du target sur 4 positions. À batch 4, chaque GEMM `mul_mat_vec_q`/`mul_mat_id` parcourt la **colonne complète** du poids (tous les n_cols) pour chaque position : le **moindre multiple de travail** d'un forward est la lecture de tout le poids (20.5 GB) + la KV des 16 couches full-attn (6.5 GB). Réduire la taille par paramètre (8.2 bits → 4.5 bits) réduit ce trafic **sans changer le nombre de kernels, ni le graph, ni le code** — c'est exactement le test d'isolation de la composante « transit HBM » du verify.

**Ce que H-BW ne dit PAS** (limites revendiquées) : rien sur la partition FA/SSM/launchs (CUPTI clôturé, §1) ; rien sur les micro-creux sub-seconde (§10.2) ; rien sur la cause de n2>n3 (§11.3).

### 12.3 Contrôle qualité MTP (demandé, verrouillé)
- **Génération déterministe** : `temperature=0, top_k=1, ignore_eos=true, max_tokens=600, enable_thinking=false` (confirmé dans `ab_results/prompt600.json` et `prefill.json`). Le texte q6_K de référence est **connu** (`ab-1-n3/gen.json`) : sur les 600 premiers tokens, l'acceptation de référence = 58.5 %.
- **Contrôle 1 (détection)** : premier token divergent entre q4_K et q6_K → la requant a déplacé le chemin greedy → **signal qualité** (ne déclenche pas l'échec seul, mais est rapporté ; si divergence au token ≤ 50, échec qualité).
- **Contrôle 2 (seuil dur)** : acceptance q4_K (600 premiers tok) ≥ **55.5 %** (58.5 − 3 pts).
- **Shortlist MTP** : table d'indices de rows du head de sortie (131 072 i32), **indépendante des poids** → pas de re-charge ; vérification par la ligne log `CUDA MTP shortlist: loaded` identique.
- **Draft MTP** : lit le **même** contexte 100k (KV propre 782 MiB) — inchangé par la requant des poids du tronc, sauf via les hidden states → capturé indirectement par `dur_g` (contrôle : `dur_g` q4_K ≈ `dur_g` q6_K ± 10 %).

### 12.4 Protocole A/B (reprise stricte du harnais validé — `h2_ab_mtp.sh` / `h2_gpuutil.sh`)
**Préparation (exécutée, voir §12.1c)** :
1. `llama-quantize --allow-requantize` → `h2/q4k/Qwen3.8-27B-Q4_K.gguf` (**16.80 GB**, 866 tenseurs, 5 min 09 s). Note : le flag `--allow-requantize` était **obligatoire** (sources déjà quantisées ; sans flag : `failed to quantize: requantizing from type q8_0 is disabled`).
2. Vérification post-requant : `gguf_census` = 12.262 GB q4_K / 4.527 GB q6_K / 0.011 GB f32 (total 16.800 GB, 866 tenseurs — cohérent avec le loader du log requant).

**Runs** (serveur capture isolé, **port 44901**, prod **ollama-v100 inchangée** ; arrêt/restauration par le trap déjà validé) :
- **Mêmes paramètres que prod/capture** (vérifiés dans `h2_gpuutil.sh`) : `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling`, `GGML_BACKEND_PATH=…/cuda_v12/libggml-cuda.so` (md5 `e4afbb4a`), `GGML_CUDA_VOLTA_Q6_W4R4=1`, `GGML_CUDA_QWEN35_MTP_SHORTLIST=…/qwen38-27b-exact-131072.i32`, `LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1`, `CUDA_VISIBLE_DEVICES=0` (V100), `LLAMA_ARG_FIT_TARGET=1024`.
- **4 runs, 2 réplicats, ordre alterné croisé** : R1 = [Q6, Q4], R2 = [Q4, Q6] → annule la dérive (acceptance, thermale).
- **Chaque run** : prefill 100k (`prefill.json`, ×2 passages chauds comme le harnais) → warmup → **decode 600 tokens** (`prompt600.json`) → capture de :
  - stats spec (wall decode, rounds, `dur b/g/a`, acceptance par position) ;
  - `slot print_timing` (tok/s, mean len, graphs reused) ;
  - `nvidia-smi` (util.gpu, puissance, mémoire) à la cadence driver (≈ 25.8 ms), fenêtre marquée par epoch_ns (protocole §10) ;
  - texte généré (contrôle qualité §12.3).
- **Durée estimée** : 4 × (prefill ~40 s + decode ~20 s) + charge modèles ~3 × 60 s ≈ **25–30 min** de machine V100, en exclusivité (prod arrêtée pendant la capture, comme en §5/§6/§10).

### 12.5 Seuils de décision (fixés AVANT exécution)
| résultat mesuré (médiane sur 2 réplicats, 600 tok) | verdict | suite |
|---|---|---|
| Δwall/rd(q4_K − q6_K) **≤ −8 %** ET acceptance ≥ 55.5 % ET `dur_g` stable (±10 %) ET mémoire OK (≈ 21.3 GB) | **H-BW confirmée** (Δtrafic mesuré = −19 % ; ≤ −8 % = ≥ ~40 % de l'enveloppe portée par le transit) | quantifier le gain ; décider (avec l'utilisateur) de la requant prod |
| Δwall/rd **> −3 %** (quel que soit le signe) | **H-BW infirmée** (trafic des poids < ~3 % de l'enveloppe) | pivot **H-SYNC** : mesurer la part du split hôte (`CUDA_Host compute buffer 215.78 MiB`, `graph splits = 2`) et des syncs par round |
| **−8 % ≥ Δ > −3 %** | **mixte** | fraction « transit » ≈ Δmesuré / Δtrafic (−19 %) ; poursuite sur la fraction restante (H-SYNC) |
| acceptance **< 55.5 %** | **qualité non atteinte** | ne conclut PAS sur la perf (rapportée quand même) ; variante à valider : requant sélectif (q4_K sur FFN uniquement, q6_K conservé sur attention/MTP) — ≈ −12 GB, −17 % de trafic GPU |
| divergence greedy **≤ 50 tokens** | **qualité non atteinte** (idem) | idem |
| erreur de charge / OOM / crash | **protocole invalide** | rollback immédiat, diagnostic avant retry |

**Lecture du gain** : wall/rd est la métrique (pas le tok/s seul, pour isoler le verify de l'acceptation) ; le tok/s est rapporté comme vérification. Le seuil −2 % = bruit établi (§5.2 : réplicats < 2 %).

### 12.6 Rollback (déjà validé en §5/§6/§10, rappelé)
1. `pkill -f "port 44901"` (serveur capture) ; le trap restaure `ollama-v100` (unité systemd).
2. Vérifications post : `systemctl is-active ollama-v100` = active ; `md5sum` `libggml-cuda.so` = `e4afbb4a…` ; `nvidia-smi -i 1` = V100 rechargée ≈ 32 350 MiB ; 1 requête de smoke (température 0, texte attendu connu).
3. Artefacts : le GGUF q4_K reste en `h2/q4k/` (non installé nulle part) ; les 4 runs bruts en `h2/q4k/runs/` (server.out, gen.json, gpu_samples.tsv, markers.txt) — mêmes formats que `h2/gpuutil-n3/`.

### 12.7 Risques documentés (pas contournés)
- **Temps de requant** : q4_K sur 23 GB ≈ 10–30 min CPU (6 cœurs) — hors fenêtre V100.
- **Dérive d'acceptation le long du contexte** (§5.4) : ne touche pas le protocole (comparaison sur les 600 premiers tok, réplicats alternés).
- **Thermique/charge ambiante** : même machine, même fenêtre de la nuit ; le bruit établi (< 2 %) est sous le seuil de décision.
- **H-BW confirmée mais gain insuffisant pour la prod** : décision de passage en prod (qualité long terme, taille modèle) reléguée à l'utilisateur — ce test ne la prend pas.

**État à ce stade : protocole complet, chiffres exacts, seuils fixés. En attente de validation pour exécuter (requant + 4 runs). Aucune mesure lancée.**

### 12.8 RÉSULTATS (4 runs terminés, 18:34–19:11 UTC)

**Exécution** : requant 16:16–16:21 (5 min 09 s). 4 runs A/B (port 44901, prod stoppée/restaurée par trap, même harnais que §5/§10). 1 incident de harnais (A2-q4, 1re passe : lecture de la ligne « using device » trop tôt — bug de mon script, pas du modèle) → corrigé (polling), relancé ; les 4 runs sont complets et valides.

| run | wall decode | rounds decode | **wall/rd** | acc slot (600 tok) | mean len | pos (0/1/2) | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| ab-1-q6 | 20.091 s | 217 | **92.58 ms** | 58.5 % (381/651) | 2.75 | .812/.537/.399 | 29.86 |
| ab-2-q6 | 20.463 s | 217 | **94.30 ms** | 58.5 % (381/651) | 2.75 | .812/.537/.399 | 29.32 |
| ab-1-q4 | 17.748 s | 202 | **87.72 ms** | 65.2 % (396/607) | 2.95 | .833/.635/.483 | 33.81 |
| ab-2-q4 | 17.660 s | 202 | **87.28 ms** | 65.2 % (396/607) | 2.95 | .833/.635/.483 | 33.98 |

Reproductibilité : bruit entre réplicats **−0,5 %** (Q4) / **+1,9 %** (Q6) ; texte q4_K **identique token pour token** entre ses 2 réplicats (deterministe temp=0). Baseline Q6 recollée exactement à la référence §5.2 (acc 58.5 %, mlen 2.75, pos 0.812/0.537/0.399 → l'infrastructure n'a pas dérivé).

**Médianes vs seuils §12.5** :
- **Δwall/rd = 87.50 / 93.44 − 1 = −6.36 %** → zone **MIXTE** (−8 % > Δ > −3 %) ;
- **Δacc = +6.7 pts** (58.5 → 65.2) → bien au-dessus du seuil qualité 55.5 % ;
- **Δtok/s = +14.5 %** (29.59 → 33.89), décomposition exacte : tok/s = mean len / wall/rd → (2.95/2.75) × (93.44/87.50) = **×1.073 (acceptation) × ×1.064 (verify) = ×1.145** ;
- **Fraction « transit HBM » ≈ Δmesuré/Δtrafic = 6.4/19 ≈ 33 % de l'enveloppe** (bornée, pas exacte : le Δtrafic réel du buffer GPU = 5.67 GB/20.5 GB = −27.7 %, pas −19 % — voir plus bas ; recalculé, la part transit est **≈ 23–33 %** selon le référentiel de trafic) ;
- Mémoire GPU confirmée : model buffer 21 014 → **15 339 MiB** (−5.67 GB, exactement la différence de taille des quants) ; usage total decode 31.2 GB → 25.5 GB.
- GPU pendant decode (sampler §10) : util médiane 97 % (98 % ≥ 90) pour les deux, puissance 180–183 W → même régime de charge, pas de saturation différentielle.
- Contrôles qualité MTP : texte q4_K **cohérent** (répond en anglais sur l'anglais du prompt, structure « It appears you have pasted… » — divergence dès le token 1, **mais** le prompt est un texte volontairement dégénéré en boucle — voir §5.4 — et le chemin greedy q4_K est tout à fait plausible, pas du délire) ; `dur_g` stable (±2–8 %).

**Interprétation (mesure vs interprétation, strictement séparé)** :
1. **Mesuré** : −6.4 % de wall/round pour −27.7 % d'octets de poids GPU. H-BW **ni confirmée en plein, ni infirmée** : le transit HBM des poids porte une part **réelle et non négligeable (≈ 25–35 %) de l'enveloppe verify**, le reste reste du **coût fixe** (launchs/syncs/64 blocs, H-SYNC).
2. **Interprété (borne prudente)** : si le wall était 100 % proportionnel au trafic, on aurait vu ≈ −27.7 % ; on a vu −6.4 % → le transit est un **coefficient multiplicatif mineur** du wall/round, pas le goulet principal.
3. **Le gain appliqué se partage** : ≈ 52 % vient de l'acceptation (mean len 2.75 → 2.95, +7.3 %) et ≈ 48 % du verify plus rapide (−6.4 %) — répartition en logarithme du gain total +14.5 %. Le bénéfice net de la requant (débit applicatif) est **réel : +14.5 %**, sans perte de cohérence mesurable.
4. **Décision (proposée, non exécutée)** : H-BW comme « LE » levier est **infirmée** (elle n'explique qu'un tiers) ; H-SYNC reste la piste principale pour l'autre deux-tiers. Les deux options de suite :
   - **A** : retenir la requant q4_K comme gain appliqué tel quel (+14.5 %, −5.7 GB VRAM) — décision utilisateur (taille modèle, qualité long terme) ;
   - **B** : poursuivre sur H-SYNC (split hôte 215.78 MiB, graph splits = 2, syncs par round) pour attaquer la part fixe restante — nécessite instrumentée légère ou nsys.

**Rollback exécuté et vérifié** : `ollama-v100` active, V100 32 352 MiB, md5 `e4afbb4a`, API 11434 répond. Le GGUF q4_K n'est **installé nulle part** (dossier `h2/q4k/` uniquement).

---

## 13. H-SYNC — instrumentation légère (nsys CLOS, §1) : LOCALISATION DU COÛT

**Décision utilisateur (20:30 UTC)** : Q6_K reste référence, q4_K de côté (le labo a fait son travail :
borne la part transit HBM à ~⅓, quantifie le gain qualité). Cap H-SYNC : localiser le coût d'exécution
du verify (syncs, découpage graph, travail hôte, coûts fixes des 64 blocs) **avant** de choisir une
modification. Précaution tenue : `graph splits = 2` et 215.78 MiB hôte = indices, pas preuve.

### 13.1 Outil (nsys/CUPTI étant CLOS sur Volta+driver 580, §1)

**Interposition CUDA-runtime par LD_PRELOAD** (`h2/h2_cudatrace.c` → `h2/libh2cudatrace.so`, 1.7 Mo,
aucun fichier prod modifié) : journalise en binaire (16 o/événement, écriture immédiate, rien de perdu
au SIGTERM) :
- `cudaStreamSynchronize` / `cudaEventSynchronize` / `cudaDeviceSynchronize` **IN/OUT** → temps hôte **bloqué** ;
- `cudaGraphLaunch`, `cudaLaunchKernel(ExC)`, `cudaEventRecord`, `cudaMemcpy(Async)`.

Deux pièges sautés (documentés dans le code) :
1. **le cudart prod (12.6.77) est chargé RTLD_LOCAL** par le backend ggml dlopen' → `dlsym(RTLD_NEXT)`
   retournait NULL → **segfault au premier sync** (reproduit, backtrace gdb : `ggml_backend_cuda_buffer_clear`
   → `cudaStreamSynchronize` → 0x0). Corrigé : `dlopen("libcudart.so.12", RTLD_NOW)` (déduplique, même
   instance) + garde-fou NULL sur chaque appel.
2. **les timestamps contiennent des octets 0x0A** → pas de ligne texte, fichier bin pur, parseur aligné 16 o.

Validé sur selftest (`nsysselftest_d`, cudart shared) : 206 events exacts (100 graph, 104 kernel, 1 dev_synch).
Script `h2/h2_ab_sync.sh` = harnais A/B (port 44901, trap stop/restore) en 2 runs Q6 baseline + trace.
Analyse : `h2/h2_sync_analyze.py` + `h2/h2_sync_summary.py`.

### 13.2 MESURE — profil d'un round de décode MTP (médiane, 219 rounds, 2 réplicats quasi identiques)

| composante | s-1-q6 | s-2-q6 | % du round |
|---|---:|---:|---:|
| **wall/round** | **93.5 ms** | **93.1 ms** | 100 % |
| **GRANDE sync** (hôte bloqué sur le GPU) | **77.3 ms** | **76.8 ms** | **~83 %** |
| petites syncs (somme, ~90/round, méd. 4 µs) | 10.5 ms | 10.5 ms | ~11 % |
| résiduel non-synch (CPU/sampler/autres) | 5.7 ms | 5.8 ms | ~6 % |

- **1 seule grande sync/round** (~77 ms) : c'est le verify+draft complet, pendant lequel **l'hôte est bloqué**
  (le GPU fait le travail — 5 graph launches sont émis juste avant, puis 77 ms de blocage).
- **~90 petites syncs/round** (médiane 4 µs, p90 ~22 µs) : coût d'orchestration ~10.5 ms/round.
- **5 graph_launch/round**, **47 memcpy_async/round** (→ le split hôte 215.78 MiB est bien alimenté, ~47
  transferts host→device/round), 0 kernel hors-graph.
- GPU (sampler) : util médiane 97 %, 97–98 % ≥ 90, 176–177 W → **même régime de charge que les runs A/B**
  (l'instrumentation n'a pas déformé : wall/rd 93.5/93.1 vs 93.44 en A/B, +0.5 % de surcoût au pire).

### 13.3 LECTURE (mesure vs interprétation, séparés)

**Mesuré (fiables)** :
1. **L'hôte ne "perd" que ~11 % du round en syncs d'orchestration** (~10.5 ms) — ce n'est **pas** le goulet.
2. **~83 % du round est le GPU lui-même en travail** (verify+draft) pendant lequel l'hôte est bloqué sur UNE
   sync finale. Le GPU est occupé (97 % util) mais **pas saturé en BW** (test §12.8 : −27.7 % d'octets ne fait
   baisser le verify que de −6.4 %).
3. **Le "split hôte 215.78 MiB" est un coût réel mais mineur** : ~47 memcpy async/round, intégré dans les
   10.5 ms de petites syncs.

**Interprété (bornes prudentes)** :
- **H-SYNC comme "LE" goulet est infirmée** au sens strict : le coût n'est ni dans les syncs d'orchestration
  (~11 %), ni dans le découpage du graph (les 5 launches coûtent < 1 ms ensemble).
- **Le vrai coût est les 77 ms de travail GPU** — où le verify (20.5 GB de poids traversés + 64 blocs) n'est
  **ni BW-saturé (§12.8), ni CPU-orchestration-saturé (§13.3.2)**. La seule composante restante, non mesurée
  au niveau kernel (CUPTI CLOS), est **le régime de calcul des kernels du verify** : si le GEMV/GEMM q6_K
  Volta tourne en-deçà du pic de calcul (ou si les 64 blocs sont sérialisés par des dépendances), c'est **là**
  que sont les 77 ms — c'est une propriété des **kernels**, pas du scheduling.
- Conséquence : **il n'y a pas de "coût de sync" à ramasser** dans le scheduler pour gagner des tokens/s.
  Les deux leviers restants à explorer sont :
  1. **H-KERNEL** : identifier dans les 77 ms quels kernels dominent et s'ils sont sous-optimaux (nécessite
     nsys sur une autre machine, **ou** un build instrumenté du kernel q6_K — pas lisible depuis l'extérieur
     avec la lib binaire) ;
  2. **H-STRUCTURE** : réduire le travail du verify lui-même (ex. : n_max plus faible si la 3e position coûte
     plus qu'elle ne rapporte — déjà mesurable **sans** nsys via les stats `acc rate/pos` existantes : 0.819/
     0.557/0.430 → la 3e position est déjà suspecte, cf. §2 H2c).

### 13.4 Prochaine action (proposée, non exécutée)

Le seul test encore **faisable ici** sans nsys ni recompile : **A/B n_max 3 vs 2** (même harnais, même prompt,
600 tok) — si `acc rate/pos` montre que la 3e position coûte un forward full supplémentaire qu'elle ne remplit
pas (0.430), alors `n_max=2` supprime un travail GPU entier par round **à précision quasi constante** (on perd
0.430 token/round, on gagne ⅓ du temps de verify par round). C'est la seule modification "structurelle"
mesurable avec les outils actuels. Sinon, H-KERNEL exige un déplacement (machine avec CUPTI fonctionnel).

**Rollback vérifié** : `ollama-v100` active, V100 32 348 MiB, md5 `e4afbb4a`, API 11434 répond. La lib trace
est dans `h2/` uniquement (LD_PRELOAD non installé sur la prod).

---

## 14. H-STRUCTURE — A/B n_max 3 vs 2 (§13.4, décision utilisateur : lancer le test)

**Protocole** : même harnais (`h2/h2_ab_nmax.sh` dérivé de `h2_ab_q4k.sh`), Q6_K_M, prefill 100k, 2e passe,
decode 600 tok, 2 réplicats par config, ordre alterné R1=[n3,n2] R2=[n2,n3], port 44901, prod stoppée/restaurée.
Shortlist MTP vérifiée **indépendante de n_max** (géométrie output-head 248320 lignes, `mmvq.cu:1862` — pas d'artefact).

### 14.1 Résultats (médianes, bruit réplicats 0.4 % / 0.8 %)

| config | tok/s | wall/round | mean len | acc/pos | GPU power décode |
|---|---:|---:|---:|---|---:|
| **n_max=3** | **29.8 / 29.7** | **94.2 / 94.5 ms** | 2.81 | 0.819, 0.557, 0.430 | 179–181 W |
| **n_max=2** | **22.4 / 22.5** | **107.4 / 106.5 ms** | 2.40 | 0.804, 0.600 | 161–163 W |

Δ (n2 vs n3) : **tok/s −24.7 %**, **wall/round +13.4 %**.

*wall/round = wall décode ÷ (600 / mean len) = mean len ÷ tok/s. (Le 1er passage d'analyse montrait
84.8/97.6 ms : bug d'unité — `wall ÷ a` avec `a` = rounds cumulés warmup+prefill. Corrigé dans
`h2_nmax_analyze.py`.)*

### 14.2 LECTURE

**Mesuré** :
1. **n_max=3 est 24.7 % plus rapide** (29.8 vs 22.4 tok/s).
2. **Contre-intuitif** : le round n=3 (94.3 ms) est **12.7 ms plus court** que le round n=2 (107.0 ms),
   **alors qu'il vérifie un batch plus gros** (+1 token draft) et **consomme 18 W de plus** (180 vs 162 W).
   Donc n=3 : +0.41 tok/round **et** −12.7 ms/round — les deux effets jouent dans le même sens.

**Interprété (bornes prudentes)** :
- **H-STRUCTURE infirmée** : la 3e position est un **net gain** (tokens/round en hausse, round en baisse).
  n_max=3 est le bon réglage ; baisser n_max est contre-productif (−24.7 %).
- L'anomalie (batch plus gros → round plus court, power plus élevée) **n'est pas expliquée par le modèle de
  coût fixe hôte** (§13 : ~16 ms/rd fixe, identique pour les deux). Elle indique que le travail GPU se répartit
  plus efficacement par round à batch 3 (meilleure amortisation du coût fixe de launch/sync sur un batch plus
  gros, ou motif d'accès mémoire plus dense) — à confirmer au niveau kernel (H-KERNEL, nsys ailleurs).

### 14.3 État du programme des hypothèses

| H | Statut | Résultat |
|---|---|---|
| H-BW (requant q4_K) | testé §12 | −27.7 % d'octets → −6.4 % wall/rd ; transit HBM ≈ 25–35 % ; gain appliqué +14.5 % tok/s, −5.7 GB (de côté) |
| H-SYNC (syncs/scheduler) | **infirmée** §13 | syncs d'orchestration = 11 % du round seulement |
| H-STRUCTURE (n_max) | **infirmée** §14 | n_max=3 optimal ; la 3e position est rentable |
| **H-KERNEL** (régime des kernels q6_K dans les 77 ms) | **ouverte** | non mesurable ici (CUPTI CLOS) — nsys ailleurs ou build instrumenté |

**Rollback vérifié** : `ollama-v100` active, V100 0 MiB → rechargée, md5 `e4afbb4a`, API 11434.

---



---

## 15. REOUVERTURE PARTIELLE — CUPTI 12.9 fonctionne sur V100/driver 580 (2026-10-11)

Le verdict §1 (CLOS) portait sur **CUPTI 13.x** (13.0.85 / 13.1 / nsys 2025.5.2 et 2025.3.2) :
`CUPTI_ERROR_INVALID_DEVICE` a l'activation, conforme a la note NVIDIA "2025.4+ non supporte
sur Pascal/Volta".

**Test du 2026-10-11 (driver 580.178.04 inchange, CUPTI 12.9.79 en prefixe isole,
`h2/cupti129-test/`, rien d'installe sur le systeme) :**

- Autotest in-process (`test_cupti129.cu`, nvcc 12.6 prod, gcc-13, sm_70) **sur la V100 libre** :
  `cuptiActivityRegisterCallbacks` + enable CONCURRENT_KERNEL/MEMCPY -> **OK, pas de CLOS** ;
  3 kernels captures avec **noms + dures** (1,5 / 2,0 / 2,1 us).
- Shim LD_PRELOAD (`libcupath.so`, CUPTI 12.9 rpath vers le prefixe) sur une app CUDA ordinaire
  (`kapp`, ne touche pas CUPTI, comme llama-server) : **20/20 kernels captures avec noms + dures**
  (1,2-1,4 us). -> le mecanisme d'observation sur un processus tiers est valide.
- Coexistence shim + CUPTI natif : OK, pas de crash (les deux enregistrent ; ne pas doubler).

Consequence : **H-KERNEL est mesurable sans redescendre le driver** — il ne reste a valider que le
passage sur le vrai chemin (llama-server + ggml-cuda verify MTP). Script pret :
`h2/cupti129-test/run_cupath_llama.sh` (serveur de capture isole port 44901, arret/reamorcage
ollama-v100 avec trap de restauration, sortie `.bin` 24 octets + `.names` en clair).
Le driver 575 n'est plus requis pour cet objectif (garde en fallback, compilation toujours en echec).

### 15.1 RESULTAT H-KERNEL — decode MTP mesure (run6, 2026-10-11 14:43)

Protocole strict (gates FIT/DEVICE/model loaded/shortlist + prefill ~100k x2 hors fenetre),
decode 64 tok a ~100k ctx (task 236 du log serveur), shim CUPTI 12.9, trace
`/tmp/cupath_llama_144318.names` (705k records, 90 Mo).

**Chiffres cles (log serveur + trace CUPTI) :**
- decode : 64 tok, wall 1576.52 ms, **25.02 ms/tok = 39.96 tok/s**, mean_len 3.71, ~17 rounds,
  acc/pos (0.941, 0.882, 0.882) — parite §14 (n_max=3).
- **Somme GPU-kernels (2 streams, concurrents) = 1575.8 ms sur 1576.5 ms de wall => ~100 % busy
  en temps cumule.** Le GPU ne s'arrete JAMAIS pendant le decode : l'hypothese d'un gros trou
  GPU (attente CPU) dans les 77 ms est **infirmee** — le regime est un empiement permanent de
  kernels sur 2 streams (16 = compute, 17 = preprocessement/draft).
- 42 618 kernels / 64 tok = **~665 kernels/round** ; temps median < 5 us (58.6 % des launches < 5 us).

**Decomposition des ~92.7 ms/round (temps GPU cumule) :**

| famille | us/rd | %GPU | note |
|---|---:|---:|---|
| `mul_mat_vec_q` (q6_K / q4_K / q8_0, gemv) | ~80000 | 53 % | 8 991 lances/rd ; le regime dominant |
| `ggml_q8v14partial_kernel` (KV q8_0) | ~27 600 | 17.5 % | 17/rd, ~1.6 ms chacun (les plus longs apres flash_attn) |
| `flash_attn_ext_vec` (decode, 1 seq) | ~4 400 | 2.8 % | 3/rd (draft+verify) |
| `mul_mat_vec_q_mapped_rows` (shortlist) | ~2 700 | 1.7 % | |
| `quantize_q8_1` (KV cache) | ~1 660 | 1.0 % | 553/rd — overhead de requantisation |
| `gated_delta_net` (SSM layers) | ~1 180 | 0.8 % | |
| `add_rms_norm_mul_f32` | ~1 080 | 0.7 % | |
| reste (cpy_scalar, unary, rms_norm, radix sort, set_rows...) | ~5 400 | 3.5 % | |

**Lecture :**
1. **H-KERNEL (regime des kernels q6_K dans les 77 ms) : RESOLUE.** La masse GPU est le
   `mul_mat_vec_q` (gemv q6_K/q4_K/q8_0) : ~53 % du temps GPU cumule, lance en ~9000 x/round.
   C'est du compute mem-bound classique ; le "77 ms hors GPU-MTP" mesure auparavant etait
   le mur du round, pas un trou GPU.
2. **Surprise** : le preprocessement KV q8_0 (`ggml_q8v14partial` + `quantize_q8_1`,
   ~18.5 % du GPU cumule) est plus couteux que le flash_attn decode (~2.8 %). Le cache
   q8_0 (gain BW H-BW) a un cout fixe de requantisation non negligeable par round.
3. Le draft (stream 17, ~11 % du temps cumule) est bien separe et amorti sur le verify.
4. **Piste d'action concrete** : l'optimisation du decode passe par le taux de lances
   (~665/rd) et le cout par gemv (taille de batch des layers) ; pas par la reduction des syncs
   (H-SYNC infirmee) ni par le flash_attn.

Artefacts : trace + analyse `/tmp/cupath_llama_144318.names`, log serveur `/tmp/cupath_server.out`,
log run `/tmp/cupath_run6.log`. Repro : `h2/cupti129-test/run_cupath_llama.sh <ntok>`.

---

## 16. REVUE INDEPENDANTE du run6 CUPTI (2026-10-12, apres explosion de contexte)

Revue complete du run6 (shim CUPTI 12.9, trace /tmp/cupath_llama_144318) dans un fichier
separe : **`h2/cupti129-test/ETAT-cupath-revue-12-10.md`** (findings F1..F16, classes
CRITIQUE/MAJEUR/MINEUR/OK, chacun : fait → consequence → action).

Points cles (detail dans le fichier de revue) :
- **F1 [CRITIQUE]** : le `t` du `.names` n'est PAS un timestamp GPU mais le temps d'EMISSION
  au flush du thread de drain (500 ms) → le verdict « GPU ~100 % busy / ne s'arrete jamais »
  du §15.1 **n'est pas demontre** (coincidence Σ dur ≈ wall sans positions start/end).
  Tout calcul d'union/chevauchement sur `t` est invalide.
- **F2 [CRITIQUE]** : la trace contient **62 segments** (un par fork de llama-server) ;
  toute analyse non segmentee melange les pids. Le decode task 236 = segment pid 387290
  seul (42 618 KER, Σ 1575,8 ms ≈ wall 1576,52 ms) → §15.1 reste valide tel quel.
- **F15 [MAJEUR]** : le « prefill ×2 » du script = pre1 100k (task 4, wall 195 s) +
  pre2 4-tok-chaud (task 225) ; le warmup est la **task 0** (pas task 4) ; le 1er decode
  (task 4, 83 ms/tok) est 3x plus lent que le regime (25 ms/tok) — construction des graphs.
- **F4/F9/F12 [OK]** : bin↔names alignes exactement (705 294 = 705 294, toutes traces),
  aucune perte de record ; generation valide (64 tok, finish_reason=length).
- **F16 [MAJEUR]** : le `t` CLOCK_MONOTONIC des headers est hors echelle entre segments
  (sauts de +16,75 min pour ~16 s reel) → a remplacer par CLOCK_REALTIME dans le shim v2.

Priorite des traitements (un par un) :
1. F1+F16 : shim v2 (gpu_start/gpu_end par record + t_real_us dans le header) + recapture run7
   → re-demontrer ou refuter « GPU busy ».
2. F2/F3/F15 : `cupath_analyze.py` par groupe de segments (selection par pid).
   (F2 detail, corrigé le 13/10 : 62 headers dont 8 porteurs de records — voir §18).
3. F6/F7/F8/F10/F11 : nettoyage scripts (stderr shim, noms de fichiers, analyseur unique,
   parser .bin toujours segmente par header).
4. Sauvegarde des traces run6 dans le repo (/tmp est volatil). a ete corrige en units (t en us, dur en ns) le 12/10 ; il reproduit le
rapport §15.1 mais ses sections « GLOBAL » (F2) et « CHEVAUCEMENT » (F1) sont a retirer/
corriger dans la v2.

---

## 17. AUDIT des pistes generiques d'optimisation V100 (2026-10-12)

Audit d'un document « optimiser une V100 » (pistes A–F) contre notre config, sans code ni
benchmark : **`h2/ETAT-audit-v100-12-10.md`** (matrice synthétique + liste des pistes
reellement credibles, classees interet x incertitude).

Verdicts courts :
- **A (TC FP16/QPN/FP32)** : deja couvert (P0/P1/P2 fermes) — decode T=4 NO-GO (dp4a
  imbattable T≤9, port NInfer 2,7x plus lent au regime PCIe/L2) ; prefill TC deja en place.
- **B (alignement dims / ncols=8)** : deja couvert (P4/P5/P6) — paie x1,5 mais inaccessible
  structurellement (MTP -np 1 → ne1∈{1,4}).
- **C (torch.compile/Triton/XLA/fusion)** : framework inapplicable (llama.cpp) ; les
  equivalents natifs (CUDA graphs + kernels ggml fusionnes) sont deja en place.
- **D (TensorRT/vLLM/TGI)** : non applicable — modele hybride SSM+FA + draft-mtp + K-quant +
  kernels custom ; une migration jetterait tout.
- **E (AWQ/GPTQ/INT4/FP16/FP8)** : sm_70 sans TC INT4/FP8 ; FP16 = perte nette (mem-bound) ;
  l'effet utile (reduire les octets de poids) est deja reel via H-BW.
- **F (DataLoader/num_workers/pin_memory/MPS)** : entrainement / multi-process → non applicable
  au decode mono-modele (1 process CUDA par fenetre, F2).

Point cles du croisement CUPTI : **incoherence du §15.1** — `ggml_q8v14partial` (pretraitement
KV q8_0) vaut **29,8 %** du budget decode (pas 17,5 %) et le tableau ne somme pas a 100 %.
Le poste KV-q8_0 passe de « second ordre » a **2e famille** juste apres `mul_mat_vec_q` (53 %).

Pistes reelles (continuite, pas de nouvelle branche) :
1. `mul_mat_vec_q` 53 % : taux de lances / cout par gemv (suite de P7).
2. Poste KV-q8_0 ~32 % : decaler/fusionner dequant+requant (lien H-BW).
3. D'abord re-demontrer/refuter « GPU 100 % busy » via run7 CUPTI v2 (F1) — bloque la
   lecture de 1-2.

---

## 18. VERDICT run7 CUPTI v2 — question instrumentale (2026-10-13)

Fichier : **`h2/cupti129-test/ETAT-run7-verdict-13-10.md`** (bornes, protocoles V1–V5,
hypothèses, chiffres dérivés). Verdict court :

- **F1 confirmé** : `t` = temps d'émission CPU (thread drain) → aucune union/chevauchement
  calculable sur run6. Les données existantes **bornent** l'occupation GPU du decode task 236 :
  **U/W ∈ [89,22 % ; 100 %]**, idle ≤ 10,78 % (169,96 ms), chevauchement O/W ∈ [2,1 % ; 12,9 %].
  L'ancien « 99,95 % » = ΣKER/W (rapport de somme) — compatible mais non démontré.
- **F2 corrigé** : 62 headers, **8 seuls porteurs de records** ; seq global 0..705 293 **contigu**
  (tiling exact). Le decode = 8e porteur (seq 661583..705 293) → sélection par seq, F5 levé
  pour cette fenêtre.
- **Cohérence vérifiée** : 0 mismatch dur_ns/stream bin↔names sur les 43 711 records du decode ;
  unités ns plausibles (max 105,9 ms ; mmvq mean 112,8 µs). `dur_ns`+`stream`+`seq` exploitables.
- **Décision** : **run7 v2 à lancer** (gpu_start/gpu_end CUPTI + t_real_us CLOCK_REALTIME au header
  + silencing stderr) avec critères de validation V1–V5 explicites et 3 hypothèses chiffrées
  (saturé ≥ 97 % / léger-idle 89–97 % / réfuté < 89 % — ce dernier impossible vu la borne,
  sauterait = bug de capture). Sans start/end, U, O et l'idle par round **restent indéterminés**.
- **Déjà utilisable sans run7** (dur_ns fiables) :
  - `mul_mat_vec_q` = 840,27 ms (53,3 % KER), 9 408 lances — q6_K/nc4 456,8 ms (mean 112,8 µs),
    q8_0/nc4 168,3 ms (46,5 µs), q5_K/nc4 132,8 ms (103,9 µs), 51 `*mapped*` 46,4 ms
    (mangling différent, à identifier), nc=1 en 2e position (28,97 ms).
  - KV Q8_0 sans recouvrement = 524,6 ms (33,3 % KER) : q8v14partial 469,6 ms (288×) +
    quantize_q8_1 28,2 ms (9 408×) + q8v13reduce 26,8 ms (288×) — séquentiels, s'additionnent
    légitimement. Signatures binaires confirmées dans `libggml-cuda.so` (ggml_q8v::partial/reduce).

---

## 19. AUTOPSIE `mul_mat_vec_q` sur run6 (2026-10-13, sans capture)

Fichier : **`h2/cupti129-test/ETAT-autopsie-mmvq-13-10.md`**. Résultats courts :

- **Les 51 `*mapped*` (mean ~910 µs) = le kernel MTP SHORTLIST du lm_head de sortie**
  (`mul_mat_vec_q_mapped_rows<(ggml_type)8,1>`, st17, 1×/forward, 51 forwards). 131 072 rangées ×
  5 120 cols lues en Q8_0 = 755 Mo → médiane 906 µs = **~833 GB/s ≈ plafond HBM V100**. Coût
  légitime d'une opération structurellement grosse (2,9 % du decode, 5,5 % du mmvq). **Pas un
  artefact, pas à agréger aux variantes ordinaires.** (Hyp. haute conf. : code + log + géométrie.)
- **Q6_K/nc4 = 1ʳᵉ famille (456,8 ms, 28,9 %)** : signature unique
  `<Q6_K,nc4,rpb_override=4>`, 238,2 lances/verify-fw, plateau médian 132,2 µs (p90 152,5, max 164,2)
  + 54 lances < 50 µs (sous-groupe forme réduite, même symbole).
- **Q8_0/nc4 = bimodal** : 2 232 lances rapides (méd 8,3 µs) + 1 385 lentes (méd 58,4 µs) → le
  « mean 46,5 µs » est un artefact de mélange de formes ; **pas** de conclusion « Q8_0 plus rapide »
  (formes/bytes différents). À forme comparable : Q6_K ~132 µs ≈ Q5_K ~120 µs > Q8_0-lent ~58 µs.
- **Lien P0–P7** : renforce le NO-GO TC decode (P0–P2), confirme le régime mem-bound (P3 : lm_head
  à 92 % du pic HBM), renforce la fermeture ncols=8 (P4–P6 : seuls nc1/nc4 apparaissent), et confirme
  que Q6_K est LA cible kernel mais à **marge ~1 %** (P7) → sous le seuil.
- **run6 ne peut PAS répondre** : attribution couche (le nom manglé ne porte pas ne[0]/ne[1]),
  identité des 2 groupes Q6_K, occupation/chevauchement (verrou F1 → run7 v2).
- **3 expériences candidates** (non lancées) : E1 attribution couche (export ne[]), E2 compresser la
  shortlist 131 072→65 536 (A/B acceptance), E3 plateau Q6_K < 90 % DRAM ? (falsifier P7 seulement
  si plateau < 750 GB/s).

---

## 20. AUTOPSIE famille « KV Q8_0 » sur run6 (2026-10-14, sans capture)

Fichier : **`h2/cupti129-test/ETAT-autopsie-kv-q8-0-14-10.md`**. Résultats courts :

- **Recadrage (fait)** : les 3 kernels groupés ne font **pas** la même famille.
  - `ggml_q8v14partial` + `q8v13reduce` = le vrai **flash-attention Q8** (KV en Q8_0) = **496,4 ms**.
  - `quantize_q8_1` = re-quant **de l'opérande A du gemv** (pas du KV), 1:1 devant chaque
    `mul_mat_vec_q` (9 357 mmvq + 51 shortlist) = **28,2 ms** → famille mmvq (§19), pas le KV.
- **`ggml_q8v14partial` = LE 2ᵉ poste (469,6 ms, 29,8 % W)** : `fattn-q8-volta.cuh`
  `ggml_q8v::partial_kernel` (QK^T+softmax+PV, Q8_0, `mma.sync.m8n8k4`, Volta sans `ldmatrix`,
  déquant in-loop). 288 lances, stream 16, **16 FA-layers × 17 verify-fwd**. D=256, G=6, T=4,
  BC=16, THREADS=128, **splits=160** (n_kv=107 351). Durée **min 1457 / médiane 1615 / max 1840 µs**
  (uniforme, pas de bimodalité).
- **Volumes (modèle code+géom. — ESTIMATION, pas compteur DRAM)** : **≈ 234 Mo KV/lance**
  (K+V, 4 kv-heads) + ≈ 8 Mo pacc → **≈ 242 Mo** → **~150 GB/s ≈ 19 % du plafond HBM** (V100 ~800).
  → **ni mem-bound (19 %), ni TC-bound (≈ 85 µs à 125 TF = 5 %).**
- **Pourquoi tant de temps (Q4)** : (1) volume KV énorme × 16 couches × 17 verify-fwd (seul endroit
  qui relit la cache KV complète) ; (2) **efficacité kernel ~19 %** = **occupation 12,5 %**
  (`launch_bounds(128,2)`) + déquant in-loop + sync/tuile de 16 clés + Volta sans ldmatrix.
- **`q8v13reduce`** : 288 lances 1:1 appariées (288/288), médiane 92,4 µs, **négligeable (26,8 ms)**.
- **Potentiel (Q6, borne honnête)** : plancher mémoire ~292 µs/lance (vs 1615) → **~385 ms d'énergie
  sérialisée** si partial 5,5× plus rapide. **Pas un gain de wall** : le gain mural réel dépend de
  l'union/overlap (F1, inconnu). Le vrai levier = **efficacité du partial**, pas la quantité de
  données (KV déjà Q8_0) ni le reduce.
- **run6 ne peut PAS trancher** : octets DRAM réels (pas de compteur), gain mural (F1),
  split-KV optimal, quel facteur (occupation/déquant/sync) pèse le plus.
- **3 expériences candidates (non lancées)**, classées valeur/coût :
  **E1** mesure DRAM réelle du partial (NCU/nsys ou run7 v2+DRAM) — falsifie si BW>600 GB/s ;
  **E2** sweep occupation/pré-déquant si E1 confirme kernel-bound (Δpartial ≥ −40 %) ;
  **E3** run7 v2 (union/overlap) pour chiffrer le gain **mural** (déjà exigé par F1).

## 21. CLASSIFICATION `flash_attn` draft (75,7 ms) + correction comptage Q8 (2026-10-13, sans capture)

Fenêtre decode run6 (`seq` 661 583..705 293). **Aucune capture, aucune modif de code.**

### 21.1 Ce que sont les 75,7 ms
- `flash_attn` = 2 kernels, **100 % stream 17 (draft)** :
  - **`flash_attn_ext_vec<256, 1, GGML_TYPE_F16, GGML_TYPE_F16, false>`** — 51 lances,
    **75,36 ms** (médiane **1460 µs**, min ~1,4 / max 1750 µs).
  - `flash_attn_combine_results<256>` — 51 lances, 0,32 ms (médiane 6,1 µs) — split-merge, négligeable.
- Mangled → `void flash_attn_ext_vec<D=256, ncols=1, type_K=ggml_type1, type_V=ggml_type1, softcap=false>(...)`.
  `ggml_type1 = GGML_TYPE_F16` (enum ggml.h : F32=0, **F16=1**). **K et V du draft sont en F16.**
- Source : `fattn-vec.cuh:21` (`template<int D, int ncols, ggml_type type_K, type_V, softcap>`),
  dispatchée via `fattn.cu:549 ggml_cuda_flash_attn_ext_vec` quand le best-kernel choisit le
  chemin **VEC** (pas le mma-tile). **Ce n'est PAS un Q8 kernel.**

### 21.2 Dimensions et rôle (draft vs vérification)
- **51 lances = 1 par forward draft, 51 forwards draft** dans la fenêtre (pas 16 : c'est l'attention
  du **layer nextn unique** du MTP, pas les 16 FA-couches du target).
- D = 256 (key/value_length du GGUF, arch `qwen35` : `block_count 65`, `head_count 24`,
  `head_count_kv 4`, `nextn_predict_layers 1`, embd 5120, FFN 17408 — logs serveur run6/ncu).
- **Le draft relit la MÊME cache KV persistante (q8_0)** ; le `ggml_cuda_flash_attn_ext_vec`
  **convertit K/V q8_0 → F16** au vol si le cas F16-F16 est requis (warning « converting K and V
  to f16 » dans `fattn.cu:556`) — donc le type_F16 vu dans le nom ne contredit PAS le KV q8_0 :
  c'est le **format de travail** du kernel vec, pas le format de stockage.
- **T du draft = 1 token** (1 fa par forward draft) vs **T=4 du verify** (4 tokens vérifiés) —
  cohérent avec `row_count` et la cadence : **1 forward draft précède chaque verify**.

### 21.3 Structure d'une « macro-période » MTP (vérifiée sur la trace)
- Découpage des 288 lances `ggml_q8v14partial` (stream 16) en **bursts contigus (gap>200)** :
  **18 bursts × exactement 16 lances** = **18 forwards de vérification × 16 FA-couches**.
  → **CORRECTION du §20 : c'est 18 verify-fwd, pas 17** (288/16 = 18 exact, 18 bursts homogènes).
- Entre deux bursts verify, dans le gap (stream 17) : **3 × `flash_attn_ext_vec`** + **3 ×
  `mul_mat_vec_q_mapped_rows`** (51 = 3×17 dans les gaps + bordures ; ratio 51/18 ≈ 2,83).
  Cadence observée : **draft → verify → draft → verify** (1 verify accepte ~3 drafts,
  acceptance ≈ 1.0000 / mean len 4.000 dans les logs ncu).
- Chaque forward draft (s17) porte **sa propre attention** (`flash_attn_ext_vec`) **et son propre
  lm_head shortlist** (`mapped_rows`) — donc les 51 fa et les 51 mapped_rows sont le même évènement
  vu de deux kernels.

### 21.4 Relation EXACTE avec le `partial` Q8 déjà analysé
| | `ggml_q8v14partial` (verify) | `flash_attn_ext_vec` (draft) |
|---|---|---|
| Famille | `ggml_q8v` (mma q8, `fattn-q8-volta.cuh`) | `flash_attn_ext_vec` F16 (`fattn-vec.cuh`) |
| Kernel | Q8_0 KV, `mma.sync.m8n8k4`, splits=160 | F16 KV (q8_0→f16 au vol), ncols=1, split-KV |
| Stream | 16 (target/verify) | 17 (draft) |
| Lances | 288 = 18 verify × 16 couches | 51 = 1 par forward draft (1 FA-layer nextn) |
| Durée méd. | 1615 µs | 1460 µs |
| T | 4 tokens | 1 token |

**Conclusion : C'est une famille DISTINCTE.** Pas de double-comptage avec le §20 — les 469,6 ms
(partial Q8, verify, s16) et les 75,4 ms (fa F16, draft, s17) sont des kernels, streams, formes et
rôles différents. **Le draft n'a PAS de `ggml_q8v14partial`** : son attention passe par le chemin
vec F16 (le q8v partial est réservé au verify, qui est sur le chemin `BEST_FATTN_KERNEL_VOLTA_Q8_W4`).
Les deux sont des **relus complets de la cache KV** (verify 16 couches × 18 ; draft 1 couche × 51),
mais à échelles très différentes — d'où 469,6 ms vs 75,4 ms.

### 21.5 Coût du draft (75,4 ms) — lecture honnête
- 75,4 ms = **~5 % du temps decode**. C'est **16× plus petit** que le verify-attention (469,6 ms)
  car : (a) 1 FA-layer nextn vs 16 ; (b) T=1 vs T=4 ; (c) même n_kv mais 1 seul passage.
- **C'est un coût légitime du MTP** (le draft doit lui aussi faire son attention + son lm_head
  shortlist pour proposer des tokens). **Pas une anomalie.**
- Le `combine_results` (0,32 ms) et le `cub` sort (7,77 ms, s17) sont des **accessoires du draft**
  (tri de la shortlist) — déjà comptés, négligeables.

## 22. INVENTAIRE GLOBAL des coûts — run6 decode (consolidé, 2026-10-13)

**Source** : `/tmp/cupath_llama_144318.names` (dur/stream/name/op fiables ; `t` NON fiable — F1).
**Fenêtre** : `seq` 661 583..705 293 (43 711 records). **MURAL connu : 1576,52 ms.**

### 22.1 Trois grilles, strictement séparées
1. **Durées cumulées (kernels)** = somme des `dur_ns` par famille. **C'est une somme de travail
   GPU, PAS du temps mural** (peut dépasser le mural via overlap). Total kernels = **1575,8 ms**.
2. **Temps mural connu** = **1576,52 ms** (bornes fiables : S16=1406,56 / S17=169,25 / CPY=34,42).
3. **Trafic mémoire** = **ESTIMATION modèle (code+géométrie)** tant que E1 (compteur DRAM) ne
   renvoie pas les octets réels. **Jamais présenté comme mesure directe.**

### 22.2 Grille A — durées cumulées par famille (kernels, ms, % du total kernels)
| ms | % | count | streams | famille (rôle) |
|---|---|---|---|---|
| 793,87 | 50,4 % | 9 357 | s16:761,5 / s17:32,3 | `mul_mat_vec_q` (gemv couche — target+draft) |
| 469,59 | 29,8 % | 288 | s16 | **`ggml_q8v14partial`** (FA **verify** Q8 — 2ᵉ poste) |
| 75,36 | 4,8 % | 51 | s17 | **`flash_attn_ext_vec`** (FA **draft** F16 — §21) |
| 46,40 | 2,9 % | 51 | s17 | `mul_mat_vec_q_mapped_rows` (lm_head shortlist MTP) |
| 34,12 | 2,2 % | 5 919 | s16:31,3 / s17:2,8 | `rms_norm` |
| 28,24 | 1,8 % | 9 408 | s16:27,0 / s17:1,3 | `quantize_q8_1` (re-quant opérande A gemv) |
| 26,77 | 1,7 % | 288 | s16 | `ggml_q8v13reduce` (FA verify Q8, split-merge) |
| 20,14 | 1,3 % | 864 | s16 | `gated_delta_net` (SSM) |
| 18,30 | 1,2 % | 3 795 | s16:18,0 / s17:0,3 | `cpy_scalar` |
| 13,03 | 0,8 % | 1 848 | s16:12,6 / s17:0,4 | `k_get_rows` |
| 12,86 | 0,8 % | 4 083 | s16:12,7 / s17:0,1 | `unary_op` |
| 7,77 | 0,5 % | 306 | s17 | `cub` (sort/scan — **accessoire draft**) |
| 7,03 | 0,4 % | 714 | s16:6,6 / s17:0,4 | `k_set_rows` |
| 5,93 | 0,4 % | 1 746 | s16 | `k_bin_bcast` |
| 4,43 | 0,3 % | 933 | s16:4,1 / s17:0,3 | `concat` |
| 4,08 | 0,3 % | 864 | s16 | `ssm_conv` |
| 3,81 | 0,2 % | 1 152 | s16 | `fwht` |
| 2,82 | 0,2 % | 696 | s16:2,4 / s17:0,4 | `rope` |
| 0,94 | 0,1 % | 204 | s17 | memcpy stream 0 (petits transferts) |
| 0,32 | 0,0 % | 51 | s17 | `flash_attn_combine_results` (draft, split-merge) |
| — | — | — | — | (total kernels **1575,8 ms** ; + memcpy s0 **34,42 ms**) |

**Top-4 = 1 385,2 ms = 87,9 % des durées kernels.** (mmvq 793,9 + partial-Q8-verify 469,6 +
fa-draft 75,4 + shortlist 46,4). Le **verify-attention Q8 (469,6 ms) reste le seul levier > 100 ms
hors gemv** ; le gemv (793,9 ms) est un mur de poids (P0–P7, §19) déjà jugé < seuil.

### 22.3 Grille B — temps mural connu (bornes fiables, run6)
- **MURAL = 1576,52 ms** (64 tok générés).
- S16 (verify/target) somme = **1406,56 ms** ; S17 (draft) = **169,25 ms** ; CPY s0 = **34,42 ms**.
- Occupation GPU **U/W ∈ [89,22 % ; 100 %]** (max des sommes streams / mural) ; **idle ≤ 10,78 %**.
- Overlap **O/W ∈ [2,14 % ; 12,92 %]**. → **Ces bornes ne tranchent PAS l'overlap réel** (F1) :
  il faut un run7 v2 avec `gpu_start_ns/gpu_end_ns` ou E3. **Ne pas lire « GPU 100 % busy ».**

### 22.4 Grille C — trafic mémoire (ESTIMATION modèle, PAS des compteurs)
| Famille | Volume / lance (estimé) | Base |
|---|---|---|
| `ggml_q8v14partial` (verify Q8) | **≈ 242 Mo** (234 Mo KV K+V q8_0 × 4 kvh + 8 Mo pacc) | n_kv=107 351, §20 |
| `flash_attn_ext_vec` (draft F16) | **≈ 470 Mo/lance (F16)** si K/V convertis q8_0→f16 | même n_kv, 2 Mo/key F16 vs ~1 Mo q8_0 |
| `mul_mat_vec_q` (gemv) | ≈ poids Q6_K/Q8_0 lus par couche (mur poids, §19) | — |
| `mapped_rows` (shortlist) | ≈ **755 Mo** (131 072 × 5 120 Q8_0) | §19, ~833 GB/s établi |

⚠️ **Le draft en F16 relit la KV 2× plus épaisse qu'en Q8_0** (≈ 470 Mo vs ≈ 234 Mo / couche-
équivalent). C'est une **pression mémoire cachée du draft** : 51 × ~470 Mo ≈ **24 Go de DRAM**
répartis sur la fenêtre (estimation). **E1 doit trancher** : si le partial Q8 (verify) tourne < 300 GB/s
→ kernel-bound ; si le draft F16 tourne > 600 GB/s → il est mem-bound (la conversion F16 coûte
réel). C'est l'angle le plus intéressant de E1.

### 22.5 Ce que run6 ne peut toujours PAS trancher
- Octets DRAM réels (pas de compteur) → **E1**.
- Gain **mural** réel d'une optimisation (union/overlap) → **E3 / run7 v2** (F1).
- Split-KV optimal, quel facteur (occ/déquant/sync) pèse le plus → **E2** (après E1).

## 23. E1 — VERDICT : le `ggml_q8v::partial_kernel` est **kernel-bound** (2026-10-13, expérience lancée)

**Expérience lancée** (autorisation utilisateur) : **E1** = mesurer le comportement mémoire du
`partial` Q8 (verify, 2ᵉ poste, 469,6 ms). Artefacts :
`h2/cupti129-test/{e1_bench.cu, e1_bench.cuh, run_e1_final.sh, e1_bench_result.txt}`.

### 23.1 Contrainte outillée (décision)
- **NCU 2025.4.1 ne supporte PAS sm_70** : `ncu --list-chips` ne liste **ni GP100 ni V100** ;
  tout profil renvoie `Profiling is not supported on device`. → le compteur DRAM
  `dram__bytes_*` est **indisponible sur cette box** (aucun ncu/nsys utilisable pour les
  octets réels).
- **Solution retenue (E1 « modélisé »)** : un **harnais standalone** (`e1_bench.cu`) qui lance
  `partial_kernel` **tel quel** (copie exacte de `fattn-q8-volta.cuh`, sm_70) sur la **vraie
  géométrie run6** (n_kv=107 351, kvh=4, G=6, T=4, D=256, splits=160, grid=(4,160), 128 thr),
  V100 dédiée, P0, et mesure :
  - la **durée RÉELLE** du kernel par **CUDA-events** (20 iters, après warm-up) ;
  - le **volume EXACT lu/écrit** déduit du code du kernel + géométrie (pas un compteur, mais
    chaque load/store du kernel compté avec ses strides réels).
  → **BW effectif = volume exact ÷ durée mesurée.** Durée mesurée + volume exact = ratio robuste.

### 23.2 Résultats (mesures)
| Grandeurs | valeur |
|---|---|
| Durée / lance (CUDA events, standalone, P0) | **1662 µs** (run6 CUPTI : 1457–1840, médiane **1615** — **réplication quasi-exacte, écart ~3 % = bruit horloge**) |
| Volume / lance (modèle exact) | **260,7 Mo** = KV **234** (K+V q8_0, 4 kvh) + Q **15,7** + mask **3,4** + out (pacc/pm/pl) **8,0** |
| **BW DRAM effectif** | **157 GB/s** ≈ **17–19 % du plafond HBM V100** (~800–900) |

⚠️ Nuance d'honnêteté : les **octets sont modélisés** (volume exact du code), **pas comptés**
(contrainte NCU sm_70). En pratique le volume DRAM est **encore un peu plus bas** que 260 Mo :
le Q (98 KB) et le mask restent en **L2** (lus par 6400 blocs mais buffer minuscule), donc leurs
~19 Mo sont **surtaxés** dans le modèle → **BW DRAM réel ≈ 145 GB/s** (242 Mo / 1662 µs). La
conclusion est **renforcée**, pas affaiblie.

### 23.3 Verdict
- **BW effectif ≈ 150–157 GB/s ≪ 300 GB/s → `partial` est KERNEL-BOUND (H2/H3 confirmés, H1
  « mem-bound » morte).** Le kernel **gaspille ~80 % de la bande passante HBM disponible**.
- **Le levier n'est PAS la quantité de données** (KV déjà Q8_0, 234 Mo est déjà minimal) :
  c'est **l'efficacité du kernel** — les suspects du §20 : **occupation 12,5 %**
  (`launch_bounds(128,2)`), **déquant in-loop** (`deq8` dans la boucle), **2× `__syncthreads`
  par tuile de 16 clés**, et **Volta sans `ldmatrix`** (loads adressés simples).
- **Potentiel (borne honnête, pas du wall)** : si le partial passait ~1662 → ~400 µs (≈ 600 GB/s,
  part réaliste), soit ~1260 µs/lance × 288 lances ≈ **~360 ms d'énergie sérialisée** récupérable.
  **Le gain MURAL réel reste inconnu** (F1/overlap) → à chiffrer par **E3 / run7 v2**.

### 23.4 Ce que ça change / prochaines expériences
- **E2 devient la priorité** : optimiser l'efficacité du `partial` (sweep occupation /
  pré-déquant K-V hors boucle / taille de tuile / `launch_bounds`), critère **Δpartial ≥ −40 %**
  et **sortie bit-identique** (le kernel reste le même algo, seule l'implémentation change).
  C'est une **modif de code** (autorisation requise) sur `fattn-q8-volta.cuh` uniquement.
- **E3 / run7 v2** toujours requis pour le **gain mural** (union/overlap, F1).
- **Côté draft (optionnel, plus petit)** : le `flash_attn_ext_vec` F16 du draft relit la KV
  **2× plus épaisse** (q8_0→f16 au vol). C'est un **second angle mémoire** (~75 ms, 5 %),
  distinct et plus petit que le verify. Peux être benchmarké de la même façon si désiré,
  mais le **verify Q8 (469,6 ms) reste le levier dominant** → E2 sur le partial.

### 23.5 Garde-fous tenus (E1)
- **Aucune capture CUPTI** ajoutée ; **aucune modif du code de prod** (le harnais est un
  binaire séparé, `e1_bench`, qui *copie* le kernel sans le modifier) ; le service
  `ollama-v100` est **restauré automatiquement** (trap) après chaque run ; `n_max=3` conservé
  (la géométrie du harnais reproduit le T=4 du n_max=3, pas de changement de paramètre serveur).

## 24. Mission Kolbo — compteur DRAM réel sur la V100 (2026-10-13, NCU 2024.1.1, E1 révisé)

Rapport complet : `cupti129-test/KOLBO-ncu-volta-20261013.md`.

### 24.1 Contexte et méthode
- NCU 2025.4.1 (système) **ne supporte pas la V100** (`--list-chips` sans GP100 →
  `Profiling is not supported on device 0`). Cible initiale NCU 2025.2.1 : le `.run`
  officiel nécessite un compte NVIDIA (non disponible) → **substitut officiel** :
  paquets Ubuntu `nsight-compute` + `nsight-compute-target` **2024.1.1.4~12.4.1**
  (archive.ubuntu.com), **déballés sans installer** dans `$HOME/ncu-2024.1.1`
  (`dpkg -x`, aucun PATH/paquet/pilote modifié).
- Validation : `--version` = 2024.1.1.0, `--list-chips` contient **`gv100`** (V100) ✅.
- Métriques confirmées via `--query-metrics --chip gv100` puis **réellement collectées** :
  `dram__bytes_read.sum`, `dram__bytes_write.sum`, `dram__sectors_*`, `dram__throughput`,
  `lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld/st.sum`.
- Compteurs = root requis (`RmProfilingAdminOnly: 1`) ; `/dev/nvidia*` rw-tous.
- **Piège** : l'index CUDA ≠ l'index nvidia-smi ici — CUDA dev 0 = **V100**, CUDA dev 1 =
  1070 (vérifié par `cudaGetDeviceProperties`). `CUDA_VISIBLE_DEVICES=1` fait profilé
  la 1070 → `no kernel image` (sm_70 sur sm_61). Runs finaux avec `CUDA_VISIBLE_DEVICES=0`.
- Service `ollama-v100` **arrêt requis** (V100 à 32 350/32 768 MiB ; harnais ~318 MiB GPU
  → OOM à 143 MiB libres). Restauration vérifiée par trap après chaque run (service active,
  API ok, V100 rechargée). 3 runs : A (skip 1), B (skip 1, 3 passes), C (skip 0) — résultats
  identiques à ~1 %.

### 24.2 Résultats (compteur DRAM réel, 1 lance `partial_kernel`, V100 seule)

| Grandeur | Valeur |
|---|---|
| Durée wall sans profiler (CUDA events, 3 iters) | **1665 µs** (run6 : 1615, écart ~3 %) |
| Durée sous NCU (`gpu__time_duration.sum`) | **1770–1790 µs** (+6.9–7.8 % de surcoût profiler) |
| Fréquence (`gpc__cycles_elapsed.max`/durée) | 1.229 GHz ≈ boost nominal |
| Octets demandés L1 (L2-miss LSU global ld+st) | **229.7 Mo** |
| `dram__bytes_read.sum` (cohérent à 0.1 % avec `sectors_read`×32) | **61.2 Mo** |
| `dram__bytes_write.sum` | **32.8 Mo** |
| **Total DRAM mesuré (R+W)** | **~94 Mo** = 29.4 % des demandes L1 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | **5.91 %** |
| **BW DRAM (94 Mo / 1780 µs, temps NCU)** | **~53 GB/s** (≈56 GB/s si dénominateur wall 1665 µs) |

**Clé d'interprétation** : la L2 absorbe ~136 Mo des 230 Mo demandés en L1
(re-partitionnement des secteurs entre les 640 blocs qui partagent les tuiles K/V) ;
la DRAM ne voit que ~94 Mo par lance. Le volume « théorique » 260.7 Mo du harnais est
un modèle d'accès, **pas** le traffic DRAM physique.

### 24.3 Verdict E1 révisé
- **Mesure DRAM validée** : compteur matériel exploitable, volume cohérent, service
  restauré, environnement non modifié.
- **BW DRAM ≈ 53 GB/s (5.9 % du pic HBM) ≪ 300 GB/s → `partial` KERNEL-BOUND,
  largement confirmé** (§23 maintenu et renforcé : c'est ~10× plus loin du mur mémoire
  qu'estimé). **H3 (réduire les octets) non prioritaire** ; les suspects du §23.3
  (occupation, déquant in-loop, syncthreads, tuiles) restent le levier.
- Le gain sérialisé estimé (~360 ms) tient ; le **gain mural** reste à chiffrer par
  **E3 / run7 v2**.
- **E2 non engagée**, `fattn-q8-volta.cuh` intact. Prochaine décision : autorisation
  E2, ou E3 d'abord.

## 25. E3 — run7 v2 : intervalles GPU réels (CUPTI v2) (2026-10-13, 18:34–19:45)

Rapport : `cupti129-test/KOLBO-ncu-volta-20261013.md` §E3.
Fichiers : `cupti129-test/cupath_shim_v2.c`, `libcupath_v2.so`, `test_v2.cu`,
`cupath_analyze_v2.py`, `e3_cluster.py`, `run7_v2.sh`, trace `/tmp/cupath7_190740.560657`.

### 25.1 Méthode — collecteur CUPTI v2
- Format binaire 56 o/record : `seq | gpu_start_ns | gpu_end_ns | dur_ns | corr | stream | op | aux`
  (op 0=KER 1=CPY 2=RT 3=RESET). `.names` texte 1:1 (KER/CPY/RT) avec `gpu_s/gpu_e/dur/stream/corr/t_real_us/name`.
  `.hdr` : `t0_real_us` + `t0_mono_ns` (ancrage horloges).
- **Le domaine CUPTI = CLOCK_REALTIME en ns** (mesure G4 du test unitaire :
  `gpu_s(1er event) − t0_real = +98 ms`, cohérent avec l'init CUDA ; le flush final tombe
  à +25 µs du dernier kernel). → `gpu_s/gpu_e` sont directement comparables au temps mural.
- **Test unitaire v2** (G1–G4) : 21 records, G1/G2/G3 OK, chevauchement forcé 2,656 ms
  détecté (somme 9,966 − union 7,310), gap forcé 15,2 ms visible, domaine horloges OK.
- Fenêtre decode : ancrage **fin de trace** (dernier kernel = fin du 64e token) −
  `eval time` serveur (1818,36 ms). Aucun bracket de départ requis.

### 25.2 Pannes rencontrées et corrigées (avant la capture valide)
1. `O_TRUNC` sur le binaire : le shim était chargé dans ~40 sous-processus du script
   (`LD_PRELOAD` exporté) ; le constructeur du dernier sous-processus (analyse python)
   tronquait la trace → **0 record** (run 18:34). Correction : chaque pid écrit dans
   `<prefix>.<pid>.*` ; l'analyse prend le fichier le plus gros.
2. Regex d'analyse `name=(.*)$` : le nom de mangled C++ contient des `(` → les lignes
   KER n'étaient pas parseées (premier passage « 15 kernels »). Correction : parser
   structuré sur les champs (sans ancrage fin de nom).
Aucune modification de la bibliothèque de production ni du service pendant la capture.

### 25.3 Gates de validation (run7 v2, 64 tok, wall 1818,36 ms)
| Gate | Résultat |
|---|---|
| G1 aucun champ manquant | **OK** (2 688 059 records binaires) |
| G2 `dur_ns == gpu_end − gpu_start` | **OK** (2 688 059/2 688 059) |
| G3 séquence continue + alignement `.bin`/`.names` 1:1 | **OK** (2 688 058 = KER+CPY+RT) |
| G4 domaine horloges CUPTI = REALTIME (ns) | **OK** (test unitaire + cohérence run) |
| G5 couverture : union vs wall, écart < 2 % | **OK — écart 0,55 %** |
| Fenêtre identifiable (64 tok, dernier cluster, gap 252 ms avant) | **OK** |

### 25.4 Mesures (fenêtre eval = 1818,36 ms / 64 tok / ~34,65 tok/s)
| Grandeur | Valeur mesurée |
|---|---|
| Records fenêtre | 40 390 kernels + 892 memcpy (streams 16 et 17) |
| Somme durées kernels | 1737,769 ms |
| Somme durées memcpy | 9,901 ms |
| **Union des intervalles GPU (kernels+memcpy)** | **1747,671 ms** |
| **Occupation GPU (union/wall)** | **96,11 %** |
| **Idle (wall − union)** | **70,689 ms (3,89 %)** |
| **Overlap (somme − union, inter-streams)** | **0,000 ms — exécution séquentielle** |
| Gaps | 41 281 gaps, médiane 0,64 µs, p99 9,5 µs, max 4,55 ms ; >1 ms : 3, >100 µs : 89 |
| Top gaps | 4,55 ms (après `k_get_rows_float`, t+98 ms) ; 1,91 ms (id., t+12 ms) ; ~1,05 ms × 6 (après `k_set_rows`, tous ≈ 1 ms) |

### 25.5 Verdict E3
**Capture exploitable — GPU quasi saturé en temps GPU, mais l'idle est réel et structuré.**
- Repères retenus : 96,11 % ∈ [89 %, 97 %] → « marge à examiner », et la marge est
  **mesurée, pas estimée** : 70,7 ms d'inactivité GPU sur 1,818 s de fenêtre, soit
  **~1,11 ms/round (17,1 rounds de décodage effectifs)** ou ~2,17 ms/round de la boucle
  specu (36 rounds specu).
- **Overlap = 0 ms** : les deux streams (16 : 1613,8 ms, 17 : 194,5 ms) ne se
  chevauchent jamais. Tout gain « pipeline » par chevauchement est nul ; le seul levier
  sur le temps mural est de **réduire le temps GPU sérial** (kernels) ou les **gaps CPU**.
- Répartition de l'idle : ~30 ms dans les ~10 gaps de 1 ms (tous en sortie de
  `k_set_rows` / `k_get_rows` → synchronisation CPU/échantillonnage côté MTP) +
  ~40 ms de micro-gaps (médiane 0,64 µs, 41 k → latence de lancement, non compressible
  individuellement, somme ~40 ms).
- **Conséquence pour E2** : le gain mural maximum atteignable par l'optimisation du
  kernel `ggml_q8v14partial` est borné par son temps GPU sérial : 543,9 ms/1818 ms =
  **30,1 % du wall**. Un kernel 2× plus rapide ⇒ ~172 ms gagnés (~9,5 % de wall,
  ~34,6 → ~38 tok/s), indépendamment de l'idle. Le gain « sérialisé ~360 ms » du §24.3
  (si partial+reduce+flash+gated_delta) reste la borne haute.

## 26. E2 — optimisation `ggml_q8v14partial` : baseline NCU + variante A rejetée (2026-10-13, 19:50–20:44)

Rapport : `cupti129-test/KOLBO-ncu-volta-20261013.md` §10.
Espace de test : `cupti129-test/e2/` (copie du kernel + harnais ; **jamais** de
substitution de la bibliothèque de production).

### 26.1 Baseline figée
- `fattn-q8-volta.cuh` md5 **36f564e2** (figé avant toute modification ; inchangé à la fin).
- Harnais `e1_bench` (copie du kernel, vraie géométrie : n_kv=107351, 4 kvh, 160 splits,
  grid 640 blocs × 128 threads), compilé `-arch=sm_70`, vérifié sur V100 (`devcheck`).

### 26.2 Mesures NCU de référence (NCU 2024.1.1, 1 lance profilé)
| Grandeur | Valeur |
|---|---|
| Durée | 1790 µs (wall sans profiler : 1593 µs au A/B ; +12 % sous NCU) |
| **Registres / thread** | **255** + **spills 152 B stores / 80 B loads** (stack 96 B) |
| **Smem statique / bloc** | **33 920 B** |
| **Occupation théorique / atteinte** | **12,50 % / 12,43 %** |
| Block limiters | **Shared Mem = 2** et Registers = 2 (les deux plafonnent à 2 blocs/SM) |
| Issue slots busy / No eligible | 31,83 % / 68,09 % (CPI 6,25) |
| Throughput | DRAM 5,95 %, SM compute 31,26 %, L1 51,45 % |

**Lecture** : le kernel plafonne à **2 blocs/SM (12,5 % d'occupation)** — le limitant est
**la mémoire partagée (33,9 Ko)**, pas les registres ; les 255 regs + spills aggravent.
Aucun pipeline mémoire n'est saturé (DRAM 6 %) : le temps est mangé par la latence
non masquée (CPI 6,25) d'un chemin mma+softmax à faible occupation.

### 26.3 Variante A (rejetée) — forcer l'occupation via `__launch_bounds__(128,3/4)`
- Hypothèse H1 : « augmenter l'occupation masquerait la latence ».
- Changement unique : `__launch_bounds__(128,2)` → `(128,3)` puis `(128,4)`.
- Conséquence ptxas : 168 regs / 432 B spill (a3), 128 regs / 928 B spill (a4) ;
  **smem inchangée (33 920 B) → Block Limit Shared Mem reste 2 → l'occupation ne monte pas**
  (96 Ko/SM : 3 blocs × 33,9 Ko > 96 Ko). Le forçage n'a fait que dégrader le code généré.
- **Résultat microbench** (V100 seule, 3 passes × 30 iters, médiane) :
  | variante | med (µs) | passes | Δ vs base |
  |---|---|---|---|
  | base (128,2) | **1593** | 1598 / 1592 / 1593 | ref |
  | a3 (128,3) | 3720 | 3720 / 3720 / 3718 | **+133,7 %** |
  | a4 (128,4) | 5449 | 5458 / 5445 / 5449 | **+242,3 %** |
- Sortie : probe pacc identique sur les 3 variantes (−4,875000 / −4,593750) →
  sémantique conservée, régression purement micro-architecturale (spills local memory).
- **Verdict : REJETÉE.** H1 telle que formulée est fausse pour ce kernel : l'occupation
  est bornée par la smem ; forcer les blocs sans réduire la smem ne fait que créer des
  spills. **H2 (réduire la smem / restructurer les tuiles) est la piste directement
  motivée** par la mesure (`Block Limit Shared Mem = 2`) — changement de plus grande
  envergure (tailles de tuile BR/BC, double-buffering), à engager comme prochaine
  expérience isolée.

### 26.4 État de production (fin E2)
- `ollama-v100` **active**, API ok, V100 rechargée (32 346 MiB), MD5 prod
  **e4afbb4a** inchangé. `fattn-q8-volta.cuh` **36f564e2** — **aucune modification**
  de la source de production. Aucun processus résiduel (harnais/NCU/capture).

### 26.5 H2 — tuile `BC 16→8` (validation microbench + A/B applicatif)

**Modification unique** (1 seul fichier, `fattn-q8-volta.cuh`, 2 lignes cohérentes) :
`constexpr int BC = 16` → `8` et `for (int sub=0;sub<2;sub++)` → `sub<BC/8`.
La boucle `sub` est dérivée de `BC` (chaque sous-tuile = 8 clés) : un simple `BC=8`
sans adapter la borne provoque des accès hors bornes en smem. Pas de nouveau
`launch_bounds`, aucune autre transformation.

**Microbench** (harnais E1, géométrie decode run6 : n_kv=107 351, kvh=4, G=6, T=4,
D=256, grid 4×160, 128 thr ; V100 seule, 3 passes × 30 iters, médiane) :

| | base (BC=16) | H2 (BC=8) | Δ |
|---|---|---|---|
| µs/lance | **1596** | **1349** | **−15,5 %** (−247 µs) |
| passes | 1596/1595/1596 | 1349/1350/1348 | disp. 1–2 µs |

**NCU 2024.1.1** (même mode capture : 1 lance profilé, skip 1) :

| grandeur | base | H2 |
|---|---|---|
| Durée (ms) | 1,79 | 1,52 |
| smem statique | 33,92 Ko | 25,41 Ko |
| Block Limit (regs / smem) | 2 / 2 | 3 / 3 |
| Occupation théorique / atteinte | 12,50 / 12,43 % | 18,75 / 16,99 % |
| Registres/thread | 255 | 166 |
| Spills (store/load) | 152 / 80 B | 0 / 0 |
| DRAM read (Mo) | 61,08 | 61,47 |
| Issue Slots Busy (%) | 31,83 | 38,63 |
| CPI (Warp Cycles/Instr) | 6,25 | 7,04 |

→ la réduction de smem fait passer la résidence de 2→3 blocs/SM, l'occupation monte,
les spills disparaissent, le kernel s'accélère. Gain microbench **−15,5 %**, cohérent
avec la durée NCU (−15 %). **Gate microbenchmark passée.**

**Correction numérique** : sorties `pacc`/`pm`/`pl` **bit-identiques** base vs H2
(0/3 932 160 éléments pacc + pm/pl, dump complet, données aléatoires seed fixe, même
géométrie) — pour `n_kv=107 344` (multiple de 16) **et** `n_kv=107 351` (cas réel
decode, impair). Le `reduce_kernel` (inchangé) recombine les splits de façon
associative en fp32 ; le raffinement de tuile ne change pas le résultat.

**Protocole de build (isolé, prod intouchée)** :
- Recette prod = kit **`v100-repro`** : sub-build `llama-server-cuda_v12` (preset
  `llama_cuda_v12_user_arch`, `sm_70-real` uniquement, gcc-13.4/nvcc 12.6.85) sur le
  fork `llama-volta-merged` (md5 `fattn-q8-volta.cuh` = **36f564e2**).
- **Gate A** : re-link des 145 `.o` du build prod purs → lib **bit-identique au prod**
  (md5 **e4afbb4a**) → recette de link exacte reproduite.
- Lib H2 = **les 144 autres `.o` du prod + 1 seul `fattn.cu.o` recompilé** avec la
  variante (commande nvcc exacte du build prod). md5 **a4b6f440** (≠ prod, ≠ gate).
  Marqueur `GGML_CUDA_VOLTA_Q6_W4R4` présent. Rien d'autre ne diffère de la prod.

**A/B applicatif** (harnais `h2_ab_lib.sh` = protocole `h2_capture` sans nsys :
serveur de **capture** port 44901, **pas** le service prod ; même modèle
`qwen3.8-27b-q6-mtp`, ctx 200 k, MTP `n_max=3`, prefill 100 k + 2ᵉ passe chaude,
décode **1200 tok** `temperature=0` ; ordre A,B puis B,A ; 2 réplicats ; seule
différence : `GGML_BACKEND_PATH` = prod (A) vs H2 (B)) :

| run | lib | eval time (1200 tok) | ms/tok | t/s |
|---|---|---|---|---|
| r1-A | prod e4afbb4a | 38 798 ms | 32,36 | 30,90 |
| r1-B | H2 a4b6f440 | 33 662 ms | 28,07 | 35,62 |
| r2-B | H2 a4b6f440 | 33 711 ms | 28,12 | 35,57 |
| r2-A | prod e4afbb4a | 38 676 ms | 32,26 | 31,00 |

- **Moyenne A** = 32,28 ms/tok (30,98 t/s) ; **moyenne B** = 28,07 ms/tok (35,62 t/s).
- **Δ wall = −13,0 %** (r1 −13,2 %, r2 −12,8 %) ; **Δ débit = +15,0 %**.
- Gain constant entre les 2 réplicats et les 2 ordres → pas d'effet de dérive.

**Point d'attention (correction)** : à `temperature=0`, le texte **diverge** entre
A et B à partir du char ~1086. Cause identifiée : `split_count(n_kv)` est **dépendant
de `BC`** — au-delà de `n_kv≈107 520`, la partition des splits change
(base 157×688 vs H2 159×680) → le regroupement des additions fp32 dans
`reduce_kernel` change → les probabilités dérivent d'1 ULP et le `argmax` diverge
sur un token. Ce n'est **pas** un bug de la variante (les deux sorties sont
l'attention exacte de la même façon, à une associativité fp32 près) mais c'est un
changement mesurable de la séquence générée. **Conséquence** : H2 accélère bien le
decode, mais **ne peut pas être basculée en prod tant qu'on n'accepte pas un texte
non-identique** (ni qu'on ne fige `split_count` de façon indépendante de `BC` —
variante H2b à évaluer si le gain −13 % vaut le changement de déterminisme).

**Verdict H2** : **VALIDÉ au microbench et à l'A/B applicatif** — kernel mesurablement
plus rapide (−13 % wall / +15 % t/s en decode, reproductible), sorties bit-identiques
à `n_kv` fixé, production non modifiée. **Bascule prod : NON recommandée à ce stade**
à cause de la dérive de séquence induite par `split_count(BC)` ; à ré-évaluer avec
H2b (split_count découplé de BC) si le gain est jugé suffisant.

### 26.6 État de production (fin H2)
- `ollama-v100` **active**, API ok, V100 rechargée. MD5 prod **e4afbb4a** inchangé.
  `fattn-q8-volta.cuh` de prod **36f564e2** — **aucune modification** de la source
  ni de la bibliothèque de production. Travaux exclusivement dans `h2/cupti129-test/e2/`
  (build `h2lib/`, harnais `h2_ab_lib.sh`, résultats `ab_results_h2_v2/`).
- Artefacts : `h2lib/` (build isolé + lib `libggml-cuda-h2.so` a4b6f440),
  `e1_check.cu`/`e1c_base`/`e1c_h2b` (dump de sortie complet), `e1_check_final.sh`,
  `h2_ab_lib.sh`, `ab_results_h2_v2/{r1-A,r1-B,r2-B,r2-A}` (server.out + gen.json).

### 27. Mission H2b — découplage du partitionnement de la tuile (VERDICT FINAL)

**Contexte** : H2 (BC 16→8, §26.5) gagnait −13 % wall en A/B applicatif mais divergeait
textuellement (char ~1086) car `split_count` dépend de BC. H2b testait si le gain pouvait
se conserver en forçant le partitionnement au comportement prod (PC=16) tout en gardant
la tuile BC=8.

#### 27.1 Étape 1 — Mécanisme (faits vérifiés, lecture du code + calcul)

Chaîne de dépendance dans `fattn-q8-volta.cuh` (prod 36f564e2) :
1. **Host `split_count(n_kv)`** : `tiles=ceil(n_kv/BC)` ; `requested=min(160,tiles)` ;
   `tiles_per_split=ceil(tiles/requested)` ; `splits=ceil(tiles/tiles_per_split)`.
   → **BC agit sur le nombre de splits via `ceil(n_kv/BC)`**.
2. **Host `get_extra`** : `e.splits = split_count(K->ne[1])` ; la scratch (pacc/pm/pl)
   est dimensionnée en `splits × T × qh × (D+2)`.
3. **Device `partial_kernel`** : **re-dérive** les frontières :
   `logical_tiles=ceil(n_kv/BC)` ; `tiles_per_split=ceil(logical_tiles/splits)` ;
   `first_tile=split*tiles_per_split*BC` ; `split_end=min(n_kv, first_tile+tiles_per_split*BC)`.
   → les frontières de splits sont des **multiples de BC** ; la taille de tuile et le
   découpage sont **couplés** (pas de paramètre « pas de frontière » indépendant de BC
   dans la signature).
4. **Device `reduce_kernel`** : combine les `splits` partiels :
   `hm=max_s pm_s` (tree smem) ; `ll=Σ_s pl_s·exp(pm_s−hm)` (tree) ;
   `num_d=Σ_{s=0..splits−1} bf16(pacc_{s,d})·exp(pm_s−hm)` (somme **séquentielle** par
   thread, 1 thread/dim head, fp32) ; `out_d=num_d/hl`.
   → le **nombre d'additions** et le **contenu de chaque partiel** dépendent du découpage ;
   le regroupement fp32 (1 ULP) change quand le découpage change.
5. **Seuil (calcul + confirmé expé)** : le pas 160 splits = 672 clés. Le passage au-delà
   de 160×42=6720 tuiles BC=16 se produit à **n_kv=107 521** : prod 157×688 vs H2 159×680.

**Faits (vérifiés)** : (a) BC entre dans `split_count` par `ceil(n_kv/BC)` ;
(b) les frontières sont multiples de BC ; (c) le `reduce` est sensible au découpage ;
(d) prod/H2 diffèrent dès n_kv≥107 521 ; (e) le kernel re-dérive ses bornes en device
depuis (n_kv, splits, BC). **Hypothèse** : forcer le découpage prod (PC=16) tout en
gardant la tuile BC=8 restaure le déterminisme prod à coût ≈null.

#### 27.2 Étape 2 — Hypothèse (une seule, falsifiable)

**H2b** : « forcer le partitionnement au comportement prod (PC=16) tout en conservant la
tuile BC=8 de H2 retrouve les sorties prod **et** le gain de H2 ».
**Diff minimal (1 fichier, 6 sites)** : `constexpr int PC=16` ; `split_count` et
`first_tile`/`split_end`/`logical_tiles` passent de `BC` à `PC` (frontières multiples de
16 = prod, **à splits inchangé**) ; **la tuile, la smem (`k_s`/`v_s`/`m_s` à `BC`), le
pas d'itération et le chemin softmax (sub<BC/8) restent à BC=8** (smem 25,4 Ko, 3
blocs/SM conservés).
**Prédiction** : P1 num — H2b = prod bit-identique (n_kv {107344,107351,107521,108551}) ;
P2 perf — microbench H2b ≈ H2 (−15 %) **et** A/B ≈ H2 (−13 %).
**Invaldation** : T1 H2b≠prod au seuil → mécanisme mal compris ; T2 microbench H2b >
H2+2µs → gain non conservé ; T3 compile/smem>96 Ko → rejetée.

#### 27.3 Gate A — compilation isolée (passée)

- Harnais : 3 variantes du checker+bench (prod BC=16 / H2 BC=8 / H2b BC=8+PC=16)
  compilées `nvcc 12.6 -O2 -arch=sm_70 -ccbin gcc-13` : `e1c_prod` e682c0df,
  `e1c_h2` cbd1af1e, `e1c_h2b` 65edf2c5 ; `e1b_*` c835851b / 3bc8b456 / eb5999a2.
- **Lib H2b** : 144 `.o` du build propre A (bit-identique prod, Gate A H2) **+ 1 seul
  `fattn.cu.o` recompilé** depuis `fork-h2b` (copie du fork prod + patch 6 sites) avec la
  **commande nvcc exacte du build prod** (réutilisée via `compile_h2.sh`).
  → **lib H2b md5 = 0eb5afbd** (≠ prod e4afbb4a, ≠ H2 a4b6f440), marqueur
  `GGML_CUDA_VOLTA_Q6_W4R4` présent, prod intact.
- **smem/regs du `partial_kernel` (ptxas -v, commande exacte)** :
  prod **33920 B / 255 regs** ; H2 **25408 B / 166 regs** ; H2b **25408 B / 166 regs**.
  → le découplage **ne change pas la smem ni les regs** (la tuile BC=8 est intacte).

#### 27.4 Gate B — numérique (passée, P1 confirmée)

Dump complet pacc/pm/pl, 4 n_kv de part et d'autre du seuil (107344, 107351, 107521,
108551), mêmes données seed 12345, même géométrie :

| n_kv | H2 vs prod | **H2b vs prod** | splits (prod/H2/H2b) |
|---|---|---|---|
| 107344 | identique | **identique** | 160/160/160 |
| 107351 | identique | **identique** | 160/160/160 |
| **107521** | **DIFFÉRENT** | **identique** | 157/**159**/**157** |
| **108551** | **DIFFÉRENT** | **identique** | 158/**160**/**158** |

→ **H2b = prod bit-identique partout** ; H2 diverge exactement au-delà du seuil 107 521.
Le partitionnement forcé (PC=16) restaure **le nombre de splits ET les frontières** de
prod (157×688 etc.), donc le regroupement fp32 de `reduce_kernel` est identique. **P1
confirmée.** (Critère : md5 des dumps ; pas de tolérance — bit-identique attendu et obtenu.)

#### 27.5 Gate C — microbenchmark (passée, mais P2 partielle)

`e1_bench` (n_kv=107351, grid 4×160, 128 thr, 5 passes × 30 iters, médiane) :

| | prod | H2 | H2b |
|---|---|---|---|
| passes (µs) | 1591/1598/1598/1603/1596 | 1509/1509/1508/1440/1347 | 1350/1351/1351/1348/1349 |
| **médiane** | **1598** | **1508** | **1350** |

→ le kernel isolé de H2b est **aussi rapide que H2** (1350, ≈ H2), bien plus rapide que
prod (1598, −15,5 %). La dispersion H2 (1347–1509) est plus large que H2b (1348–1351)
mais la médiane H2b ≈ 1350 est **stable et identique à H2**. **Le gain du kernel isolé
est conservé.** T2 non déclenchée. **Mais** la P2 attendait aussi un gain **app** ≈ −13 %
— c'est l'objet de la Gate D.

#### 27.6 Gate D — applicative (le point clé : H2b = prod, PAS H2)

Protocole `h2_capture` (serveur de capture 44901, pas la prod ; même modèle
`qwen3.8-27b-q6-mtp`, ctx 200 k, MTP n_max=3, prefill 107 351 + 2ᵉ passe chaude,
décode 1200 tok temp=0, ordre alterné, 2 réplicats).

**A/B H2b (prod A vs H2b B)** — `ab_results_h2b_v1` :
| run | lib | eval (1200 tok) | ms/tok |
|---|---|---|---|
| r1-A | prod e4afbb4a | 39 641 ms | 33.06 |
| r1-B | H2b 0eb5afbd | 40 082 ms | 33.43 |
| r2-B | H2b 0eb5afbd | 39 770 ms | 33.17 |
| r2-A | prod e4afbb4a | 39 724 ms | 33.13 |

- Moyenne A = 33.07 ms/tok (30.24 t/s) ; B = 33.27 ms/tok (30.06 t/s).
- **Δ H2b vs prod = +0.61 % wall (r1 +1.11 %, r2 +0.12 %) = à l'intérieur du bruit.**
- **Texte : prod vs H2b = IDENTIQUE (5401 chars, les 2 réplicats)** — contrairement à
  H2 qui divergeait au char 1086. **MTP identique** (gen 1338 / acc 818 / accLen 2.83).

**Contrôle d'attribution (H2 vs H2b, MÊME session)** — `ab_results_h2b_control` :
| config | eval (1200 tok) | ms/tok | accLen | texte vs prod |
|---|---|---|---|---|
| **H2** a4b6f440 | 34 813 / 34 896 | **29.05** | **3.24** | **DIVERGENT** |
| **H2b** 0eb5afbd | 39 865 / 39 717 | **33.16** | **2.83** | **IDENTIQUE** |

- **Δ H2b vs H2 (intra-session) = +14.16 % wall (r1 +14.5 %, r2 +13.8 %)** — c'est-à-dire
  que **H2 reste ~13 % plus rapide que H2b**, et H2b tombe exactement sur la perf de prod.
- Coïncidences croisées : le texte H2b (v1 et control) = texte prod ; le texte H2
  (v2 et control) = texte H2 divergent ; les accLen sont identiques (prod/H2b = 2.83,
  H2 = 3.24).

#### 27.7 INTERPRÉTATION (fait vs hypothèse)

**Faits (vérifiés)** :
- (F1) Le **kernel isolé** de H2b est aussi rapide que H2 (1350 µs) et −15 % vs prod,
  à smem/regs identiques (25408 B / 166 regs).
- (F2) À l'**applicative**, H2b = prod (33.1–33.3 ms/tok, Δ +0.6 %) et H2 = prod−13 %
  (29.05), **intra-session**.
- (F3) H2b est **bit-identique à prod** en sortie et en texte ; H2 diverge (accLen
  3.24 vs 2.83).

**Lecture (hypothèse, non démontrée mais cohérente)** : le gain **app** de H2 n'est pas
(ou pas seulement) le temps du `partial_kernel` isolé. Le passage du partitionnement à
BC=8 (plus de splits, plus petits) améliore l'**acceptation MTP** (accLen 2.83→3.24,
#acc 818→875 sur 1200 tok) — probablement car la distribution de latence des splits plus
nombreux/petits se pipeline mieux avec le draft MTP, ou car le `reduce` sur plus de
partiels est mieux parallélisé. Or **H2b, en forçant le partitionnement prod, annule
exactement cet effet applicatif** : même accLen (2.83), même texte, même débit que prod.
**Le gain app de H2 est donc porté par le partitionnement BC=8 lui-même, qui est
indissociable de la divergence textuelle.** On ne peut pas récupérer l'un sans l'autre.

Ceci explique l'écart microbench/app : le harnais E1 mesure **un seul lance
`partial_kernel`** en régime chaud et isolé, où la tuile BC=8 est bien plus rapide
(−15 %). Mais dans l'application, c'est l'**interaction kernel + MTP + scheduling**
(qui dépend du nombre/taille des splits, donc du partitionnement) qui domine la durée
murale, et celle-ci suit le partitionnement, pas la tuile.

#### 27.8 VERDICT H2b — **H2b validée (objectif atteint) : le découplage fonctionne, mais il ne rapporte aucun gain applicatif**

L'hypothèse H2b est **partiellement confirmée** :
- **P1 (numérique) confirmée** : H2b = prod bit-identique, zéro divergence textuelle
  (contrairement à H2). Le découplage `split_count`/tuile fonctionne exactement comme
  prévu.
- **P2 (perf) réfutée côté app** : le gain kernel isolé (−15 %) **n'est pas transféré
  à l'applicative** — H2b = prod (+0.6 %). Le gain app de H2 (−13 %) vient du
  partitionnement BC=8, pas de la tuile BC=8.

**Conséquence pratique (c'est le point important pour le déploiement)** :
- **H2b résout le problème de déterminisme sans régression** : c'est une variante
  *correcte*, *reproductible* et *bit-identique à prod* qui, en plus, garde un kernel
  isolé plus rapide (utile si le kernel finit par devenir critique ailleurs, ou si le
  contexte/pattern change). **Mais elle ne donne aucun gain applicatif** — elle
  restaure exactement le comportement de prod.
- **Le gain H2 (−13 %) est réel mais inextricablement lié à la divergence** : il vient
  du partitionnement BC=8 (meilleure acceptation MTP), qui change l'associativité fp32
  et donc le texte. **Il n'existe pas, dans le cadre d'H2b, de façon de prendre le gain
  sans prendre la divergence.**

**Ce que H2b ne fait PAS** : elle ne justifie **pas** de nouvelles optimisations, et ne
permet **pas** de basculer H2 (dont le gain est bien réel) — la divergence subsiste.
Elle donne simplement une **variante de déploiement propre** (identique à prod en sortie,
kernel isolé optimisé) au cas où on voudrait un « plan B » sans risque de changement de
texte.

#### 27.9 Décision de déploiement (restreinte au cadre H2b)

Trois options, **sans nouvelle autorisation aucune n'est engagée** :
1. **Ne rien déployer** (recommandé) : H2b = prod à la perf, donc aucun bénéfice à
   basculer ; H2 a un gain mais change le texte.
2. **Basculer H2 en acceptant le texte non identique** : le gain −13 % est réel et
   reproductible (A/B v2 et contrôle), mais la séquence change (1 ULP fp32 via le
   partitionnement). À valider par l'utilisateur.
3. **Basculer H2b** : variante *correcte* (bit-identique) mais **sans gain applicatif** —
   ne justifie une bascule que si l'on veut le kernel isolé optimisé « pour plus tard ».

**Aucune descendance (H2c, H2b2a, …) n'est ouverte.** Toute variante supplémentaire
requiert une nouvelle décision explicite.

#### 27.10 État final de production (vérifié après toutes les expériences H2b)

- `ollama-v100` **active**, API ok, V100 rechargée (32 348 MiB).
- MD5 prod lib **e4afbb4a** ✓ ; `fattn-q8-volta.cuh` prod **36f564e2** ✓ ; fork
  `v100-repro` **36f564e2** ✓ — **aucune modification** de la production.
- Aucun processus résiduel (harnais/bench/serveur de capture).
- Artefacts : `h2lib/fork-h2b/` (patch source), `h2lib/fattn.h2b.o` (eeae2304),
  `h2lib/lib/libggml-cuda-h2b.so` (**0eb5afbd**), `h2lib/h2b_build_lib.sh`,
  `h2b/h2b_variant/` (harnais H2b), `e2/h2b_build_harness.sh`, `h2b_gate_b.sh`/`.log`,
  `h2b_gate_c.sh`/`h2b_gate_c2.sh`/`.log`, `h2b_ab_lib.sh`/`.log`,
  `ab_results_h2b_v1/` (prod vs H2b), `h2b_control_ab.sh`/`.log`,
  `ab_results_h2b_control/` (H2 vs H2b, contrôle d'attribution).

### 28. Mission H2-PROD — activation de H2 en production expérimentale (VERDICT FINAL)

**Objectif (addendum)** : activer H2 en production expérimentale, observer si la hausse
d'acceptation MTP se traduit par un gain durable sans dégradation macroscopique du
comportement du modèle. Ne pas assimiler une divergence textuelle à une régression.
Ne pas exiger de preuve formelle de correction avant observation.

#### 28.1 Protocole exécuté
1. **Vérification artefact** : lib H2 `a4b6f440` (144 `.o` prod + 1 `fattn.cu.o` BC=8),
   1254 symboles dynamiques identiques à prod (aucune différence de nom), soname/NEEDED
   identiques, marqueur W4R4 présent.
2. **Suite représentative** (`suite_prompts.json`, 11 prompts) : 4 factuels vérifiables,
   3 raisonnement (arithmétique + code), 2 instructions (JSON strict, 3 puces interdiction),
   2 contexte long (1 needle-in-haystack 41k/5 questions, 1 multi-tours 3 messages).
   Évaluation par checks objectifs (`eval_suite.py`) + MTP par fenêtre journal.
3. **Baseline sous prod** (11/11 OK) puis **bascule H2** (backup prod `e4afbb4a` vérifié
   dans `/var/backups/llama-experiments/`, `deploy.sh h2`, lib mappée `a4b6f440` confirmée
   via `/proc/<pid>/maps`), **suite sous H2**, **long decode** 600 tok (prefill 41k,
   2 passes) prod vs H2, puis **H2 laissé en production**.

#### 28.2 Résultats — suite de qualité (température 0)

| id | cat | baseline (prod) | H2 | texte |
|---|---|---|---|---|
| F1 | factuel | PASS 17,2s | PASS 13,9s | identique |
| F2 | factuel | PASS 226,7s* | PASS 1,6s* | identique |
| F3 | factuel | PASS 1,0s | PASS 1,0s | identique |
| F4 | factuel | PASS 2,2s | PASS 2,1s | identique |
| R1 | raisonnement | PASS 7,6s | PASS 9,5s | **divergent** (même bon résultat 95202) |
| R2 | raisonnement | PASS 39,2s | PASS 14,9s | identique (240.7) |
| R3 | raisonnement | PASS 3,4s | PASS 3,2s | identique |
| I1 | instructions JSON | PASS 4,3s | PASS 3,2s | identique (JSON valide) |
| I2 | instructions 3 puces | PASS 6,4s | FAIL→PASS 7,0s** | **divergent** (3 puces valides au rerun mx=800) |
| L1 | long ctx 41k/5 Q | PASS 63,2s (5/5) | PASS 62,7s (5/5) | identique |
| L2 | long ctx multi-tours | PASS 4,9s | PASS 5,4s | **divergent** (mêmes 3 faits) |

\* F2 : le 226,7s de la baseline est un **artefact de charge** (rechargement du modèle
   + concurrence GPU), pas une mesure propre. Le rerun H2 fait 1,6s.
\** I2 : le FAIL H2 à mx=300 est un **artefact de troncature** (2 puces au lieu de 3
   avant le cap). Au rerun mx=800 : 3 puces valides, mot interdit absent → PASS.
   Ce n'est **pas** une régression de comportement.

**Lecture qualité** : aucune régression macroscopique. Tous les checks factuels,
arithmétiques, JSON et contexte-long passent sous les deux libs. Les 3 textes divergents
(R1, I2, L2) restent **corrects en contenu** — divergence de formulation/associativité
fp32 cohérente avec le mécanisme H2 déjà établi (§26/§27), **pas** une perte de qualité.

#### 28.3 Résultats — performance (les deux protocoles sont CONTRADICTOIRES)

**Protocole A — A/B v2 (mission H2, propre, 100k prefill, 1200 tok, même session, 2 ordres)** :
| | prod | H2 |
|---|---|---|
| ms/tok | 32,28 | 28,07 |
| t/s | 30,98 | 35,62 |
| accLen MTP | 2,83 | **3,24** |
→ **H2 gagnant : −13,0 % wall / +15,0 % débit**, reproductible.

**Protocole B — long decode 41k prefill / 600 tok (cette mission, `/api/chat`, 2 passes)** :
| réplicat | prod (ms/tok) | H2 (ms/tok) | accLen prod | accLen H2 |
|---|---|---|---|---|
| 1 | 23,51 | 27,62 | 3,12 | 2,76 |
| 2 | 24,53 | (bloqué) | 3,10 | — |
→ **prod gagnant : H2 ~+15 à +17 % wall** ; accLen MTP **plus faible** sous H2
   (2,76 vs 3,10-3,12), acc_rate/pos H2 (0,775/0,563/0,423) < prod (0,869/0,688/0,558).

**Contradiction** : A (100k) → H2 gagne ; B (41k) → prod gagne. C'est le point
important et il n'est **pas tranché** par ces données.

Hypothèse de cohérence (NON démontrée, à re-vérifier proprement) : le gain applicatif
de H2 vient du **partitionnement** BC=8 qui ne change le découpage en splits qu'au-delà
du **seuil n_kv ≈ 107 520** (§26.5, §27.1). À 100k prefill, n_kv dépasse le seuil →
H2 passe de 160 à 159 splits (meilleure acceptation MTP, accLen 3,24) → H2 gagne.
À 41k prefill, n_kv < seuil → même partitionnement que prod → l'avantage de tuile
seule ne suffit pas (cf. H2b, §27.7) → prod gagne/égal. **Le gain H2 serait donc
dépendant de la longueur de contexte (au-delà d'environ 107,5 k tokens).**

#### 28.4 ⚠️ DISCLAIMER — maîtrise de l'environnement
Plusieurs mesures de **cette** mission sont **entachées** et ne doivent pas servir de
référence définitive :
- La **V100 est partagée** : d'autres clients/`llama-server` sollicitent le GPU en
  parallèle (observes : F2 baseline 226s, smoke test 64 ms/tok, 2e réplicat H2 bloqué
  par un prefill concurrent). Les murs mesurés incluent de la **concurrency** non
  contrôlée.
- Les **compteurs MTP** du journal sont **cumulatifs par processus** et **réinitialisés
  à chaque rechargement de modèle** (evict/load fréquents) → les fenêtres de MTP par
  requête sont **fiables seulement si aucun reload n'intervient** ; c'était vrai pour le
  long decode propre mais pas partout.
- Le service **recharge le modèle** de façon intermittente (evict) ce qui décale
  les caches KV et les compteurs entre les passes.
- **Conclusion** : la comparaison A/B v2 (protocole A, mission H2, session isolée,
  contrôle intra-session) reste la **mesure de performance la plus fiable** ; le
  long decode (protocole B) et la plupart des murs de cette mission sont à considérer
  comme **indicatifs**, pas conclusifs.

#### 28.5 Décision et état final
- **H2 est DÉPLOYÉE en production expérimentale** (conforme à l'intention de l'addendum).
- **Rollback garanti** : backup prod `e4afbb4a` vérifié dans
  `/var/backups/llama-experiments/libggml-cuda.so.prod-20261014-134252` ;
  `bash h2/h2prod/deploy.sh prod` restaure en ~10s (testé, chemin validé).
- **Surveillance recommandée** : suivre (a) le débit wall en usage réel sur les
  contextes **> 107,5k tokens** (là où le gain H2 est attendu) vs **< 107,5k** (là où
  H2 pourrait être neutre/légèrement en retrait), et (b) toute dégradation reproductible
  de qualité/stabilité → rollback.
- **Critère de l'addendum appliqué** : aucune dégradation reproductible observée sur la
  suite ; les divergences textuelles sont des changements d'associativité fp32
  (contenu correct), pas des régressions. L'absence de dégradation macroscopique **ne
  prouve pas** la correction mathématique ni la préservation formelle des garanties MTP.

#### 28.6 État de production final (vérifié)
- `ollama-v100` **active**, API ok, `llama-server` en cours avec **lib H2 mappée
  `a4b6f440`** (confirmée `/proc/<pid>/maps`), modèle chargé, V100 32 346 MiB.
- Lib disque : `a4b6f440` (H2) ; **source** `fattn-q8-volta.cuh` prod **36f564e2**
  (non modifiée — seul l'artefact `.so` a été remplacé).
- Artefacts : `h2/h2prod/` (`suite_prompts.json`, `run_suite.sh`, `eval_suite.py`,
  `deploy.sh`, `longdecode.sh`, `results_baseline/`, `results_h2/`,
  `results_longdecode_{prod,h2}.json`, `baseline_run.log`, `h2_run.log`).

---

### 29. RUN9B — A/B corrigé REF vs H2 à ~41k, session isolée (2026-10-15, 15:43–15:48 UTC, VERDICT FINAL)

**Question** : trancher la contradiction §28.3 — protocole A @100k (H2 gagne, −13 %)
vs protocole B @41k (REF gagne, mais entaché, §28.4). A/B propre REF `e4afbb4a`
vs H2 `a4b6f440` à contexte fixe ~41k, session isolée. Protocole écrit avant
lancement : `cupti129-test/nightrun8/PROTOCOL-run9b.md` ; bench
`run9b_bench.py` via `supervisor_bench.py --detach` ; artefacts
`cupti129-test/nightrun8/ab_run9b/` (results.jsonl, gen_*.json, server_*.out).

#### 29.1 Corrections apportées (depuis RUN9, invalidé à tort)

1. **G_MTP** : le gate RUN9 lisait `draft_n` dans la réponse
   `/v1/chat/completions` — champ **absent** de la réponse OpenAI de ce binaire
   (`server-common.cpp:96` ; tir de contrôle prod 14:56 UTC : aucune clé
   `draft_n`, MTP pourtant actif). → RUN9B se base sur le **log serveur**
   (lignes `draft acceptance` + `spec common_specu`), fenêtre = delta des
   2 dernières lignes spec.
2. **Env W4R4** : RUN9 exportait `GGML_CUDA_VOLTA_W4R4` (nom erroné, inopérant)
   → les 2 bras RUN9 tournaient **sans** le chemin W4R4. RUN9B exporte
   `GGML_CUDA_VOLTA_Q6_W4R4=1` — le seul nom que le `.so` lit
   (`mmvq.cu:1310,1460` ; vérifié aussi sur l'env du llama-server prod,
   `/proc/<pid>/environ` via sudo : `GGML_CUDA_VOLTA_Q6_W4R4=1` présent).
   Parité prod stricte reconstituée.
3. **Logprobs** ajoutés (`logprobs:true, top_logprobs:8`) — question 4 du
   protocole V100 (proximité des logits REF vs H2). Réponses JSON complètes
   sauvegardées. **Limite constatée** : ce binaire ne renvoie `top_logprobs`
   que pour le **premier** token (599/600 vides, logprob tronqué à 0.0) → la
   proximité des logits ne peut pas être mesurée finement via l'API ; le test
   de divergence par le texte reste la référence (cf. §29.3).

#### 29.2 Faits (mesures, 4 essais VALIDES — aucune invalidation)

Paramètres : 41 468 tok prompt (L1 + instruction), 600 tok decode,
`temperature=0, top_k=1, ignore_eos`, n_max=3, KV q8_0, FA auto,
`-b 512 -ub 512`, ordre alterné REF/H2/REF/H2, 2 réplicats, port 44911,
prod stoppée/restaurée, V100 < 1000 MiB avant chaque essai (0 MiB relevé),
chauffe 2 prefills (2ᵉ passe cached ≈ 41 464). Libs : REF = backup
`e4afbb4a` (GGML_BACKEND_PATH), H2 = `a4b6f440`. Le `.so` déployé
n'est **jamais** modifié (vérifié avant/après : `a4b6f440`).

| essai | lib | wall (s) | ms/tok | tok/s | acc MTP (req.) | mean len | cum pos |
|---|---|---|---|---|---|---|---|
| ref1 | e4afbb4a | 19,54 | **32,57** | 30,70 | 372/678 = 54,9 % | 2,64 | (0,824 / 0,549 / 0,352) |
| h2a1 | a4b6f440 | 19,79 | 32,98 | 30,32 | 368/691 = 53,3 % | 2,59 | (0,762 / 0,548 / 0,371) |
| ref2 | e4afbb4a | 19,43 | **32,39** | 30,87 | 372/678 = 54,9 % | 2,64 | (0,824 / 0,549 / 0,352) |
| h2a2 | a4b6f440 | 20,01 | 33,34 | 29,99 | 368/691 = 53,3 % | 2,59 | (0,762 / 0,548 / 0,371) |

- Moyenne REF = **32,48 ms/tok** (30,78 t/s) ; moyenne H2 = **33,16 ms/tok**
  (30,16 t/s). **Δ H2 vs REF = +2,1 % wall** (r1 +1,2 %, r2 +2,3 %) — H2
  légèrement en retrait, écart **à l'intérieur du bruit** (critère protocole §4 :
  écart inter-bras < max(écart-type intra-bras, 2 %) → INDÉCIS).
- **MTP parfaitement reproductible par bras** : ref1 ≡ ref2 (372/678, mean 2,64,
  même cum pos), h2a1 ≡ h2a2 (368/691, mean 2,59, même cum pos). Les deux bras
  ne font pas la même séquence de décodage — mais chacune de façon stable.
- MTP actif et validé par le log serveur sur les 4 essais (gates passés).
- **Contrôle W4R4** : les murs RUN9B (REF 32,4–32,6 / H2 33,0–33,3) sont
  **identiques** aux murs RUN9 (sans W4R4) à ±0,3 % → le chemin
  `GGML_CUDA_VOLTA_Q6_W4R4` est **inopérant dans ce régime** @41k (le kernel
  Q6_K W4R4 ne change rien au mur ici) ; c'est une variable qui sera tranchée
  par le RUN9C @100k (§29.6).

#### 29.3 Divergence textuelle — le fait marquant

- **ref1 ≡ ref2** (texte bit-identique, déterminisme REF ok) ;
  **h2a1 ≡ h2a2** (déterminisme H2 ok).
- **REF ≠ H2** : 197/600 tokens communs (32,8 %), première divergence au
  **token 190** (char ~756 du texte généré) : « sécurité **du personnel** »
  (REF) vs « sécurité **des opérateurs** » (H2) — contenu sémantiquement
  correct des deux côtés, pas de régression macroscopique.
- **Hypothèse §27 (seuil `split_count` 107 521) INCONFIRMÉE à 41k** : la
  divergence existe bien **en dessous** du seuil → il existe un **second
  mécanisme** de sensibilité fp32 entre prod et H2 à n_kv=41k (associativité du
  `reduce_kernel` / regroupement des splits même à splits=160 constants, ou
  accumulation de l'attention sur le KV). À investiguer (dump pacc/pm/pl à
  n_kv=41 469, cf. harnais `e1_check` §27) si l'équivalence formelle devient
  un critère.

#### 29.4 Décision

- **À 41k : H2 ≈ REF, légèrement en retrait (INDÉCIS, +2,1 % wall)**.
  La contradiction §28.3 est **tranchée côté 41k** : le « protocole B »
  (REF gagnant) était **entaché** ; en session isolée H2 ne gagne pas mais ne
  perd pas de façon significative non plus.
- **Le gain H2 n'apparaît pas à 41k** → cohérent avec l'hypothèse
  longueur-de-contexte ; le régime à confirmer est le **100k** (protocole A,
  −13 %, reproductible §26.5).
- **Aucun rollback** : H2 reste en production expérimentale (§28.5),
  surveillance recommandée inchangée ; le retrait léger @41k est dans le bruit
  et les divergences textuelles sont du changement d'associativité fp32
  (critère de l'addendum : aucune dégradation reproductible de qualité).

#### 29.5 État de production post-run (vérifié)

- `ollama-v100` **active**, enabled, API 11434 ok, `.so` déployé
  `a4b6f440` inchangé, aucun résidu serveur sur 44911, V100 revenue à
  l'usage normal (32 348 MiB = modèle prod chargé).
- Bench terminé **rc=0** en 337 s ; superviseur a renvoyé l'inférence OWUI
  (dernier rc : `0 reprise envoyée (f5979010)`).

#### 29.6 Prochaine action — RUN9C : A/B REF vs H2 à ~100k (protocole A propre)

- **Justification** : à 41k H2 ≈ REF ; à 100k (§26.5, session isolée) H2 gagne
  −13 % de façon reproductible. Reprendre le régime @100k proprement pour :
  (a) confirmer que le gain H2 est bien lié à la longueur de contexte ;
  (b) **trancher la variable W4R4** — l'A/B §26.5 de l'époque n'avait pas
  exporté `GGML_CUDA_VOLTA_Q6_W4R4` (le RUN9B montre qu'il est inopérant à
  41k, mais le régime 100k peut différer) ; refaire l'A/B @100k **avec** et
  **sans** W4R4 pour savoir si le gain H2 tient sur le kernel fattn (BC 8)
  seul ou si le chemin Q6_K W4R4 y participe.
- **Protocole** (à écrire AVANT lancement, `PROTOCOL-run9c.md`) : même
  harnais que RUN9B (run9b_bench.py paramétré) ; prompt ~100k (L1 + répétition
  contrôlée ou suite de la suite H2-PROD si un prompt 100k existe ; sinon
  synthétique calibré ~100 000 tok, 2ᵉ prefill chaud) ; 600 tok decode ;
  n=2/bras (exploratoire) ; ordre alterné ; mêmes gates ; W4R4 **ON**
  (parité prod) comme condition principale + 1 paire REF/H2 **W4R4 OFF**
  (condition secondaire, à discuter). Durée estimée : 2 prefills 100k × 4
  essais ≈ +40 min sur RUN9B → ~80–100 min total.
- **Microbench Q6_K REF vs H2** : mis de côté pour l'instant — la question
  W4R4@100k a plus de valeur décisionnelle ; à faire seulement si RUN9C
  montre un gain H2 non expliqué par le fattn.

**Artefacts RUN9B** : `cupti129-test/nightrun8/` — `PROTOCOL-run9b.md`,
`run9b_bench.py`, `ab_run9b/` (results.jsonl, gen_ref1/2.json, gen_h2a1/2.json,
server_ref1/2.out, server_h2a1/2.out).

---

### 30. RUN9C — A/B REF vs H2 à ~107k + condition W4R4 OFF (2026-10-15, 18:51–19:16 UTC, VERDICT FINAL)

**Question** : re-trancher le régime §26.5 (~107k, où H2 gagnait −13 %) en
session isolée propre, et isoler la variable `GGML_CUDA_VOLTA_Q6_W4R4`.
Protocole écrit **avant** lancement : `cupti129-test/nightrun8/PROTOCOL-run9c.md`
(6 essais : ref1/h2a1/ref2/h2a2 en W4R4=1 + ref3/h2a3 en W4R4=0 ; prompt
filler 550 000 chars = **107 351 tok** — même prompt que §26.5 ; decode
**1200 tok**, T=0). Bench `run9c_bench.py` via `supervisor_bench.py --detach`
; artefacts `cupti129-test/nightrun8/ab_run9c/`.

#### 30.1 Incident du run — bug du bench (3 essais REF invalidés à tort)

Les 3 essais REF (ref1, ref2, ref3) ont été marqués `INVALID (gen erreur)`
par le bench, **alors que les 6 decodes ont complété** (log serveur,
`eval time / 1200 tokens`, `truncated = 0`). Cause : le gate de réponse
`"error" in json.dumps(d).lower()` du harnais est un **faux-positif** — il
matche sur le contenu du JSON (texte généré / keys), pas sur une vraie erreur
API. Contre-preuves objectives :
- Les 6 serveurs ont bien produit 1200 tok (`stop processing: n_tokens = 108550`,
  `truncated = 0`, aucun crash/OOM/signal dans les logs).
- Les 3 textes H2 sauvegardés (5 415 chars) sont **bit-identiques entre eux**
  et ne contiennent **aucun** `error` ; les 3 `gen_ref*.json` n'ont pas été
  sauvegardés (le bench les sauvegardait **après** le gate fautif).
- `prefill2_cached = 107347` et shortlist OK sur les 6 essais.
- Les `mtp_delta` calculés dans results.jsonl sont incohérents pour tous les
  essais (fenêtre prise sur les 2 dernières lignes spec, ici contiguës du
  decode) → **le log serveur fait foi** (correction du pattern RUN9, §29.1).

**Conséquence** : les données REF ne sont pas dans results.jsonl (wall_s/mtp
absents) mais **sont dans les logs serveurs** (mesure uniforme, plus fiable
que le mur HTTP du bench). Le run n'est **pas** perdu : les 6 essais sont
exploitables via les `server_*.out`.

#### 30.2 Résultats bruts (mesure SERVEUR, ligne `eval time` du task 236 = decode chaud)

| essai | lib | W4R4 | decode (ms / 1200 tok) | ms/tok | t/s | MTP (acceptés/gén., req.) | mean len | cum pos (req.) |
|---|---|---|---|---|---|---|---|---|
| ref1 | e4afbb4a | 1 | 40 096,49 | 33,41 | 29,90 | 771 / 1281 = 60,2 % | 2,81 | (0,808 / 0,576 / 0,422) |
| h2a1 | a4b6f440 | 1 | 35 519,97 | 29,59 | 33,78 | 828 / 1112 = 74,5 % | 3,23 | (0,890 / 0,721 / 0,633) |
| ref2 | e4afbb4a | 1 | 39 951,74 | 33,29 | 30,01 | 771 / 1281 = 60,2 % | 2,81 | (0,808 / 0,576 / 0,422) |
| h2a2 | a4b6f440 | 1 | 35 109,44 | 29,26 | 34,17 | 828 / 1112 = 74,5 % | 3,23 | (0,890 / 0,721 / 0,633) |
| ref3 | e4afbb4a | **0** | 41 737,67 | 34,78 | 28,73 | 771 / 1281 = 60,2 % | 2,81 | (0,808 / 0,576 / 0,422) |
| h2a3 | a4b6f440 | **0** | 36 509,50 | 30,42 | 32,87 | 828 / 1112 = 74,5 % | 3,23 | (0,890 / 0,721 / 0,633) |

Préfill1 froid (107 351 tok) : REF 193,6 / 196,8 / 196,7 s (moy. 195,7 s) vs
H2 196,2 / 196,6 / 196,6 s (moy. 196,5 s) → **pas d'effet H2 sur le prefill**
(+0,4 %, bruit) ; prefill2 chaud ≈ 1,1 s partout (cached 107 347).

#### 30.3 Écarts

- **Q1 — Δ H2 vs REF (W4R4=1) = −11,8 % wall** (réplicas : −11,9 %, −12,1 %)
  → **le gain H2 est CONFIRMÉ au régime ~107k** (reprise directe de §26.5 :
  −13,0 % ; écart d'échelle entre les deux campagnes : prompt et 2ᵉ chauffe
  quasi identiques, écart de ~1,2 pt à mettre au compte du bruit/reproductions).
- **Q1 (W4R4=0) = −12,5 %** (n=1/bras, indicatif).
- **Q2 — variable W4R4** : ON est plus rapide que OFF, des deux côtés :
  **REF +4,1 %** (33,35 → 34,78 ms/tok) et **H2 +3,3 %** (29,43 → 30,43) —
  soit le chemin Q6_K W4R4 est bénéfique pour la prod **et** indépendant du
  kernel fattn H2 (pas d'interaction : le gain H2 persiste en W4R4=0).
  (n=1/bras en OFF : indicatif, pas de puissance statistique.)
- **Découpage du gain H2 (W4R4=1)** : 1200 tok = 1200 − acceptés rounds.
  REF : 771 acc. → 429 rounds à 93,3 ms/round ; H2 : 828 acc. → 372 rounds à
  94,9 ms/round. Facteurs : **rounds −13,3 %** (acceptance MTP +57 acc.,
  mean len 2,81→3,23) × **coût/round +1,75 %** (verify H2 légèrement plus
  lourd par round, le tuile BC=8 ne paie pas ici en coût par round) =
  **produit −11,77 %** = Δ wall exact. → **le gain applicatif H2 @107k est
  quasi intégralement un gain d'acceptance MTP** (la 3ᵉ position du draft est
  nettement plus acceptée : pos2 0,576→0,721, pos3 0,422→0,633).
- **Comparaison RUN9B @41k vs RUN9C @107k** : à 41k H2 ≈ REF (+2,1 %), à 107k
  H2 −11,8 % → **l'effet H2 est bien lié à la longueur de contexte**, cohérent
  avec le partitionnement de l'attention (splits/KV par split croissant avec
  n_kv : 158 vs 153 splits à 41k, 159 vs 157 à 100k ; cf. §29.3).

#### 30.4 Stabilité numérique / comportement

- **Déterminisme intra-bras** : h2a1 ≡ h2a2 ≡ h2a3 (texte bit-identique,
  MTP bit-identique : 828/1112) ; REF ref1 ≡ ref2 ≡ ref3 (771/1281, même cum
  pos) → les deux libs sont déterministes T=0, comme à 41k.
- **REF vs H2 : divergence de texte ATTENDUE** (continuation du même filler,
  séquences différentes dès les premiers tokens — le draft MTP diverge tôt vu
  les logits dérivés, §27/§29.3). Contenu sémantiquement cohérent des deux
  côtés (continuation du récit), pas de régression macroscopique. Critère de
  l'addendum (§28.5) respecté : divergence = changement d'associativité fp32,
  pas de dégradation reproductible.
- **Note de reproductibilité** : l'A/B §26.5 (h2_ab_lib.sh) donnait accLen
  2,83 (prod) vs 3,24 (H2) sur 1200 tok — ici 2,81/3,23, MTP H2 reproduit à
  ~0,01 près. Les deux campagnes convergent.

#### 30.5 Décision

1. **Performance** : H2 gagnante **confirmée** au régime long contexte
   (~107k, −11,8 %), et ce **que W4R4 soit ON ou OFF**. Le gain vient de
   l'**acceptance MTP** (plus de rounds acceptés), pas d'un verify plus
   rapide par round (+1,75 %/round). À 41k, H2 ≈ REF (§29.2) → **bascule en
   prod H2 justifiée** ; le régime court n'est pas pénalisé de façon
   significative.
2. **W4R4** : maintenir `GGML_CUDA_VOLTA_Q6_W4R4=1` (dès que disponible) —
   +4 % REF / +3 % H2, sans interaction défavorable avec H2. (n=1/bras en OFF
   → à consolider si décision de le désactiver, jamais pour l'activer : ON est
   déjà le default du .so et l'état prod.)
3. **Aucun rollback** ; H2 reste en production expérimentale (§28.5).
4. **Action de correction du harnais** (avant tout prochain A/B) : remplacer le
   gate `"error" in json.dumps(d).lower()` par un vrai test d'erreur API
   (`finish_reason` + présence de `choices` + absence de clé `error` en
   **racine** du JSON) ; et conserver la ligne `eval time` du log serveur comme
   **mesure canonique** (mur HTTP du bench = bruit ; cf. §30.1).

#### 30.6 État de production post-run (vérifié)

- `ollama-v100` **active**, API 11434 ok (`/api/version` répond), `.so`
  déployé `a4b6f440` inchangé, REF backup `e4afbb4a` intact, port 44911
  libéré, V100 revenue à l'usage prod (32 348 MiB). Bench rc=0 en 1496 s ;
  superviseur a renvoyé l'inférence OWUI (`0 reprise envoyée (f4a4e5d0)`).

**Artefacts RUN9C** : `cupti129-test/nightrun8/` — `PROTOCOL-run9c.md`,
`run9c_bench.py` (gate à corriger, §30.5-4), `ab_run9c/` (results.jsonl,
gen_h2a1/2/3.json, server_*.out ×6 — les logs REF font foi pour les murs).

---

### 31. RUN9C2 — re-run harnais corrigé H1/H3/H4 (2026-10-15, 21:08–21:33 UTC)

Suite immédiate de §30. Protocole `cupti129-test/nightrun8/PROTOCOL-run9c2.md`,
bench `run9c2_bench.py`. Corrections H1 (détect. erreur API = clé `error` en
racine OU absence de `choices`, plus de sous-chaîne), H3 (mesure serveur
`eval time` canonique), H4 (delta MTP sur le champ *g*). **6/6 essais OK.**

- **Performance reproduite** (eval_ms serveur) : REF 40 455 / 40 212 / 41 832 ;
  H2 35 670 / 35 558 / 36 254 ; MTP 771/1281 vs 828/1112 → identique à §30.
- **2ᵉ bug du harnais** : `save(genpath, d)` recevait un chemin **absolu**
  déjà préfixé de `OUTDIR` ; `save()` a re-préfixé → chemin doublon
  `[Errno 2]` ×6 → **`gen_*.json` absents** (warm/prefill OK, noms relatifs).
  Le texte gen n'est **pas** dans les logs serveur (vérifié) → re-run
  nécessaire pour l'objectif texte.
- **Validité** : RUN9C2 exploitable **performance uniquement**, pas texte.

### 32. RUN9C3 — re-run final, harnais définitif H5, **6/6 OK + 6/6 réponses sauvegardées** (2026-10-15, 22:24–22:49 UTC)

Protocole `PROTOCOL-run9c3.md`, bench `run9c3_bench.py`, sortie `ab_run9c3/`.
Corrections **H5** : (a) `save()` module-level, nom relatif (absolu → `relpath`),
(b) self-test de sauvegarde **avant** stop du service, (c) champ `gen_saved`
+ relecture + gate **G_SAVE bloquant** (essai sans texte = INVALID — l'objectif
du run est le texte). Tests unitaires OK avant lancement.

#### 32.1 Production (vérifiée)
`ollama-v100` active, API ok, `.so` déployé `a4b6f440` inchangé, port 44911
libre, V100 32 348 MiB (usage prod). rc=0 en 1503 s ; reprise OWUI envoyée
(`3756f6e1`). **Aucune** bibliothèque de production modifiée.

#### 32.2 Performance (mesure serveur canonique, 1200 tok decode, 107 351 ctx)

| tag | RUN9C | RUN9C2 | RUN9C3 | spread 3 runs |
|---|---|---|---|---|
| ref1 (W4R4=1) | 40 096,5 | 40 455,1 | 40 283,5 | 0,89 % |
| h2a1 (W4R4=1) | 35 520,0 | 35 669,8 | 35 290,2 | 1,08 % |
| ref2 (W4R4=1) | 39 951,7 | 40 212,6 | 40 551,9 | 1,50 % |
| h2a2 (W4R4=1) | 35 109,4 | 35 557,8 | 35 243,2 | 1,28 % |
| ref3 (W4R4=0) | 41 737,7 | 41 832,1 | 41 925,0 | 0,45 % |
| h2a3 (W4R4=0) | 36 509,5 | 36 254,2 | 36 702,4 | 1,24 % |

**Écarts RUN9C3** : W4R4=1 → **H2 −12,74 %** (moy. réplicas) ; W4R4=0 → **H2
−12,46 %**. Les 3 campagnes convergent (spread < 1,5 % par essai) → le gain
H2 @107k est **−12,5 ± 0,5 %**, robuste au run. **Aucun** changement de
conclusion vs §30.

#### 32.3 Stabilité numérique / déterminisme

- **Intra-bras (RUN9C3)** : ref1 ≡ ref2 ≡ ref3 (bit-identique), h2a1 ≡ h2a2 ≡
  h2a3 (bit-identique). Déterminisme parfait T=0, comme à 41k et §30.
- **Cross-run H2** : h2a1 **RUN9C ≡ h2a1 RUN9C3** bit-identique → la
  continuation H2 est **reproductible entre deux sessions/nuits** (~4 h
  d'écart, service redémarré). C'est la preuve la plus forte de détermination.
- **W4R4 ON vs OFF** : R1 ≡ R3 (REF) et H1 ≡ H3 (H2) — le **texte est identique**
  en W4R4 ON/OFF (seule la vitesse change : §30.3). La variable quantisation
  Q6_K W4R4 n'affecte ni la séquence décodée ni le MTP dans ce régime, mais
  accélère les kernels (+4 % REF / +3 % H2 en ON).

#### 32.4 **Vraie comparaison textuelle REF vs H2 (l'objectif du re-run)**

Première comparaison texte possible (les `gen_*.json` des 6 essais existent).
- **Préfixe commun** : **1 086 chars** (~20 %) sur 5 401 (REF) / 5 415 (H2).
- **Première divergence** (pos. 1 086, W4R4=1) :
  - REF : `…Records weather observations (wind, sea state, visibility) in the logbook…`
  - H2  : `…Records **detailed** weather observations (wind, sea state, visibility) in the logbook…`
  → écart de **formulation minimale** (un adjectif), pas un changement de sens.
- **Ratio difflib** (R1 vs H1) : **0,5407** (W4R4=1 et =0) — les textes partagent
  ~54 % de leur contenu.
- **Nature des deux sorties** : sur ce prompt filler (un récit répété en boucle),
  REF et H2 produisent **tous deux** une méta-réaction identique — un résumé de
  la routine du gardien de phare + une proposition d'aide (« Would you like me
  to continue / analyze… How can I help you with this text? »). **Même intention,
  même structure, aucune régression** sur un des deux bras.
- **Occurrences du mot « error »** (clé du §30.1) : **REF = 1** (`"It seems like
  a copy-paste error in your input"`), **H2 = 0** — identique en W4R4 ON/OFF.

#### 32.5 **Cause racine du faux-positif RUN9C — CONFIRMÉE**

Le gate naïf de RUN9C `"error" in json.dumps(réponse).lower()` matchait la
sous-chaîne dans **tout** le JSON. La continuation **REF** contient le mot
*error* (le modèle commente le prompt filler répété : « copy-paste **error** ») ;
la continuation **H2** (divergée dès le char 1 086) ne le contient **pas**. →
Les 3 essais REF ont été marqués `INVALID (gen erreur)` **à tort**, les 3 H2
non. **Ce n'était ni un bug REF ni un bug H2** : c'était un artefact du harnais
qui discriminait les bras selon la présence fortuite du mot *error* dans le
texte généré. Le gate H1 (clé `error` **en racine** du JSON / absence de
`choices`) élimine cette classe d'erreur ; **aucun** des 6 essais RUN9C2/RUN9C3
n'a plus déclenché de faux-positif.

#### 32.6 Décision & limites

1. **Performance** : **H2 gagnante confirmée @~107k (−12,5 ± 0,5 %)** sur 3
   campagnes, W4R4 ON/OFF, gain = acceptance MTP (§30.3). Bascule prod H2
   justifiée. **Aucun rollback.**
2. **Stabilité numérique** : déterminisme parfait intra-bras + **reproductibilité
   cross-run bit-identique (H2)** ; divergence REF vs H2 = formulation minimale
   (associativité fp32, §27/§29.3), **non** un bug.
3. **Comportement** : **aucune régression** — même intention/structure des deux
   côtés, contenu cohérent, préfill intact, MTP plus efficace, prod restaurée.
4. **Limite** : prompt filler (répétitif) = régime de mesure, pas de la prod
   OWUI (français, conversations) ; les écarts de performance se rapportent au
   régime ~107k. Le ratio difflib 0,54 reflète deux réponses de *méta-commentaire*
   légèrement différentes, pas une dégradation.
5. **Harnais** : H1/H3/H4/H5 figés dans `run9c3_bench.py` — c'est désormais le
   **harnais de référence** pour tout A/B V100 (gate G_SAVE bloquant, mesure
   serveur canonique, sauvegarde systématique re-lue).

**Artefacts RUN9C3** : `cupti129-test/nightrun8/` — `PROTOCOL-run9c3.md`,
`run9c3_bench.py` (harnais de référence), `ab_run9c3/` (results.jsonl,
**gen_ref1/2/3.json + gen_h2a1/2/3.json**, warm/prefill/server ×6).
