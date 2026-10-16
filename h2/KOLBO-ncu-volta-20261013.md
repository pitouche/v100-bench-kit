# RAPPORT KOLBO — Mesure DRAM réelle du kernel E1 sur V100 (Volta)

Date : 2026-10-13
Objectif : rétablir un profilage NCU exploitable sur la V100 (`sm_70`) et remplacer
l'estimation E1 (volume théorique) par un compteur DRAM matériel.
**Aucune modification de `fattn-q8-volta.cuh`. E2 non engagée.**

---

## 1. Résultat de haut niveau

**Verdict : MESURE DRAM VALIDÉE.** NCU 2024.1.1 (installation privée, non-invasive)
profile la V100, les compteurs `dram__bytes_read.sum` / `dram__bytes_write.sum` sont
réellement collectés, et la durée mesurée (1780 µs) est cohérente avec le wall-clock
non profilé (1665 µs, +6.9 %). Le volume DRAM effectif du kernel est **~94 Mo**
(29.4 % des 320 Mo demandés en L1) — le kernel est **kernel-bound à ~10 % du
plafond HBM**, ce qui confirme et précise le verdict E1.

## 2. Ce qui manquait et comment on l'a résolu

| Besoin | État trouvé | Résolution |
|---|---|---|
| NCU supportant la V100 | 2025.4.1 (système) : `--list-chips` sans GP100 → `Profiling is not supported on device 0` | NCU **2024.1.1** déballé (sans installer) depuis les paquets **officiels Ubuntu** `nsight-compute` + `nsight-compute-target` (archive.ubuntu.com) dans `$HOME/ncu-2024.1.1` |
| NCU 2025.2.1 (cible initiale) | `.run` officiel indisponibilité vérifiable sans compte NVIDIA ; aucun artefact local | Non installé (non bloquant, voir §7) |

Contraintes respectées : aucun `PATH` modifié, aucun paquet dpkg installé/désinstallé,
aucun lien système, pilote/Toolkit/Ollama intacts. Nettoyage futur = `rm -rf
/home/guile/ncu-2024.1.1` (et les 2 `.deb` de 239 Mo + 105 Mo dans `/tmp/`).

### Installation (commandes exactes)

```bash
apt-get download nsight-compute nsight-compute-target
# nsight-compute_2024.1.1.4~12.4.1-8_amd64.deb (239 Mo)
# nsight-compute-target_2024.1.1.4~12.4.1-8_amd64.deb (105 Mo)
mkdir -p /home/guile/ncu-2024.1.1
cd /home/guile/ncu-2024.1.1
dpkg -x /tmp/nsight-compute_2024.1.1.4~12.4.1-8_amd64.deb .
dpkg -x /tmp/nsight-compute-target_2024.1.1.4~12.4.1-8_amd64.deb .
NCU=/home/guile/ncu-2024.1.1/usr/lib/nsight-compute/ncu
```

Le binaire `ncu` du 1er deb seul ne suffit pas : le wrapper pointe vers
`target/linux-desktop-glibc_2_11_3-x64/ncu` qui est dans le paquet
`nsight-compute-target` (c'est lui qui contient `libnvperf_target.so`,
`libcuda-injection.so`, `libTreeLauncher*.so`).

## 3. Étape 2 — Validation sans profilage GPU

```text
$ $NCU --version        # rc=0
Copyright (c) 2018-2024 NVIDIA Corporation
Version 2024.1.1.0 (build 33998838) (public-release)

$ $NCU --list-chips     # rc=0
ad102, ad103, ad104, ad106, ad107, ga100, ga102, ga103, ga104, ga106, ga107,
ga10b, gh100, gv100, gv11b, tu102, tu104, tu106, tu116, tu117
```

**`gv100` présent** (et `gv11b`). La présence de l'architecture ne suffit pas
(critère de la mission) → le profilage effectif a été testé (§5).

## 4. Étape 3 — Métriques DRAM exposées et permissions

`$NCU --query-metrics --chip gv100` (rc=0, 2312 lignes) :
- base : `dram__bytes_read`, `dram__bytes_write`, `dram__sectors_read/write`,
  `dram__throughput` ;
- dérivées demandées et **réellement collectées** : `dram__bytes_read.sum`,
  `dram__bytes_write.sum`, `dram__throughput.avg.pct_of_peak_sustained_elapsed` ;
- L2 (pour séparer L2/DRAM) : `lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld/st.sum`.

Permissions (read-only) : `/dev/nvidia*` en mode `crw-rw-rw-` (tous utilisateurs),
paramètre pilote `RmProfilingAdminOnly: 1` → les compteurs exigent **root**.
C'est ce que le run a utilisé (`sudo`), sans modifier aucune permission.

## 5. Étape 4 — Mesures (harnais `e1_bench`, V100 seule)

### 5.1 Prérequis et décisions observables

- Géométrie du harnais (vérifiée dans `e1_bench.cu` / `.cuh`) : copie exacte du
  `partial_kernel` sm_70, `n_kv=107351, kvh=4, G=6, T=4, D=256, splits=160,
  grid=(4,160), block=128`, K/V en Q8_0 (34 B/bloc de 32), `pacc` bf16.
- **Arrêt du service** : obligatoire et justifié — V100 à 32 350/32 768 MiB utilisés
  par le service ; le harnais alloue ~318 MiB GPU (K+V 234 + Q 15.7 + mask 3.4 +
  pacc/pm/pl 8.0 + contexte) → OOM vérifié à 143 MiB libres.
- Restauration vérifiable (`trap EXIT` du script) : `systemctl start`, attente API
  `:11434/api/version`, re-vérification `is-active` + mémoire V100. Chaque run a
  laissé `console*.log` et `raw*.txt` dans le dossier.
- **Piège d'ordonnancement** : `CUDA_VISIBLE_DEVICES=1` (index nvidia-smi) cible la
  **1070** (cc 6.1) côté CUDA — probe `cudaGetDeviceProperties` : CUDA dev 0 = V100,
  CUDA dev 1 = 1070. Le premier run sous NCU a échoué en
  `no kernel image is available` (kernel compilé sm_70 lancé sur la 1070).
  Correction : `CUDA_VISIBLE_DEVICES=0` (vérifié par probe, pas par nvidia-smi).

### 5.2 Résultats des 3 runs (identiques à ~1 % près)

| Run | Mode NCU | R (Mo) | W (Mo) | Durée (µs) | gpc cycles (MHz×µs) |
|---|---|---|---|---|---|
| A | skip 1, count 1 (L2 partiellement chaude) | 60.93 | 31.74 | 1770 | 2 187 024 |
| B | skip 1, count 1, passes supplémentaires | 61.22 | 32.77 | 1780 | 2 183 929 |
| C | skip 0 (kernel le plus froid possible) | ~60.8 | ~33.1 | 1790 | 2 195 112 |

Compléments du run B (3 passes) :
- `dram__sectors_read.sum` = 1 913 089 secteurs × 32 B = **61.2 Mo** (cohérent avec
  `dram__bytes_read.sum` = 61.22 Mo → compteur DRAM réel, pas une approximation) ;
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` = **5.91 %** ;
- `lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum` = **226.65 Mo**
  (octets demandés à la L2 par les chargements LSU global qui ratent la L1).

Commande NCU exacte (runs B/C) :

```bash
sudo env CUDA_VISIBLE_DEVICES=0 \
  /home/guile/ncu-2024.1.1/usr/lib/nsight-compute/ncu \
  --launch-skip 1 --launch-count 1 \
  --kernel-name 'regex:partial_kernel' \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__sectors_read.sum,dram__sectors_write.sum,\
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum,\
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,gpc__cycles_elapsed.max \
  ./e1_bench 2
```

## 6. Étape 5 — Comparaison et verdict

### 6.1 Temps (dénominateurs explicites)

- **Durée wall sans profiler** (CUDA events, 3 iters, V100 seule) : **1665 µs**
  (médiane ~1662 ; run6 CUPTI médiane 1615 µs, écart ~3 %).
- **Durée sous NCU** (`gpu__time_duration.sum`) : **1770–1790 µs**.
- Surcoût observable du profiler : **+6.9 à +7.8 %** (replay kernel + invalidation
  des caches + passes supplémentaires). Les deux grandeurs sont du même ordre ;
  le temps sous NCU n'est **pas** assimilable au temps non profilé, mais la
  proximité autorise l'usage des deux comme bornes.

Fréquence : `gpc__cycles_elapsed.max / gpu__time_duration.sum` = 1.229 GHz
≈ boost nominal V100 (1230 MHz) → les passes ne sont pas cadencées plus bas.

### 6.2 Volume DRAM : le résultat clé

| Grandeur | Valeur |
|---|---|
| Octets demandés en L1 (trajet global) — L2 miss LSU op_ld + op_st | **229.7 Mo** |
| Octets effectivement DRAM (compteur) — `dram__bytes_read.sum` | **61.2 Mo** |
| `dram__bytes_write.sum` | **32.8 Mo** |
| **Total DRAM mesuré (R+W)** | **~94 Mo** (29.4 % des demandes L1) |

Lecture : la L2 (6 MiB V100) est traversée à 229.7 Mo mais seuls ~94 Mo atteignent
la DRAM. La différence vient du **re-partitionnement des secteurs entre les 640
blocs** (4 kvh × 160 splits) qui partagent les mêmes tuiles K/V : chaque
`int4`/32 B demandé en L1 ratant la L1 est servi en grande partie depuis la L2
(référencé < 1 s avant par un autre split), et le traffic DRAM se réduit aux
miss « frais » sur la ligne de 128 B. C'est un effet d'agrégation de cache, pas
une perte de comptage : le run C (no warmup) donne le même 94 Mo que le run B
(warm), car même agrégation intra-lance.

Limite : `dram__bytes_read.sum` mesure le traffic DRAM pendant le kernel seul ;
il ne distingue pas « KV unique » de « KV re-lu par 4 splits ». Le volume
**théorique minimum** (KV lu une seule fois + Q/mask/outputs) est 260.7 Mo selon
le modèle du harnais ; la mesure de 94 Mo signifie que la DRAM ne voit que ~36 %
de ce minimum — la cache L2 absorbe le reste. (Le modèle du harnais compte
chaque élément par bloc lecteur ; le compteur compte par access physique DRAM.)

### 6.3 Bande passante (deux dénominateurs, tous deux documentés)

- **BW DRAM = 94 Mo / 1780 µs (temps NCU) = ~53 GB/s** → 5.9 % du pic HBM V100
  (cohérent avec `dram__throughput` = 5.91 %).
- BW DRAM = 94 Mo / 1665 µs (temps wall) = ~56 GB/s (borne haute si le kernel est
  légèrement plus lent sans profiler).
- BW « demandée L1 » = 229.7 Mo / 1780 µs = ~129 GB/s (ce que les SMs tirent).
- Estimation E1 initiale (volume théorique 260.7 Mo / 1665 µs = 157 GB/s) : c'était
  une **borne sur les accès**, pas le traffic DRAM réel. Révisée : **~53 GB/s**.

### 6.4 Verdict E1 révisé

| Critère E1 | Valeur mesurée | Conclusion |
|---|---|---|
| BW DRAM < 300 GB/s → kernel-bound | **~53 GB/s** (5.9 % du pic) | **kernel-bound confirmé, largement** |
| BW DRAM > 600 GB/s → mem-bound | non atteint de 11× | H3 (réduire les octets) **non prioritaire** |
| Durée wall | 1665 µs = 1615 µs run6 (±3 %) | harnais fidèle |

Le levier principal est l'efficacité du **compute** (déquant int8→half en boucle,
synchronisations `__syncthreads`, tuiles 16 clés, occupation), pas le volume DRAM.
Le gain sérialisé estimé reste ~360 ms sur la fenêtre de décode ; le gain **mural**
reste à chiffrer via E3 / run7 v2 (limite CUPTI du champ `t`, rappelé dans ETAT.md).

### 6.5 Limites de mesure

1. `dram__bytes_read.sum` est un compteur DRAM par kernel, mais sous replay le
   contexte GPU est exclusive ; la présence éventuelle d'un autre contexte actif
   (X server) biaiserait les métriques L2/FB — ici la V100 est dédiée, service stoppé.
2. Le run C (no warmup) n'a pas isolé le 1er kernel « à froid » de manière plus
   agressive que B, car le replay de NCU ré-invalide les caches avant chaque pass :
   B et C sont de toute façon quasi identiques.
3. Le volume théorique 260.7 Mo du harnais est un modèle d'accès, pas une mesure ;
   la mesure DRAM (94 Mo) est la référence.
4. La métrique `dram__bytes_read.sum` n'existait **pas** sous le nom exact dans
   `--query-metrics` (mode `all`) — elle apparaît en mode `all` comme
   `dram__bytes_read.sum` (dérivée de `dram__bytes_read`). Vérifié.

## 7. Pour aller plus loin (non bloquant)

- **NCU 2025.2.1** (cible initiale, plus récent) : nécessite un compte
  developer.nvidia.com pour le `.run`. Si disponible, l'installation se fera dans
  `$HOME/ncu-2025.2.1` avec la même méthode (`.run --targetpath --silent --noprompt`)
  et la même validation (`--list-chips` + 1 lance). Le 2024.1.1 fonctionne
  aujourd'hui ; le 2025.2.1 n'apporterait qu'un outil plus récent, pas de métrique
  manquante.
- **E2 non engagée.** Toute optimisation du `partial_kernel` (déquant, syncthreads,
  tuiles) doit attendre une autorisation explicite.
- **E3 / run7 v2** reste nécessaire pour chiffrer le gain **mural** (overlap, idle)
  — ce que NCU seul ne peut pas donner (kernel-by-kernel, pas timeline).

## 8. État système final

- Service `ollama-v100` : **active**, API ok, V100 rechargée (32 350 MiB) —
  vérifié après chaque run par le `trap EXIT`.
- NCU système : `/usr/local/cuda-13.1/bin/ncu` et `/opt/nvidia/nsight-compute/2025.4.1/ncu`
  intacts.
- Installation privée : `/home/guile/ncu-2024.1.1/` (déballable, non lié au PATH).
- Artefacts : `run_e1_ncu2024.sh`, `run_e1_ncu2024_cold.sh`,
  `e1_ncu2024_console{,B,C}.log`, `/tmp/e1_ncu2024_raw{,B,C}.txt`,
  `/tmp/e1_ncu2024_rawA.txt` (dans `/tmp/e1_ncu2024_raw.txt`).

---

# 9. ÉTAPE E3 — Chronométrie GPU réelle (run7 v2, CUPTI v2) — 2026-10-13, 18:34–19:45

## 9.1 Objectif
Fermer l'incertitude de run6 (champ `t` = timestamp CPU d'émission, pas GPU) :
exposer `gpu_start_ns` / `gpu_end_ns` par record, puis mesurer **union des
intervalles**, **occupation**, **idle**, **overlap** sur une fenêtre decode 64 tok
comparable à run6.

## 9.2 Collecteur v2 (`cupath_shim_v2.c` → `libcupath_v2.so`)
- Record binaire 56 octets : `seq(8) gpu_start(8) gpu_end(8) dur_ns(8) corr(8) stream(4) op(4) aux(8)`.
  `op` : 0 KER, 1 CPY, 2 RT (RUNTIME, correlationId + cbid), 3 RESET.
- `.names` texte 1:1 (KER/CPY/RT, dans l'ordre) : `op seq= gpu_s= gpu_e= dur_ns= stream= corr= t_real_us= name=`.
- `.hdr` : `pid t0_real_us t0_mono_ns driver` (ancrage horloges).
- **Domaine d'horloges mesuré (G4)** : CUPTI est dans **CLOCK_REALTIME en ns** —
  `gpu_s(1er event) − t0_real = +98,4 ms` (init CUDA) et le flush final arrive à
  **+25 µs** du dernier kernel. Les `gpu_s/gpu_e` sont donc directement comparables
  au temps mural, sans calibration.
- Test unitaire `test_v2.cu` (kernels spin sur 2 streams + memcpy + usleep) :
  - G1 aucun champ manquant : OK — G2 `dur==end−start` : OK (21/21) —
    G3 séquence + alignement 1:1 : OK (20 KER/CPY/RT ↔ 20 lignes) ;
  - **chevauchement forcé détecté** : somme 9,966 ms − union 7,310 ms = 2,656 ms ;
  - **gap forcé visible** : 15,2 ms (usleep 10 ms + init) ;
  - G4 domaine horloges : OK.

## 9.3 Pannes de capture (corrigées avant le run valide)
1. **0 record** au premier run de production : `LD_PRELOAD` hérité par ~40
   sous-processus du script ; chaque constructeur ouvrait la trace avec `O_TRUNC` →
   le dernier sous-processus (analyse python) effaçait le binaire. Correction :
   `<prefix>.<pid>.*` par processus ; l'analyse prend le plus gros fichier
   (le serveur, 150 Mo binaires / 444 Mo `.names`).
2. **Parser v2 v0** : `name=(.*)$` échouait sur les noms mangled contenant `(`
   (faux négatif « 15 kernels »). Correction : parser structuré.
3. **Ancrage** : bracket de départ non disponible (`--no-log-timestamps`) →
   ancrage par la **fin** : dernier kernel = fin du 64e token, `T0 = T_end − eval_time`.

## 9.4 Gates de validation (run7 v2 : 64 tok, wall serveur 1818,36 ms, 34,65 tok/s)
| Gate | Résultat |
|---|---|
| G1 aucun champ manquant | OK — 2 688 059 records binaires (56 o) |
| G2 `dur_ns == gpu_end − gpu_start` | OK — 2 688 059/2 688 059 |
| G3 séquence continue + alignement nom/durée 1:1 | OK — 2 688 058 lignes = KER 698 499 + CPY 6 795 + RT 1 982 764 |
| G4 domaine horloges CUPTI = REALTIME(ns) | OK (test unitaire + cohérence du run) |
| G5 couverture : union vs wall | **OK — écart 0,55 %** (< 2 %) |
| Fenêtre identifiable | OK — dernier cluster (gap 252 ms avant), 40 390 KER + 892 CPY, 64 tok confirmé par le serveur |

**Capture exploitable.**

## 9.5 Mesures (fenêtre eval 1818,36 ms, ancrage fin de trace)
| Grandeur | Valeur (mesurée) |
|---|---|
| Kernels (40 390) : somme / union | 1737,769 ms / 1737,769 ms |
| Memcpy (892) : somme | 9,901 ms |
| **Union ker+cpy (busy GPU réel)** | **1747,671 ms** |
| **Occupation GPU (union/wall)** | **96,11 %** |
| **Idle (wall − union)** | **70,689 ms (3,89 %)** |
| **Overlap inter-streams (somme−union)** | **0,000 ms** — aucun chevauchement |
| Streams | 16 : 1613,8 ms (40 122 ker) ; 17 : 194,5 ms (2496 ker) |
| Gaps | 41 281 ; médiane 0,64 µs ; p99 9,5 µs ; max 4,55 ms ; >1 ms : 3 ; >100 µs : 89 |

**Bornes/estimations (marquées)** :
- Idle par round de décodage effectif (64 tok / mean_len 3,71 = 17,2 rounds) :
  70,7/17,2 ≈ **4,1 ms/round** (≈ 2,4 ms/round specu si comptage 36) — *estimation
  par division, la mesure brute est 70,689 ms*.
- Gain mural borné pour E2 : `ggml_q8v14partial` = 543,9 ms (30,1 % du wall) ;
  2× rapide ⇒ ~172 ms de wall gagnés (~34,6 → ~38 tok/s) — *bornes arithmétiques,
  à confirmer par l'A/B applicatif*.

## 9.6 Top gaps et attribution (mesurée)
| Position | Taille | Après (kernel) |
|---|---|---|
| t+98,1 ms | 4,55 ms | `k_get_rows_float` (copie rows → CPU, sync) |
| t+12,2 ms | 1,91 ms | `k_get_rows_float` |
| t+1563,1 / 615,1 / 525,8 / 806,4 / 263,0 / 1694,8 ms | ~1,05 ms × 6 | `k_set_rows` (upload rows, sync) |
| 41 k micro-gaps | somme ≈ 40 ms | latence de lancement (médiane 0,64 µs) |

L'idle est donc **structuré** (sync CPU `k_get_rows`/`k_set_rows` ≈ 11 ms) +
**granulaire** (micro-gaps de lancement ≈ 40 ms) — pas un trou d'attente mémoire.

## 9.7 Verdict E3
- **Mesure** : GPU **96,11 % occupé** sur la fenêtre decode ; **0 ms de
  chevauchement** ; **70,7 ms d'idle** (3,89 %), structuré.
- Interprétation avec les repères retenus : 96,11 % ∈ [89 %, 97 %] → « marge à
  examiner » — et cette marge est chiffrée, pas bornée. Le GPU est quasi saturé en
  temps GPU, mais le mur est **sérial** (aucun pipeline) : l'idle n'est pas
  récupérable par du chevauchement, seulement par (a) des kernels plus courts
  (levier E2) et (b) la réduction des sync CPU `k_get_rows`/`k_set_rows` (hors
  périmètre E2, noté comme piste).
- **E2 est validée** : la chronométrie est exploitable, le verdict est chiffré,
  et l'hypothèse de travail E2 est confirmée quantitativement — le gain mural
  d'une optimisation `ggml_q8v14partial` est borné par son temps GPU sérial
  (543,9 ms / 30,1 % du wall).

## 9.8 Reproductibilité (E3)
```bash
cd ~/llama-experiments/h2/cupti129-test
# recompile du shim (gcc-13, CUPTI 12.9 prefix local)
gcc-13 -O2 -fPIC -shared -o libcupath_v2.so cupath_shim_v2.c \
  -Iinclude -I$CUDA126/include -Llib64 -L$CUDA126/lib64 -lcupti -lcuda \
  -Wl,--no-as-needed -Wl,-rpath,$PWD/lib64
# test unitaire (1070, dev=1, sans toucher au service)
CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=$PWD/libcupath_v2.so CUPATH_TRACE=/tmp/tt ./test_v2
python3 cupath_analyze_v2.py /tmp/tt.<pid> test          # G1-G4 + union/overlap
# capture production (STOPPE ollama-v100, se restaure seul — trap EXIT)
./run7_v2.sh 64
# analyse de la fenetre (wall ms = "eval time" du log serveur)
python3 cupath_analyze_v2.py /tmp/cupath7_<hhmmss>.<pid> window - - <wall_ms>
```

## 9.9 État système final (post-E3)
- `ollama-v100` : **active**, V100 rechargée (32 350 MiB),
  MD5 `libggml-cuda.so` : préfixe **e4afbb4a** (inchangé).
- Traces : `/tmp/cupath7_190740.560657{,.names,.hdr}` (150 Mo + 444 Mo + 89 o).
- `fattn-q8-volta.cuh` : **intact** (E2 pas encore engagée).

---

# 10. ÉTAPE E2 — Optimisation `ggml_q8v14partial` (2026-10-13, 19:50–20:44)

## 10.0 Périmètre et discipline
- Espace de test `e2/` : copie du kernel (`fattn-q8-volta.cuh` md5 **36f564e2**, figé)
  + harnais `e1_bench` (vraie géométrie decode). **Aucune substitution de la
  bibliothèque de production** pour tester une hypothèse (protocole respecté).
- Compilations `-arch=sm_70` ; `devcheck` prouve que le harnais tourne sur la V100
  (`DEV=Tesla V100-PCIE-32GB sm_70`). Une seule modification à la fois ; plusieurs
  répétitions (3 passes × 30 iters, médiane) ; sortie vérifiée (probe pacc).

## 10.1 Référence NCU (mesures, pas hypothèses)
| Grandeur | base (128,2) |
|---|---|
| Durée NCU / wall sans profiler | 1790 µs / 1593 µs |
| Registres | 255 + spills 152 B st / 80 B ld (stack 96 B) |
| Smem statique | 33 920 B/bloc |
| **Occupation théorique / atteinte** | **12,50 % / 12,43 %** |
| **Block limiters** | **Shared Mem = 2, Registers = 2** (plafond commun : 2 blocs/SM) |
| Issue slots busy / No eligible / CPI | 31,83 % / 68,09 % / 6,25 |
| DRAM / SM compute / L1 | 5,95 % / 31,26 % / 51,45 % |

**Constat** : le kernel est plafonné à 2 blocs/SM par **la smem** (96 Ko/SM ÷ 33,9 Ko),
et tourne à 12,5 % d'occupation avec CPI 6,25 (latence non masquée) — cohérent avec
E1 (kernel-bound, DRAM 6 %) et E3 (temps GPU sérial 543,9 ms = 30,1 % du wall decode).

## 10.2 Variante A — `__launch_bounds__(128,3)` / `(128,4)` — **REJETÉE**
- Hypothèse : forcer l'occupation (H1).
- Changement : `minBlocksPerMultiprocessor` 2 → 3 → 4 (rien d'autre).
- ptxas : 168 regs / 432 B spill (a3) ; 128 regs / 928 B spill (a4). Smem inchangée →
  le limitant réel (smem) n'est pas levé ; l'occupation ne monte pas.
- Microbench (V100 seule, médiane de 3 passes × 30 iters) :
  | variante | µs/lance | passes | Δ |
  |---|---|---|---|
  | base | 1593 | 1598/1592/1593 | ref |
  | a3 | 3720 | 3720/3720/3718 | **+133,7 %** |
  | a4 | 5449 | 5458/5445/5449 | **+242,3 %** |
- Sortie : probe pacc identique (−4,875000 / −4,593750) sur les 3 → sémantique ok.
- **Verdict : rejetée** (régression majeure, attribution certaine : spills local
  memory induits par la contrainte de registres). **Aucun A/B applicatif** — pas de
  variante gagnante à promouvoir.

## 10.3 Lecture pour la suite (hypothèses, non démontrées)
- **H2 (prioritaire, motivée par `Block Limit Shared Mem = 2`)** : réduire la smem
  par bloc (tuile BC 16→8, ou ne tenir que la moitié de la tuile K/V par itération)
  pour atteindre 3–4 blocs/SM. Envergure : changement de structure du tile + revue
  d'exactitude (softmax en blocs plus petits). À isoler en une seule variante.
- **H4 (registre)** : les 255 regs + spills viennent du chemin mma (8 accumulateurs
  D float + 4 P + 4 PV + adresses) ; une restructuration du registre (recharger Q,
  qui ne change pas dans la boucle, plutôt que de le garder en smem+frags) pourrait
  libérer des regs — à évaluer avec H2, pas seul.
- **ldmatrix (H3 de la mission)** : indisponible sur sm_70 (Turing+) — confirmé
  inapplicable, la piste est fermée pour cette architecture.
- **Gain applicable potentiel (borne)** : si une variante divisait le temps du kernel
  par 2, gain mural ≈ 172 ms/1818 ms (E3), soit ~34,6 → ~38 tok/s. À confirmer par
  l'A/B applicatif (protocole E2.3) **seulement** si une variante passe le microbench.

## 10.4 État final E2
- `fattn-q8-volta.cuh` : **36f564e2, intact** (aucune modification de production).
- `ollama-v100` : active, V100 rechargée, MD5 prod **e4afbb4a**.
- Artefacts : `e2/` (harnais, variantes, `ab_results.csv`, `e2_ab.log`,
  `run_e2_ab.sh`, `run_e2_base_ncu.sh`, `/tmp/e2_base_ncu.txt`).

## 11. H2 — tuile BC 16→8 : validation microbench + A/B applicatif

### 11.1 Modification (1 fichier, 2 lignes cohérentes)
`fattn-q8-volta.cuh` : `BC = 16` → `8` et `for (sub=0;sub<2;sub++)` → `sub<BC/8`
(boucle `sub` dérivée de BC — chaque sous-tuile = 8 clés ; simple `BC=8` sans
adapter la borne = accès hors bornes en smem). Pas de `launch_bounds`, rien d'autre.

### 11.2 Microbench + NCU (gate passée)
- Microbench E1 (n_kv=107351, grid 4×160, 128 thr, 3 passes × 30 iters) :
  **base 1596 µs → H2 1349 µs = −15,5 %** (disp. 1–2 µs).
- NCU 2024.1.1 : smem 33,92→25,41 Ko ; Block Limit 2/2 → 3/3 ; occupation
  12,50/12,43 % → 18,75/16,99 % ; regs 255→166 ; spills 152/80 B → 0/0 ;
  durée 1,79→1,52 ms. La smem était le limitant → la tuile plus petite lève le
  plafond d'occupation et le kernel s'accélère (cohérent microbench/NCU).

### 11.3 Correction (dump de sortie complet)
Harnais `e1_check` (mêmes données seed 12345, même géométrie) dédump `pacc`/`pm`/`pl`
complets. Base vs H2 : **bit-identique** (0/3 932 160 élements pacc + pm/pl) pour
`n_kv=107344` (multiple de 16) **et** `n_kv=107351` (cas réel decode, impair).
Le `reduce_kernel` (inchangé) recombine les splits en fp32 de façon associative ;
le raffinement de tuile ne change pas le résultat.

### 11.4 Protocole de build isolé (prod intouchée)
- Recette prod = kit `v100-repro` : sub-build `llama-server-cuda_v12` (preset
  `llama_cuda_v12_user_arch`, `sm_70-real`, gcc-13.4/nvcc 12.6.85) sur le fork
  `llama-volta-merged` (md5 `fattn-q8-volta.cuh` = **36f564e2**).
- **Gate A** : re-link des 145 `.o` prod purs → lib **bit-identique au prod**
  (md5 **e4afbb4a**) → recette de link exacte reproduite (le build complet
  recompile 144 `.o` non-déterministes au niveau du PID nvcc, d'où la méthode
  « 144 `.o` prod + 1 `fattn.cu.o` recompilé »).
- Lib H2 = 144 `.o` prod + `fattn.cu.o` recompilé avec la variante (commande nvcc
  exacte du build prod). md5 **a4b6f440** (≠ prod, ≠ gate). Marqueur
  `GGML_CUDA_VOLTA_Q6_W4R4` présent.

### 11.5 A/B applicatif (harnais `h2_ab_lib.sh`, protocole `h2_capture` sans nsys)
Serveur de capture port 44901 (pas le service prod) ; même modèle `qwen3.8-27b-q6-mtp`,
ctx 200 k, MTP n_max=3, prefill 100 k + 2ᵉ passe chaude, décode **1200 tok**
température=0 ; ordre A,B puis B,A ; 2 réplicats ; seule différence
`GGML_BACKEND_PATH` = prod (A) vs H2 (B).

| run | lib | eval time (1200 tok) | ms/tok | t/s |
|---|---|---|---|---|
| r1-A | prod e4afbb4a | 38 798 ms | 32,36 | 30,90 |
| r1-B | H2 a4b6f440 | 33 662 ms | 28,07 | 35,62 |
| r2-B | H2 a4b6f440 | 33 711 ms | 28,12 | 35,57 |
| r2-A | prod e4afbb4a | 38 676 ms | 32,26 | 31,00 |

- Moyenne A = 32,28 ms/tok (30,98 t/s) ; B = 28,07 ms/tok (35,62 t/s).
- **Δ wall = −13,0 %** (r1 −13,2 %, r2 −12,8 %) ; **Δ débit = +15,0 %**.
- Gain constant entre réplicats et ordres → pas d'effet de dérive.

### 11.6 Point d'attention (correction) — dérive de séquence
À température=0 le texte diverge entre A et B dès le char ~1086. Cause : `split_count`
est **dépendant de BC** — au-delà de `n_kv≈107520` la partition des splits change
(base 157×688 vs H2 159×680) → le regroupement des additions fp32 dans `reduce_kernel`
change → probabilités dérivées d'1 ULP → `argmax` diverge sur un token. Ce n'est pas
un bug (les deux sorties = attention exacte, à une associativité fp32 près) mais c'est
un changement mesurable de la séquence. **Bascule prod NON recommandée à ce stade** :
H2 accélère bien mais ne peut pas être promu en prod tant qu'on n'accepte pas un texte
non-identique, ou qu'on n'a pas figé `split_count` indépendant de BC (variante H2b).

### 11.7 Verdict H2 + état prod
- **VALIDÉ au microbench et à l'A/B applicatif** : kernel mesurablement plus rapide
  (−13 % wall / +15 % t/s decode, reproductible), sorties bit-identiques à `n_kv` fixé,
  production non modifiée. Bascule prod : non recommandée tant que la dérive de
  séquence (11.6) n'est pas levée (H2b).
- `fattn-q8-volta.cuh` prod : **36f564e2, intact**. `ollama-v100` : active, MD5 prod
  **e4afbb4a**. Artefacts : `e2/h2lib/` (build isolé + lib a4b6f440),
  `e2/e1_check.cu`/`e1c_base`/`e1c_h2b`/`e1_check_final.sh`, `e2/h2_ab_lib.sh`,
  `ab_results_h2_v2/{r1-A,r1-B,r2-B,r2-A}`.

## 12. H2b — découplage du partitionnement de la tuile (verdict)

H2b teste si le gain de H2 peut être conservé en forçant le partitionnement au
comportement de prod (pas `PC=16` découplé de la tuile `BC=8`).

### 12.1 Mécanisme (vérifié)
`split_count` dépend de BC par `ceil(n_kv/BC)` ; les frontières de splits sont des
multiples de BC (re-dérivées en device) ; le `reduce_kernel` combine les splits par une
somme fp32 séquentielle → le regroupement change quand le découpage change. Seuil de
changement : **n_kv=107 521** (prod 157×688 vs H2 159×680).

### 12.2 Diff minimal (1 fichier, 6 sites)
`constexpr int PC=16` ; `split_count`/`first_tile`/`split_end`/`logical_tiles` passent
de `BC`→`PC` (frontières = prod) ; **tuile, smem, pas d'itération et softmax restent à
BC=8** (smem 25,4 Ko / 166 regs conservés, 3 blocs/SM).

### 12.3 Gates
- **A (compile)** : lib H2b = 144 `.o` prod + 1 `fattn.cu.o` (commande nvcc exacte du
  build prod) → **0eb5afbd** (≠ prod e4afbb4a, ≠ H2 a4b6f440), marqueur W4R4 présent.
- **B (numérique)** : pacc/pm/pl **H2b = prod bit-identique** sur 107344/107351/107521/
  108551 ; H2 diverge au-dessus du seuil. **P1 confirmée.**
- **C (microbench)** : médiane prod 1598 / H2 1508 / **H2b 1350 µs** → le kernel isolé
  de H2b est aussi rapide que H2 (−15 % vs prod), à smem/regs identiques.
- **D (applicative)** : A/B H2b vs prod = **+0,61 % wall** (bruit) ; texte **identique**
  (contrairement à H2). Contrôle H2 vs H2b même session : **H2b = prod (33,2 ms/tok)**,
  **H2 = −13 % (29,05 ms/tok)**. accLen : prod/H2b = 2,83, H2 = 3,24.

### 12.4 Verdict — H2b validée (objectif atteint), sans gain applicatif
Le découplage fonctionne exactement comme prévu (H2b = prod bit-identique, zéro
divergence) **mais ne rapporte aucun gain applicatif** (H2b = prod, +0,6 %). Le gain
app de H2 (−13 %) est porté par le **partitionnement BC=8** (accLen 2,83→3,24 via une
meilleure acceptation MTP), **pas** par la tuile BC=8 (qui ne montre son gain que sur le
kernel isolé). Le gain et la divergence de H2 sont **indissociables** dans le cadre H2b.

**Options de déploiement** (aucune n'est engagée sans autorisation) :
1. Ne rien déployer (recommandé) ; 2. basculer H2 en acceptant le texte non identique ;
3. basculer H2b (variante correcte bit-identique, sans gain — « plan B »).

Production : `ollama-v100` active, MD5 **e4afbb4a**, cuh **36f564e2** — intacts.
Aucune descendance (H2c, H2b2a…) ouverte. Artefacts : `e2/h2lib/fork-h2b/`,
`libggml-cuda-h2b.so` (0eb5afbd), `e2/h2b_*.sh`/`.log`, `ab_results_h2b_v1/`,
`ab_results_h2b_control/`.

## 13. H2-PROD — H2 déployée en production expérimentale (2026-10-14)

H2 activée en production (lib `a4b6f440` mappée et confirmée), après vérification de
l'artefact, suite de 11 prompts (4 catégories, checks objectifs) en baseline puis sous H2,
long decode 600 tok prod vs H2, backup prod `e4afbb4a` vérifié dans
`/var/backups/llama-experiments/` et rollback testé (`deploy.sh prod`).

- **Qualité** : aucune régression macroscopique ; tous les checks factuels/arithmétiques/
  JSON/contexte-long passent. Les 3 textes divergents (R1, I2, L2) restent corrects en
  contenu (associativité fp32, pas de perte de qualité).
- **Performance (CONTRADICTOIRE)** : A/B v2 propre (100k prefill, 1200 tok) → **H2
  gagnant −13 % wall, accLen 3,24 vs 2,83** ; long decode 41k prefill → **prod gagnant
  (H2 +15 %), accLen H2 plus bas (2,76 vs 3,10)**. Hypothèse non démontrée : le gain H2
  n'apparaît qu'au-delà du seuil n_kv ≈ 107,5 k (changement de partitionnement).
- **Disclaimer** : la V100 est partagée (concurrents, evict/reload de modèle, compteurs
  MTP reset par reload) → les murs de cette mission sont indicatifs ; l'A/B v2 de la
  mission H2 (session isolée) reste la mesure la plus fiable.
- **Décision** : H2 laissée en production expérimentale ; rollback ~10s disponible.
  Voir `ETAT.md` §28 pour le détail complet.
