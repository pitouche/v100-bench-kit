# ETAT-cupath — Revue du tracage CUPTI 12.9 (run6) — findings séparés
Date : 2026-10-12 (05:30–08:00). Revue indépendante de la session 2026-10-11 (§15 ETAT.md) après
un `run_command` qui a explosé le contexte agent (894k tokens). Tout ici est vérifié sur les
artefacts au repos : `cupath_shim.c`, `run_cupath_llama.sh`, `cupath_analyze.py`,
`/tmp/cupath_llama_144318{,.names}`, `/tmp/cupath_server.out`, `/tmp/cupath_run6.log`.

Format : findings F1..F16 classés [CRITIQUE]/[MAJEUR]/[MINEUR]/[OK]. Chaque finding :
**Fait** (observé sur les artefacts) → **Conséquence** → **Action proposée**. À traiter un par un.

---

## F1 — CRITIQUE — `t` du `.names` n'est PAS un timestamp GPU : c'est le temps d'émission au flush
**Fait** (`cupath_shim.c` `emit_name()` : `now_real_us()` appelé dans `on_buf_done`, donc dans le
thread de drain CUPTI, 500 ms après les événements ; vérifié sur la trace run6) :
- Decode task 236 (pid 387290, 43 711 records) : `t` est borné sur **73,7 ms** pour **1575,8 ms**
  de durée kernels cumulée. Les 43 711 records sont émis en un seul bloc (0 saut > 2 ms dans `t`).
- `dur_ns` (horloge GPU CUPTI) et `t` (CLOCK_REALTIME au moment de l'émission) sont deux horloges
  indépendantes ; `t` ne donne ni start ni end GPU, ni position dans le wall du decode.
**Conséquence** :
- Le verdict §15.1 ETAT.md « GPU ~100 % busy, le GPU ne s'arrête JAMAIS pendant le decode,
  empiètement permanent » **n'est PAS démontré** : Σ dur (1575,8 ms) ≈ wall (1576,5 ms) mais sans
  positions start/end on ne peut pas prouver l'absence de trous.
- Tout calcul d'union d'intervalles / chevauchement sur `t` est invalide (c'est ce qui produit
  « union 0,075 ms / double-compte 1575,7 ms » dans `cupath_analyze.py`).
- `t` reste utile pour : l'ordre relatif (monotone croissant, vérifié), et l'attribution des
  records à un pid (le flush se fait avant la sortie du process).
**Action proposée** : v2 du shim — écrire dans chaque record les `start`/`end` GPU bruts de CUPTI
(`CUpti_ActivityKernel9.start/end`, 2×u64), garder `dur_ns` ; format v2 = 40 octets/record :
`u64 seq | u64 gpu_start | u64 gpu_end | u32 stream | u32 op`. Relancer UNE capture (run7) et
re-démontrer ou réfuter « 100 % busy » par union des intervalles GPU réels.

## F2 — CRITIQUE — 62 segments (forks) dans le MÊME fichier : tout parser non segmenté mélange les pids
**Fait** : 62 headers `[cupath] pid=` dans `cupath_llama_144318`. Le constructeur du shim tourne
dans **chaque fork** de llama-server (le server en fait des dizaines), et tous appendent sur le
même fd O_APPEND. Les segments CUDA significatifs (attribution par **groupe**, cf. F15) :
  | segment(s) (pid) | groupe de tâche (corrélation server.out, cf. F15) | KER | Σ dur |
  |---|---|---:|---:|
  | 387193 | **task 0 = warmup « Say OK » (~2 tok)** — profil identique à la trace de référence 142722 (mul_mat_vec_q<2>) | 2 405 | 41,3 ms |
  | **387221** | **task 4 = prefill 100k (prompt wall 195 219 ms)** — profil pur prefill (cutlass gemm, flash_attn_ext_f16, dequant q6_K), Σ ≈ wall | 608 469 | 195 186 ms |
  | **387280 + 387284 + 387287** | **groupe central = task 4 decode 1er (32 tok, wall 2 577,5 ms, 83 ms/tok : construction des CUDA graphs) + task 225 (prompt 4 tok + decode 32 tok, wall 961,8 ms)** — profil decode (mul_mat_vec_q<4>, q8v14partial, flash_attn_ext_vec) | 45 213 | 1 584,2 ms |
  | **387290** | **task 236 = decode 64 tok (eval)** — le prompt 4 tok de task 236 (125,47 ms) est en queue de 387287 | 42 618 | 1 575,8 ms |
- 612 134 records de 387221 = 87 % des records : tout parser qui lit le fichier sans segmenter
  produit des compteurs aberrants (c'est exactement ce que les runs d'analyse des 01:00–01:07 ont
  fait — cf. F11).
- Les 4 traces précédentes (112305/115901/123510/142722) ne contiennent que **2 405 KER / 888 CPY**
  (= le seul profil warmup 16 tok) : ces captures n'ont pas atteint le prefill/decode.
**Conséquence** : le run6 est **analysable** (F4 : bin↔names alignés) **uniquement par segment /
par groupe de segments**. Toute agrégation globale (la section « GLOBAL » de `cupath_analyze.py`)
est fausse.
**Action proposée** : `cupath_analyze.py` prend le pid (ou un ensemble de pids / range de t) en
argument ; à terme, un fichier **par pid** (`$OUT.pid<N>`) côté shim.

## F3 — MAJEUR — le mapping pid → tâche est déduit (contraintes multiples), pas observé directement
**Fait** : `cupath_server.out` n'expose aucun pid (`--no-log-prefix`) ; run6.log ne donne que
`llama-server pid=387075`. L'attribution (387290 = eval task 236) est **déduite** par :
(a) Σ dur du segment ≈ wall du log (1575,8 ≈ 1576,52 ms), (b) ordre des headers dans le fichier,
(c) profil de familles. La contrainte (a) seule ne suffit PAS pour le groupe central
(387280+387284+387287 = task 4 decode 1er + task 225, frontière interne indéterminable — cf. F15).
**Conséquence** : attribution **saine pour le run6** (contraintes multiples satisfaites), mais
fragile comme méthode générale.
**Action proposée** : v2 du shim — `t_real_us` (CLOCK_REALTIME) dans le header pour corrélater
segments ↔ horodatages du log serveur (date) ; ou logger le pid du slot.

## F4 — OK — `.bin` et `.names` sont parfaitement alignés : 705 294 = 705 294
**Fait** : en suivant l'ordre des segments, les records non-reset du `.bin` (op 0/1) et les lignes
du `.names` sont en bijection exacte (705 294 de chaque côté, même ordre, `dur_ns` identiques,
`stream` identiques). `op=2` (reset) n'existe que dans le `.bin` (62). Vérifié sur les 5 autres
traces aussi (toutes alignées).
**Conséquence** : l'attribution par pid est fiable ; on peut rejouer toute l'analyse du run6
aujourd'hui sans recapturer.
**Action proposée** : rien ; documenter dans la docstring de `cupath_analyze.py`.

## F5 — MAJEUR — la sélection de fenêtre de `cupath_analyze.py` retombe sur le bon segment par chance
**Fait** : `cupath_analyze.py` recalcule `t_end = max(t+dur)` puis `t0 = t_end − 1576,52 ms` sur le
`CLOCK_REALTIME` d'émission. Le flush final de 387290 se faisant juste après la fin du decode, la
fenêtre retombe presque exactement sur le segment 387290 (42 618 KER = le compte exact du
segment). Mais c'est **accidentel** : ça ne marcherait pas pour task 225 (groupe de segments,
F15) ni pour un run où un autre segment succède au decode.
**Conséquence** : fragile, non reproductible.
**Action proposée** : sélection explicite par pid/groupe (F2/F3) ; garder la fenêtre t seulement
comme croisement de contrôle.

## F6 — MINEUR — les sorties HTTP sont polluéées par le stderr du shim → `json.load()` échoue
**Fait** : `cupath_gen.json`, `cupath_prefill.out`, `cupath_prefill2.out`, `cupath_warm.json`
commencent tous par `[cupath] tracing actif -> ...`. Mécanisme : le shell de `run_cupath_llama.sh`
a son stderr sur le tty ; le server hérite (redirigé vers `$SOUT`) mais les **curls de capture**
héritent du stderr tty ; les forks de llama-server créés pendant le curl héritent du même tty et
y écrivent leur ligne de constructeur. Résultat : `json.load()` échoue sur les 4 fichiers et le
script d'analyse inline de run6 **crashe** (`JSONDecodeError` en fin de run6.log).
**Conséquence** : le tableau final d'analyse du run6 n'a jamais été produit par le script inline ;
l'analyse a été refaite à la main (les runs de 01:00, puis cette revue).
**Action proposée** : le shim écrit son statut dans le fichier de trace (header) et/ou silencie le
message ; côté script, isoler le stderr des curls (`2>/dev/null` sur le corps JSON ou parse
tolérant).

## F7 — MINEUR — `run_cupath_llama.sh` : le message de restore annonce `$OUT.bin` mais le fichier est `$OUT`
**Fait** : `CUPATH_TRACE=$OUT` avec `OUT=/tmp/cupath_llama_$H%M%S` → le shim écrit `$OUT` et
`$OUT.names`. run5.log montre l'impact réel : l'analyse inline cherchait
`/tmp/cupath_llama_142722.bin.names` → **FileNotFoundError** (run5 mort avant analyse).
**Conséquence** : confusion récurrente entre les noms `$OUT` et `$OUT.bin` (les logs disent
« traces : X.bin / X.bin.names »).
**Action proposée** : soit renommer la sortie `$OUT.bin`, soit corriger les messages ; pointer
l'analyse inline sur le bon nom.

## F8 — MINEUR — deux implémentations d'analyse dupliquées et divergentes
**Fait** : l'analyse fenêtre-decode existe deux fois : inline dans `run_cupath_llama.sh`
(fenêtre TS0..TS1, top 18 familles) et dans `cupath_analyze.py` (fenêtre t0..t_end, familles +
variantes exactes + parité). Bornes et métriques différentes.
**Action proposée** : garder `cupath_analyze.py` comme seul analyseur ; le .sh l'appelle.

## F9 — OK — la cohérence bin↔names est parfaite segment par segment
**Fait** : sur les 62 segments du run6 (et sur les 5 autres traces), pour chacun : nb KER/CPY du
`.bin` == nb lignes du `.names`, `stream` et `dur_ns` identiques. Globales run6 : 698 499 KER +
6 795 CPY = 705 294.
**Conséquence** : aucune perte de record ; le drain 500 ms + flush final forcé (atexit) a tout
capturé. Le format 24B `u64 seq | u64 dur_ns | u32 stream | u32 op` est correct et stable.
**Action proposée** : rien.

## F10 — MINEUR — `cupath_analyze.py` (état du 12/10) : unités corrigées, mais 3 réserves
**Fait** : les conversions t(us)/dur_ns(ns) sont corrigées (`r[1]+r[3]//1000`, seuils fenêtre
200 ms = 200 000 us, gaps > 200 ms). Le script tourne et reproduit le rapport §15.1. MAIS :
(a) la section « CHEVAUCEMENT » reste **fausse en principe** (F1) ; (b) la section « GLOBAL »
agrège tous les pids (F2) ; (c) `ROUNDS = 64/3.71 = 17,25` : le log donne 51 drafts générés /
33 drafts acceptés → 18 rounds réels, pas 17,25 (le mean_len 3,71 = 64/17,25 est une moyenne,
pas un nombre de rounds entier).
**Action proposée** : (1) sélection par pid ; (2) retirer/désactiver « CHEVAUCEMENT » tant que F1
n'est pas résolu ; (3) prendre les rounds depuis le log serveur (`#gen drafts`), pas depuis
mean_len.

## F11 — MINEUR — l'analyse des 01:00–01:07 (qui a explosé le contexte) a parsé le `.bin` sans segmenter
**Fait** : le run `20261012-010751` (log 893 kB) a parsé le `.bin` de 144318 comme un seul flux
`<QQII>` **sans sauter les 62 headers texte** → les bytes des headers interprétés comme records
→ compteurs d'op corrompus (ex. `op=959528308: 8`), `max dur = 3,5e18 ns`, et impression d'un
`Counter` de ~600 000 entrées = source de l'explosion de contexte (894k tokens).
**Conséquence** : ces résultats sont à jeter (déjà abandonnés). Leçon : tout parser `.bin` DOIT
segmenter par header (F2).
**Action proposée** : encoder la segmentation par header dans `cupath_analyze.py` (ou utilitaire
`cupath_bin.py`) ; ne jamais parser le `.bin` en un seul bloc.

## F12 — OK — la génération est valide : 64 tok, finish_reason=length, wall 1576,52 ms
**Fait** : `cupath_gen.json` (après suppression de la ligne shim) → `completion_tokens=64`,
`finish_reason=length` ; server.out task 236 : `eval time = 1576.52 ms / 64 tokens`,
`draft acceptance = 0.90196 (46/51)`, `mean len = 3.71`, `acc per pos = (0.941, 0.882, 0.882)`,
`graphs reused = 32` ; prompt chaud (checkpoint 149,646 MiB restauré, `hits=2`).
**Conséquence** : le scénario de capture est conforme au protocole §15.1 ; le run6 est exploitable.

## F13 — MINEUR — les 5 premiers headers vides (seg 0..4) = forks non-CUDA
**Fait** : seg 0 (pid 387076, 0 rec), seg 1 (387075, 1 reset), seg 2 (387081, 0 rec), seg 3
(387082, 1 reset), seg 4 (387090, 1 reset). Processus sans CUDA (wrapper, init) qui initialisent
le shim puis sortent.
**Conséquence** : normal ; confirme F2 (coût nul, mais bruit).
**Action proposée** : optionnel — n'activer le shim que si `cuInit`/contexte CUDA détecté.

## F14 — MINEUR — `WALL_MS=1576.52` codé en dur dans `cupath_analyze.py`
**Fait** : défaut = wall du log serveur task 236. Pour task 225 (803,53 ms) ou un autre run il
faut passer `argv[2]` (déjà prévu). Documenter que ce wall provient du `print_timing` du log
serveur, pas de la trace.

## F15 — MAJEUR — un task s'étale sur PLUSIEURS segments (forks) consécutifs + le « prefill ×2 »
du script est en réalité pre1 (100k) + pre2 (4 tok chaud) + le warmup est la task 0 (pas task 4)
**Fait** (server.out, grep `print_timing`) : 4 tâches —
- task 0 : `prompt eval time = 196,58 ms / 15 tok` + `eval time` (chaîne « Say OK » → **warmup** ~2 tok).
- task 4 : `prompt eval time = 195219,27 ms / 107351 tok` (= **pre1, 100k**, ~549,9 tok/s) +
  `eval time = 2577,50 ms / 32 tok` = **1er decode** à **83,15 ms/tok** (3× plus lent que les suivants
  → construction des CUDA graphs / JIT : `graphs reused = 8`).
- task 225 : `prompt eval time = 158,28 ms / 4 tok` (pre2 **chaud** : checkpoint 149,6 MiB restauré,
  4 tok restants) + `eval time = 803,53 ms / 32 tok` (decode, 25,92 ms/tok).
- task 236 : `prompt eval time = 125,47 ms / 4 tok` + `eval time = 1576,52 ms / 64 tok` (decode).
→ Le « prefill ~100k ×2 hors fenêtre » de `run_cupath_llama.sh` = **pre1 100k (task 4) + pre2
4-tok-chaud (task 225)** ; la « 2e demande identique » n'est PAS une 2e fois 100k.
**Fait** (trace, profils de familles) :
- 387193 (2 405 ker, 41,3 ms) : profil **identique** à la trace de référence 142722
  (mul_mat_vec_q<2> : 20,0 / 6,7 / 4,8 ms) = warmup ~2 tok (task 0). **Certain** (profil + échelle).
- 387221 (608 469 ker, 195 186 ms) : Σ ≈ wall du prompt task 4 (195 219 ms) ; profil 100 % prefill
  (le dernier 10 % est encore préfill : flash_attn_ext_f16 dominant). **Certain** (Σ ≈ wall + profil).
- 387280 (735,0) + 387284 (122,4) + 387287 (726,8) : profil **decode** ; Σ = 1 584,2 ms. Ce groupe
  ne peut être que **task 4 decode (2 577,5 ms wall, 1er decode lent) + task 225 (961,8 ms wall)** :
  les deux sont les seuls tâches decode restantes (task 236 = 387290, F12), et le Σ du groupe est
  cohérent (1er decode CPU-bound : ratio kernel/wall ~45 %, decode chaud ~90 %). **Déduit**
  (exclusion + échelle) ; la frontière interne au groupe n'est pas déterminable sans t_real_us (F16).
- 387290 (42 618 ker, 1 575,8 ms) : Σ ≈ wall **eval** task 236 (1 576,52 ms ; pas le total time
  1 701,99 ms) → c'est l'eval de task 236, son prompt (125,47 ms) étant en queue de 387287.
  **Certain** (Σ ≈ wall + ordre + profil).
**Conséquence** : l'attribution par « le segment dont Σ dur ≈ wall » ne marche que pour task 236
et pre1 task 4 ; pour le groupe central c'est par **exclusion + échelle**. Le §15.1 (task 236 =
segment 387290 seul) reste **valide tel quel**.
**Action proposée** : `cupath_analyze.py` agrège au niveau **groupe de segments** (ensemble de
pids) ; v2 du shim (F16) pour dater précisément chaque segment ; corriger le commentaire du script
de capture (« ×2 » → « pre1 + pre2 chaud »).

## F16 — MAJEUR — le `t` (CLOCK_MONOTONIC) des headers n'est PAS une horloge exploitable entre segments
**Fait** : les `t` des headers (bruts, extraits du fichier) :
  - seg37 (387221, prefill) : 197 376 234 809 734
  - seg38 (387277) : 197 574 326 757 420 → **Δ = +198 091 s** entre deux segments espacés de
    quelques secondes en réel
  - seg39..42 (task 225/236) : 197 574 333 566 760 / 197 575 474 971 532 / 197 575 484 526 709 /
    197 577 341 998 653 → deltas cohérents (s)
  - et Δ ≈ +1005 s récurrent entre d'autres paires consécutives (seg5→6, 16→17, 19→20, 22→23,
    25→26, 28→29, 31→32…)
  - la série est globalement croissante mais avec des sauts de +16,75 min pour ~16 s de réel.
  Hypothèse : llama-server crée des forks dans des **namespaces d'horloges** (ou des sous-systèmes
  dont le CLOCK_MONOTONIC est referencé différemment — conteneur/sandbox) ; `now_ns()` du shim ne
  renvoie donc pas une horloge commune. Le `t` d'émission du `.names` (CLOCK_REALTIME) est en
  revanche cohérent et monotone (F1) — ce sont les `t` des **headers** qui sont hors échelle.
**Conséquence** : on ne peut PAS utiliser les `t` de headers pour dater les segments ou les
corréler au log serveur. Seuls l'ordre d'append dans le fichier + les sommes de durées + le
`CLOCK_REALTIME` d'émission (`.names`) sont exploitables.
**Action proposée** : v2 du shim — header avec `t_real_us` (CLOCK_REALTIME) ; ne plus lire
CLOCK_MONOTONIC dans le header.

---

## Synthèse (état de l'art, 2026-10-12)
- **Le run6 est valide et exploitable** (F4, F9, F12, F15) : 64 tok, 1576,52 ms, 42 618 kernels
  (segment 387290 = eval task 236 seul), Σ dur 1575,8 ms, profils §15.1 confirmés (mul_mat_vec_q
  53 %, ggml_q8v14partial 29,8 %, flash_attn 4,8 %, draft stream 17 = 10,7 %). La réattribution des
  segments (F15) confirme que le §15.1, limité à task 236, est cohérent.
- **Le verdict « GPU 100 % busy » est à re-démontrer** (F1) : c'est la seule affirmation du §15.1
  qui repose sur une coïncidence de sommes, pas sur des positions GPU. Le reste du §15.1
  (répartition par famille, nb de lances, streams) tient sur `dur_ns` + attribution par groupe,
  donc est sain.
- **Les artefacts de référence** : `/tmp/cupath_llama_144318` (bin, 16,9 Mo, 62 segments) +
  `/tmp/cupath_llama_144318.names` (89,6 Mo) ; à **copier vers le repo** (survivre au reboot).
  Les 4 traces précédentes ne contiennent que le warmup (2 405 ker) — aucune valeur run.
- `report2.nsys-rep`/`.sqlite` = le rapport nsys vide de §1 (confirmé : aucune table GPU/CUPTI,
  uniquement OSRT/DIAGNOSTIC) — inerte pour le run6.
- **Travaux à faire, par priorité** :
  1. F1 (+F16) — v2 du shim : gpu_start/gpu_end par record + t_real_us dans le header +
     silencing du message stderr (F6) ; recapture courte (run7) → re-démontrer « busy ».
  2. F2/F3/F15 — `cupath_analyze.py` par groupe de segments (ensemble de pids), rounds depuis le
     log (F10c), suppression de « CHEVAUCEMENT » tant que F1 est ouvert.
  3. F7/F8/F11 — unifier les noms de fichiers, un seul analyseur, parser `.bin` toujours
     segmenté par header.
  4. Sauvegarde des traces du run6 dans le repo.

## Garde-fou
- Aucune donnée système modifiée pendant la revue : lecture seule sur les traces ; seules
  modifications = corrections d'unités dans `cupath_analyze.py` (12/10) et création du présent
  fichier. `ollama-v100` inchangé (aucune capture lancée).
