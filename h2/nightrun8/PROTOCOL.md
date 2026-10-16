# RUN8 — capture CUPTI v2 @ ctx ≈ 41k (mission nuit 14→15/10)

## Objectif (piste hors H2, §28.3/§28.4 d'ETAT.md)

Le protocole B de la mission H2-PROD (41k, service partagé) est **entaché**
(disclaimer §28.4 : GPU partagé, evict/reload, réplicat bloqué). Le protocole A
(100k, session isolée) est fiable mais ne couvre que ctx > seuil.
**RUN8 produit la moitié manquante : un CUPTI v2 propre à ctx ≈ 41k**, pour
trancher les questions ouvertes documentées :

1. **Niveau** : le régime `ggml_q8v14partial` à 41k est-il le même qu'à 100k
   (17 lances/round, ~1.6 ms/lance, ~30 % du wall) ? (mesure directe, pas de
   modèle)
2. **Split_count / seuil n_kv ≈ 107 521** (§26.5, §27.1, §28.3) : à n_kv < seuil,
   le découpage en splits de la variante de partitionnement diffère-t-il déjà,
   ou pas du tout ? → valide/infirme l'hypothèse "le gain du partitionnement
   n'existe qu'au-delà de ~107.5k" — **sans comparer H2** (H2 exclue des tests
   par la mission) : on mesure le régime de référence et on le compare aux
   valeurs déjà établies (run6 @100k : 160 splits/17 lances).
3. **Micro-gaps & syncs `k_set_rows`/`k_get_rows`** (~1 ms × 10 à 100k, E3) :
   présents à 41k ? même amplitude ? → borne du gain "gaps" par ctx.
4. **Occupation GPU & idle** à 41k : réplication de E3 (96.11 % @100k) hors du
   point de mesure habituel.

## Pourquoi ce choix (et pas autre chose)

- Hypothèse **documentée ET ouverte** (§28.3 "à re-vérifier proprement") ;
  toutes les autres pistes sont refermées (Tensor Core, ncols=8, P7, H-SYNC,
  H-BW, n_max=2, `__launch_bounds__`, partial mem-bound, overlap, FA draft,
  MoE, CUDA graphs — voir §17/§19-§27 d'ETAT.md).
- Aucune création de variante H2, aucun test H2 : RUN8 utilise uniquement la
  **bibliothèque de production** (H2, en place intentionnellement — jamais
  modifiée par ce run) comme binaire de capture, à la manière de run6/run7
  qui utilisaient la prod de l'époque.
- Instrumentation prouvée : shim CUPTI v2 (libcupath_v2.so) + analyseur
  (cupath_analyze_v2.py) + gates G1–G5 — pas de nouvelle instrumentation.

## Protocole (fixé AVANT mesure)

Harnais : `run8_capture.sh` (dérivé à l'identique de `run7_v2.sh`, seul
paramètre ajouté : `H2_CTX_CHARS=186000` ≈ 41.5k tokens — même taille de
prompt que le protocole B de la mission H2-PROD : 41 508 tokens, pour que les
résultats soient comparables).

1. Service `ollama-v100` stoppé (sudo, script), V100 libérée (< 1000 MiB),
   sinon `pkill -9 llama-server` (restauré ensuite).
2. `llama-server` dédié port 44901 : **mêmes args que prod et que run6/run7**
   (GGUF `sha256-4933…63430bb`, `-c 200000 -np 1`, `--spec-type draft-mtp
   --spec-draft-n-max 3 --spec-draft-backend-sampling`, KV q8_0,
   flash-attn auto, `-b 512 -ub 512`, `--flash-attn auto`,
   shortlist `qwen38-27b-exact-131072.i32`, `LLAMA_ARG_FIT_TARGET=1024`,
   `GGML_CUDA_VOLTA_Q6_W4R4=1`, `LD_PRELOAD=libcupath_v2.so`,
   `CUPATH_TRACE=/tmp/cupath8_<HHMMSS>`).
3. Gates avant mesure : FIT OK, device = Tesla V100, model loaded, warmup 16
   tok, shortlist chargée.
4. Prefill hors fenêtre : 2 passes du prompt 41.5k (1ère ~2-3 min, 2nde chaude).
5. Fenêtre de decode : **64 tokens**, température 0, `ignore_eos`,
   bracket temps réels (TS_START/TS_END) + `eval time` serveur comme wall.
6. Restauration (trap EXIT, toujours exécuté) : kill du server de capture,
   attente libération VRAM, `systemctl start ollama-v100`, attente API 11434,
   vérification `systemctl is-active` + md5 de la `.so` déployée.
7. Analyse : `cupath_analyze_v2.py` mode window (gates G1–G5, occupation,
   idle, gaps, top familles, overlap par paires de streams).

Enveloppe totale : ~15-25 min (stop+libération ~1-2 min, chargement ~2-3 min,
2 prefills ~4-6 min, decode ~3-4 s, analyse, restauration ~1 min).

## Critères de décision (fixés AVANT mesure)

- **Gates G1–G5** : passage obligatoire pour que la capture soit exploitable
  (G5 : |wall − union| < 2 %). Sinon : capture NON exploitable, documentée,
  pas de conclusion (pas de relance automatique).
- **Valide vs bruité** :
  - 64 tok générés et `finish_reason` présent ;
  - écart wall trace vs `eval time` serveur < 2 % ;
  - ≥ 25 000 records KER dans la fenêtre (comparable à E3 : 40 390 @64 tok) ;
  - pas de ligne `out of memory` / `fit failed` / crash dans le log serveur.
- **Lecture attendue (comparaison aux références)** :
  - `q8v14partial` : lances/round et ms/lance vs run6 (17 lances/rd, ~1.6 ms,
    543.9 ms/1818 ms = 30.1 % @100k) ;
  - gaps > 200 µs : nombre et amplitude vs E3 (3 gaps > 1 ms, 89 > 100 µs,
    médiane 0.64 µs) ;
  - occupation union/wall vs E3 (96.11 %) ;
  - splits : si identifiable dans les géométries loggées (grid de q8v14partial
    = (head_count_kv, splits)) → comparaison 160 vs valeur 41k.

## Contraintes de la mission (garde-fous)

- H2 exclue de toute comparaison/variante : RUN8 ne teste rien contre H2 ;
  c'est un régime de référence.
- Aucune modification de la production, aucun rollback : le script ne touche
  la `.so` de prod que par **vérification md5 en fin de restauration**
  (lecture seule).
- Le service est stoppé pendant le run (~15-25 min) : le superviseur
  (`supervisor_bench.py --detach`) relance l'inférence OWUI si une erreur est
  visible sur le chat agent au moment de la fin du run.
- Traces : `/tmp/cupath8_*` (volatiles) → archivage immédiat dans
  `h2/cupti129-test/nightrun8/traces/` avec MD5SUMS.

## Artefacts produits (attendus)

- `/tmp/cupath8_<HHMMSS>.<pid>` (.bin 56 o/record), `.names`, `.hdr`
- `/tmp/cupath8_server.out` (log serveur complet)
- `nightrun8/run8_capture.log` (stdout du script : gates, restauration)
- `nightrun8/analyse_window.txt` (sortie cupath_analyze_v2)
- `nightrun8/traces/` (copies vérifiées) + `MD5SUMS.txt`
