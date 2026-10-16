# RUN9 — A/B contrôlé REF (`e4afbb4a`) vs H2 (`a4b6f440`), contexte fixe ~41k

**Écrit AVANT tout lancement** (consigne de guile). Expérience comparative
suivant RUN8, pour trancher la contradiction des A/B précédents (§28.3 :
protocole A @100k → H2 gagne ; protocole B @41k → REF gagne).

## 0. Question / hypothèse (sans présupposer le vainqueur)

À **contexte fixe ~41k tokens**, quelle lib (référence `e4afbb4a` vs H2
`a4b6f440`) donne le meilleur débit de décodage et/ou le meilleur comportement
MTP, en session isolée ? Le but est de **clarifier** la contradiction A/B, pas
de confirmer un vainqueur a priori.

**Variable modifiée** : uniquement la lib `libggml-cuda.so` (ref vs H2).
**Tout le reste est figé** : même `llama-server`, mêmes flags, même prompt,
même contexte, même modèle, même GPU, même ordre de chauffe.

## 1. Méthode — sélection de lib SANS toucher la production

Le `.so` déployé (`ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so`, = H2
`a4b6f440`, état production-expérimental voulus) **n'est JAMAIS modifié** pendant
l'A/B. Chaque essai lance un `llama-server` dédié avec
`GGML_BACKEND_PATH` pointant vers la lib de l'essai :

| bras | lib | chemin | md5 |
|---|---|---|---|
| REF | référence prod | `/var/backups/llama-experiments/libggml-cuda.so.prod-20261014-134252` | `e4afbb4a…` |
| H2 | tuile BC 16→8 | `h2/cupti129-test/e2/h2lib/lib/libggml-cuda-h2.so` | `a4b6f440…` |

Avantage : **zéro mutation d'artefact de production**, pas de `deploy.sh`,
isolation totale. « Restauration de la référence » = vérifier à la fin que le
`.so` déployé reste `a4b6f440` (inchangé) ET que la lib ref `e4afbb4a` est
intacte dans le backup. **Pas de rollback de H2** (ce serait annuler la décision
H2-PROD §28.5, hors périmètre de cet A/B) — c'est la seule décision de périmètre
signalée.

Les 2 libs exportent `ggml_backend_init` (vérifié `nm -D` avant lancement) →
aucun risque de fallback CPU silencieux.

## 2. Protocole d'un essai (identique pour les 4, dérivé de `run8_capture.sh` validé)

Séquence, dans l'ordre, `llama-server` dédié (port 44911, `CUDA_VISIBLE_DEVICES=0`
= V100) :

1. `systemctl stop ollama-v100` (une seule fois au **début** de l'A/B), attendre
   V100 < 1000 MiB (aucun autre compute-app).
2. Lancement `llama-server` dédié, env figés (= RUN8) : `GGML_BACKEND_PATH=<lib
   de l'essai>`, `GGML_CUDA_VOLTA_W4R4=1`, shortlist MTP,
   `LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1`, `LLAMA_ARG_FIT_TARGET=1024`.
   Flags : `-c 200000 -np 1 -lv 4 --spec-type draft-mtp --spec-draft-n-max 3
   --spec-draft-backend-sampling --cache-type-k q8_0 --cache-type-v q8_0
   --flash-attn auto -b 512 -ub 512`.
3. **Gates de démarrage** : FIT (`successfully fit params`), DEVICE
   (`using device CUDA` contient `Tesla V100`), MODEL LOADED, SHORTLIST
   (`CUDA MTP shortlist`).
4. **Chauffe** : warmup 16 tok (« Say OK », retry ×5) puis **2 prefills du prompt
   L1** (num_predict=16) → KV/prompt cache chaud à 41k.
5. **Mesure** : decode **600 tokens** sur `L1 + instruction courte` (même prompt
   que §28.3 protocole B), mur mesuré par `time.time()` (pas `total_duration`
   ollama, non fiable — voir §anomalies), `temperature=0 top_k=1 ignore_eos`.
6. **MTP** : dernière ligne `draft acceptance = X (A accepted / B generated),
   mean len = Y` (per-requête) + `spec common_specu: statistics` (cumulée) du
   log `-lv 4`, et `draft_n/draft_n_accepted` de la réponse.
7. Enregistrement (JSONL) + kill du server dédié, attendre V100 < 1000 MiB avant
   l'essai suivant.

À la **fin** des 4 essais : `systemctl start ollama-v100`, vérifier service
active + `.so` déployé md5 = `a4b6f440` (inchangé) + lib ref intacte.

## 3. Paramètres de l'A/B

| paramètre | valeur |
|---|---|
| contexte cible | ~41k tokens (prompt L1 = 153 940 chars ≈ 41 508 tok) — **contexte réel ENREGISTRÉ** par essai (`prompt_eval_count`) |
| génération | 600 tokens / essai (`num_predict=600`) |
| répétitions | **2 par variante** (passage exploratoire) |
| **ordre** | **alterné** : `REF, H2, REF, H2` (limite l'effet d'ordre) |
| chauffe | warmup 16 tok + 2 prefills L1 (identique chaque essai) |
| isolation | `llama-server` dédié, service prod arrêté, V100 < 1000 MiB avant chaque essai |
| paramètres fixes | flags/env §2 (inchangés entre les bras) |

**Mesures par essai** : mur total, durée préremplissage (passes L1), débit
décodage (ms/tok et tok/s, sur le mur), longueur réellement traitée
(`prompt_eval_count` + `eval_count`), comportement MTP (acc_len, draft_n /
accepted), anomalies.

## 4. Critères de validité / invalidation (fixés AVANT lancement)

Un essai est **INVALIDE** (marqué, non compté) si l'un de :
- **G_FIT / G_DEVICE / G_LOAD / G_SHORTLIST** en échec (server pas bon).
- **G_CTX** : `prompt_eval_count` ∉ [38 000, 45 000] (contexte réel ≠ ~41k).
- **G_GEN** : `eval_count` ≠ 600 (decode tronqué / pas complet).
- **G_WALL** : mur ∉ [10 s, 120 s] (incohérent 600 tok à ~25–30 ms/tok).
- **G_MTP** : `draft_n` = 0 (MTP inactif → essai non comparable).
- **G_ISOLATION** : V100 ≥ 1000 MiB au démarrage d'un essai (autre client présent).

**Critère de décision comparatif** (exploratoire, n=2/bras) : on compare
ms/tok moyen et acc_len moyen par bras. Si l'écart inter-bras est **< la
variabilité intra-bras** (écart-type), le résultat est **indécis** → on propose
la répétition ciblée (n=3 ou 4/bras). Pas de seuil de « victoire » fixé a
priori — on rapporte les mesures brutes + la variance.

**Incident** : si la restauration du service est incertaine à la fin (service
pas `active` ou `.so` déployé ≠ `a4b6f440`), **arrêt + signalement précis**
(avant toute conclusion).

## 5. Supervision

- Lancement : `cd bench && python3 supervisor_bench.py run run9_ab.py --detach`
  (wrapper Python requis — le superviseur lance `python3 <bench>` ; cf README).
- Le superviseur attend la fin, vérifie passivement `ollama-v100`, relance
  l'inférence OWUI si erreur (tir non bloquant).
- **Après le cycle** : vérifier rc, état du service, ET confirmation OWUI
  (`done=True`, `error` vide) — la reprise n'est pas confirmée par le superviseur.
- Artefacts → `nightrun8/ab_run9/` (results.jsonl + logs serveur + summary) avec
  checksums.

## 6. Ce que l'A/B NE PRÉTEND PAS

- n=2/bras = **passage exploratoire** : pas de conclusion définitive, pas de
  déduction de seuil de performance (le 107 521 = divergence numérique, pas un
  crossover établi — RUN8 est un point de départ de profilage, pas une preuve).
- La comparaison est **à contexte ~41k** : elle éclaire le cas court/intermédiaire,
  pas le cas >107.5k (protocole A).
