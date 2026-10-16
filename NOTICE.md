# NOTICE — composants repris et leurs licences

Ce kit reprend du code et des artefacts de tiers. Voici l'inventaire et la
licenciation de chacun.

## 1. Code de ce kit (harnais, scripts, docs)
**Licence : MIT** (voir `LICENSE`). Écrit pour ces expériences ; aucune
dépendance avec du code tiers non listé ci-dessous.

## 2. llama.cpp (fork `b11081-v100-full`)
- **Base** : upstream llama.cpp `b11081` (commit `161755f29…`) — **licence MIT**
  (`llama.cpp/LICENSE`, « Copyright (c) 2025 ggml.ai / llama.cpp contributors »).
- **Fork** : merge de `v100-optimized` (kernels custom Volta : W4R4 Q6_K, Q5_X4,
  Q8 FA tensor cores, GDN 128×4, shortlist-head MTP, defer-prompt, checkpoints
  de prompt) + fixes. Les fichiers custom sont sous la même licence MIT (pas de
  clause restrictive ajoutée dans le fork).
- **Contenu du kit** : le fork **complet n'est pas inclus** (175 Mo de sources).
  Sont inclus : la source de référence `ggml/src/ggml-cuda/fattn-q8-volta.cuh`
  (md5 36f564e2) + les 2 patches du kit. Pour reconstruire le build complet,
  voir le kit `v100-repro` de la machine de référence (tarball de sources +
  `build_all.sh`, pins documentés dans `v100-repro/manifest.txt`).
- **Obligation MIT conservée** : le copyright ci-dessus reste applicable à tout
  code dérivé du fork.

  ```
  Copyright (c) 2025 ggml.ai / llama.cpp contributors
  Permission is hereby granted, free of charge, to any person obtaining a copy…
  ```

## 3. Ollama
- **Base** : ollama `v0.34.4-5` (commit pin `16b4376a…`) — **licence MIT**
  (`ollama/LICENSE`). Le fork ne modifie le binaire Go qu'à la compilation
  (pins `LLAMA_CPP_VERSION`), sans code Ollama modifié dans ce kit.

## 4. NVIDIA CUPTI / Nsight
- Les scripts `h2/cupath_shim*.c` utilisent l'**API CUPTI** (12.9) et les
  **headers** de Nsight Compute (2024.1.1).
- **Licence** : CUPTI est fourni sous licence NVIDIA (pas open source). L'usage
  dans ce kit est **interne/analytique** (tracing de processus locaux, pas de
  redistribution des libs CUPTI). Le **code shim est redistribuable** (MIT,
  `cupath_shim*.c`) mais **n'inclut aucune lib NVIDIA** — il faut installer
  CUPTI 12.9 soi-même (voir `REQUIREMENTS.md` §4).
- Les `.so` de CUPTI **ne sont PAS dans ce repo** (ni dans `bigfiles/`).

## 5. Modèle (qwen3.8-27b-q6-mtp)
- **Base** : Qwen3.8-27B (Unsloth quant Q6_K_M + couche MTP draft) — **licence
  Qwen** (voir le LICENSE du dépôt Qwen correspondant). Le modèle **n'est pas
  redistribué** par ce kit : il doit être (re)quantisé et chargé localement.
- Les **prompts de test** (`h2/prompt1200.json`, `prefill-107k.json`) sont du
  texte synthétique généré pour ces tests (récit répéteur) — libres d'usage.

## 6. CUDA toolkit / drivers
- Les scripts ciblent CUDA 12.6 (nvcc 12.6.85) et un driver ≥ 575 (tests faits
  sur 580.178.04). **Aucun binaire CUDA n'est inclus** dans le kit : tout est
  compilé chez l'utilisateur (`bigfiles/` contient uniquement les artefacts
  **du kit** : libs `libggml-cuda-h2*.so` = fork + patch, et les binaires
  microbench `bin-e1*` — tous compilés depuis les sources fournies).

## 7. Exclusions volontaires (secrets / données privées)
- **Aucun token, clé ou identifiant** n'est présent dans ce repo (vérifiable :
  grep `token|key|password` ne renvoie que les *noms* de variables).
- Le superviseur OWUI lit son token d'un **fichier 0600 hors repo**
  (`~/.config/kolbo-supervisor/token` par défaut, override `OWUI_SUPERVISOR_TOKEN_FILE`).
- Les résultats bruts ne contiennent que des sorties de bench (pas de données
  utilisateur : prompts synthétiques, réponses à des prompts synthétiques).

## 8. Résumé des obligations
| Composant | Licence | Action requise |
|---|---|---|
| Code du kit | MIT | copyright + licence conservés |
| llama.cpp (fork) | MIT | copyright conservé, pas de redistribution du fork non fourni |
| Ollama | MIT | idem |
| CUPTI / NCU (NVIDIA) | Licence NVIDIA | ne pas redistribuer les libs NVIDIA ; les shims sont MIT |
| Modèle Qwen | Licence Qwen | ne pas redistribuer le modèle |
| Prompts synthétiques | Libre (kit) | — |
