# RUN9B — A/B corrigé REF (`e4afbb4a`) vs H2 (`a4b6f440`), contexte fixe ~41k (v2)

**Écrit AVANT tout lancement** (2026-10-15). Correction du RUN9 (13:56–14:02 UTC),
dont les 4 essais ont été invalidés par le gate G_MTP — invalidation **mal fondée**,
car le gate lisait `draft_n` de la réponse `/v1/chat/completions`, un champ que ce
binaire n'expose pas (vérifié dans la source : `server-common.cpp:96` conditionne
`draft_n` sur `n_draft_tokens > 0`, mais le handler OpenAI du binaire déployé ne
l'inclut pas dans la réponse — confirmé par un tir de contrôle sur la prod à
14:56 UTC : aucune clé `draft_n` dans la réponse, alors que le log serveur montre
bien MTP actif). MTP était actif sur les 4 essais (mtp_request 54.9 %/53.3 %,
mean len 2.64/2.59) : la performance mesurée était donc valable, seul le gate
était faux.

## 0. Question

À contexte fixe ~41k tokens (prompt L1 41 508 tok + instruction), quelle lib
(REF `e4afbb4a` vs H2 `a4b6f440`) donne le meilleur débit de décodage et le même
ou différent comportement MTP, en session isolée ? Objectif : trancher la
contradiction §28.3 (protocole A @100k → H2 gagne ; protocole B @41k → REF
gagne), dont le protocole B était entaché (GPU partagé, §28.4).

## 1. Variables

- **Variable modifiée** : uniquement `GGML_BACKEND_PATH` (lib REF vs lib H2) sur un
  `llama-server` dédié. Zéro mutation d'artefact de production pendant l'A/B.
- **Tout le reste figé** : même binaire `llama-server` (chemin prod), mêmes flags,
  même GGUF (blob sha256-4933…30bb), même prompt, même chauffe, même GPU, même
  ordre de chauffe, même modèle de mesure.

| bras | lib | md5 (vérifié à chaque essai avant lancement) |
|---|---|---|
| REF | `/var/backups/llama-experiments/libggml-cuda.so.prod-20261014-134252` | `e4afbb4a` |
| H2 | `h2/cupti129-test/e2/h2lib/lib/libggml-cuda-h2.so` | `a4b6f440` |

**Correction d'environnement** (leçon RUN9) : le .so ne lit que
`GGML_CUDA_VOLTA_Q6_W4R4` (vérifié par strings sur les deux libs) ; RUN9 exportait
`GGML_CUDA_VOLTA_W4R4=1` (nom erroné, inopérant) et **pas** `GGML_CUDA_VOLTA_Q6_W4R4`
→ les deux bras de RUN9 tournaient **sans** le chemin W4R4. RUN9B exporte
`GGML_CUDA_VOLTA_Q6_W4R4=1` (parité prod stricte, lue dans /proc/<pid>/environ du
llama-server prod via sudo).

## 2. Protocole d'un essai (identique REF/H2)

Serveur dédié port **44911**, `CUDA_VISIBLE_DEVICES=0` (V100), service prod
stoppé une seule fois au début, restauré à la fin :

1. Gates de démarrage : port up, **FIT** (`successfully fit params`),
   **DEVICE** (`using device CUDA…Tesla V100`), **MODEL LOADED**
   (`llama_server: model loaded`), shortlist (`CUDA MTP shortlist: loaded`).
2. Chaude : warmup 16 tok (« Say OK. ») + **2 prefills** du prompt (max_tokens=32)
   → KV chaud à 41k (2ᵉ prefill doit montrer cached_tokens ≈ 41 464).
3. Mesure : **decode 600 tokens** sur le même prompt + instruction courte,
   `temperature=0, top_k=1, ignore_eos, enable_thinking=false`,
   **`logprobs: true, top_logprobs: 8`** (nouveauté : proximité des logits
   REF vs H2, question 4 du protocole V100), mur mesuré par `time.time()`.
4. MTP : dernière ligne `draft acceptance = X (A/B), mean len` du log (par
   requête) + fenêtre = delta des 2 dernières lignes `spec common_specu:
   statistics` (gen/acc/calls de la requête seulement) + acc_rate/pos cumulée.
5. Sauvegarde : réponse JSON complète (texte + logprobs), log serveur complet,
   ligne de config effective (env exact du Popen), md5 de la lib chargée.

## 3. Paramètres

| paramètre | valeur |
|---|---|
| contexte réel | ~41 508 tok (prompt L1 153 940 chars + instruction 119 chars) — enregistré par essai (`prompt_tokens`) |
| génération | 600 tokens (`max_tokens=600`) |
| répétitions | **2 par bras** (passage exploratoire) |
| ordre | **alterné : REF, H2, REF, H2** |
| flags | `-c 200000 -np 1 -lv 4 --no-log-prefix --no-log-timestamps --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn auto -b 512 -ub 512` |
| env | `GGML_BACKEND_PATH=<lib>`, `GGML_CUDA_VOLTA_Q6_W4R4=1`, `GGML_CUDA_QWEN35_MTP_SHORTLIST=…/qwen38-27b-exact-131072.i32`, `LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1`, `LLAMA_ARG_FIT_TARGET=1024`, `CUDA_VISIBLE_DEVICES=0`, `LD_LIBRARY_PATH` (prod `lib/ollama` avant `cuda_v12`, puis cuda-12.6) |

## 4. Critères de validité (fixés AVANT lancement)

Un essai est **INVALIDE** si :
- **G_FIT / G_DEVICE / G_LOAD** en échec ;
- **G_CTX** : `prompt_tokens` ∉ [38 000, 45 000] ;
- **G_GEN** : `completion_tokens` ≠ 600 ou `finish_reason` ≠ `length` ;
- **G_WALL** : mur ∉ [10 s, 120 s] ;
- **G_MTP** (corrigé) : `generated` de la requête (dernière ligne `draft
  acceptance`) ≤ 0 **et** `mtp_cum delta gen` ≤ 0 — c'est le log serveur qui fait
  foi, plus la réponse ;
- **G_ISOLATION** : V100 (idx nvidia-smi 1) ≥ 1000 MiB au démarrage de l'essai.

**Décision comparative** : n=2/bras = exploratoire. On compare ms/tok moyen et
mtp_acceptance moyen par bras ; si l'écart inter-bras < max(écart-type intra-bras,
2 %) → INDÉCIS → proposer n=3-4/bras. Pas de seuil de victoire a priori.

**Logprobs (correction, question 4)** : pour chaque token i, distance =
|logprob_REF(i) − logprob_H2(i)| sur le top-8 commun ; on rapporte la
distribution (médiane, p90, max, fraction > 0.1 nats) et les positions des
tokens où l'argmax diverge. À texte égal : proximité directe des logits. À texte
divergent (au-delà du seuil n_kv≈107 521 — non attendu ici à 41k) : divergence
signalée, comparaison de logits limitée aux tokens communs.

## 5. Protection production / supervision

- Le `.so` déployé (H2 `a4b6f440`, état production-expérimental voulu, §28.5)
  n'est **jamais modifié** ; vérifié avant/après (doit rester `a4b6f440`).
- Service stoppé au début, `systemctl start` à la fin ; vérification
  `systemctl is-active` + API 11434. Tout incident de restauration → arrêt,
  signalement, aucune conclusion.
- Lancement via `bench/supervisor_bench.py run run9b_bench.py --detach`
  (wrapper Python obligatoire, cf README) ; le superviseur attend la fin,
  vérifie passivement le service et relance l'inférence OWUI si erreur.
- Artefacts : `ab_run9b/` (results.jsonl, gen_*.json, server_*.out) ;
  durée estimée ~30–40 min (4 × (load ~2 min + 2 prefills ~2 min + decode ~20 s)).

## 6. Ce que l'A/B NE PRÉTEND PAS

- n=2/bras : passage exploratoire, pas de validation statistique définitive.
- Contexte 41k : éclaire le cas < seuil (107 521), pas le cas > seuil.
- Les 4 essais RUN9 restent **non comparables entre eux** (env W4R4 inopérant,
  voir §1) : ils servent de témoin du régime « sans W4R4 » (REF ≈ 32.3 ms/tok,
  H2 ≈ 33.3 ms/tok) mais ne font pas partie des résultats.
