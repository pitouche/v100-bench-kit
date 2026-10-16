# RUN9C — A/B REF (`e4afbb4a`) vs H2 (`a4b6f440`) à ~100k, avec condition W4R4 OFF (v1)

**Écrit AVANT tout lancement** (2026-10-15, ~18:50 UTC). Suite de RUN9B
(`PROTOCOL-run9b.md`, §29 de `h2/ETAT.md`) : à 41k, H2 ≈ REF (Δ +2,1 % wall,
INDÉCIS, divergence textuelle dès le token 190). Le gain H2 reproductible
(−13 % wall, §26.5) a été mesuré à **~100k/107 351** — il faut le re-mesurer
proprement au même régime, et trancher la variable W4R4.

## 0. Questions

1. **Q1 (performance)** : à ~107k tokens, H2 est-elle plus rapide que REF ?
   (reprise de §26.5 : −13 % attendu si l'effet se confirme).
2. **Q2 (variable W4R4)** : le gain H2 tient-il avec `GGML_CUDA_VOLTA_Q6_W4R4=0`
   (chemin Q6_K W4R4 désactivé explicitement) ? — 1 paire REF + 1 paire H2 en
   condition OFF.
3. **Q3 (numérique/comportement)** : même divergences de texte et de MTP qu'à
   41k ? (le texte fait référence ; top_logprobs inutilisables, §29.1-3.)

## 1. Variables

- **Seule variable modifiée** : `GGML_BACKEND_PATH` (lib REF vs H2) et, pour la
  condition OFF, `GGML_CUDA_VOLTA_Q6_W4R4` (`1` → `0`). Zéro mutation d'artefact
  de production pendant l'A/B.
- Tout le reste figé et **identique à RUN9B** (même binaire, même GGUF, même
  port, mêmes flags, même chauffe, même modèle de mesure).

| bras | lib (md5 attendu) | W4R4 |
|---|---|---|
| REF | `/var/backups/llama-experiments/libggml-cuda.so.prod-20261014-134252` (`e4afbb4a`) | — |
| H2 | `h2/cupti129-test/e2/h2lib/lib/libggml-cuda-h2.so` (`a4b6f440`) | — |

**Régime W4R4** : le `.so` traite l'absence de variable comme **ON**
(`mmvq.cu:1460` : `env == nullptr || atoi != 0`). La condition `W4R4=1`
reproduit donc le régime de §26.5 (variable non exportée) ; la condition
`W4R4=0` isole la contribution du chemin Q6_K W4R4.

## 2. Essais et ordre (fixés AVANT lancement)

| # | tag | lib | W4R4 |
|---|---|---|---|
| 1 | ref1 | REF `e4afbb4a` | 1 |
| 2 | h2a1 | H2 `a4b6f440` | 1 |
| 3 | ref2 | REF `e4afbb4a` | 1 |
| 4 | h2a2 | H2 `a4b6f440` | 1 |
| 5 | ref3 | REF `e4afbb4a` | **0** |
| 6 | h2a3 | H2 `a4b6f440` | **0** |

Ordre alterné REF/H2, 2 réplicats en W4R4=1, puis **1 paire W4R4=0** (ajout
uniquement prévu par la question de l'utilisateur du 15/10). n=2/bras =
exploratoire ; n=1/bras en condition OFF = indicative (variable isolée, pas de
conclusion de puissance).

## 3. Paramètres

| paramètre | valeur |
|---|---|
| prompt | **identique au régime §26.5** : `h2/runs/run-20261009-152723/prompt.json` = premier tiers de `v100-repro/work/filler-200k.txt` (550 000 chars), **107 351 tokens** (mesuré) — même prompt que l'A/B §26.5 pour comparabilité directe |
| génération | **1200 tokens** (`max_tokens=1200`) — même que §26.5 |
| décodage | `temperature=0, top_k=1, ignore_eos, enable_thinking=false` |
| MTP | `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling` (parité prod) |
| chauffe | warmup 16 tok + **2 prefills** du prompt 107k (2ᵉ passe chaude, cached ≈ 107 347) |
| flags | `-c 200000 -np 1 -lv 4 --no-log-prefix --no-log-timestamps --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn auto -b 512 -ub 512` |
| env | `GGML_BACKEND_PATH=<lib>`, `GGML_CUDA_VOLTA_Q6_W4R4=<1\|0>`, shortlist, `LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1`, `CUDA_VISIBLE_DEVICES=0`, `LLAMA_ARG_FIT_TARGET=1024`, `LD_LIBRARY_PATH` prod (cf. RUN9B) |
| serveur | dédié port **44911** ; service prod stoppé **une fois** au début, restauré **une fois** à la fin |
| logprobs | NON activés en mesure (inutilisables sur ce binaire, §29.1-3) — texte seul comme référence de divergence |

## 4. Critères de validité (fixés AVANT lancement)

Un essai est **INVALIDE** si :
- **G_START** : serveur mort avant load, port non up, `fit_params: failed`/OOM,
  `model loaded` absent après 900 s, device ≠ `Tesla V100`.
- **G_WARM/PF** : warmup ou l'un des 2 prefills en erreur.
- **G_CTX** : `prompt_tokens` hors **[100 000, 115 000]** (attendu ≈ 107 351).
- **G_GEN** : `completion_tokens ≠ 1200` ou `finish_reason ≠ length`.
- **G_WALL** : mur du decode hors **[30 s, 900 s]**.
- **G_MTP** : aucune ligne `draft acceptance` avec generated > 0 ET aucun
  delta `spec common_specu` > 0 (log serveur fait foi — correction RUN9/RUN9B).
- **G_SHORTLIST** : `CUDA MTP shortlist` jamais vue dans le log.
- **G_ISOLATION** : V100 ≥ 1000 MiB au démarrage de l'essai.
- **G_LIB** : md5 de la lib testée ≠ attendu.

Fenêtre MTP : delta des 2 dernières lignes `spec common_specu: statistics`
(prefill2 → decode), comme RUN9B.

## 5. Protection production / supervision

- Le `.so` déployé (`a4b6f440`) n'est **jamais modifié** (vérifié avant/après).
  Sélection de lib par `GGML_BACKEND_PATH` uniquement.
- Service stoppé au début, `systemctl start` à la fin ; vérification
  `is-active` + API 11434. Tout incident de restauration → arrêt, signalement,
  aucune conclusion.
- Lancement : `cd bench && python3 supervisor_bench.py run ../h2/cupti129-test/nightrun8/run9c_bench.py --detach`
  (wrapper Python obligatoire, cf. `supervisor_bench_README.md`).
- Artefacts : `ab_run9c/` (results.jsonl, gen_*.json, server_*.out).
- **Durée estimée** : ~6 min/essai (load ~2 min + prefill1 ~190 s + prefill2
  chaud ~1 s + decode 1200 ~40 s) → **~40–50 min** au total.
- Surveillances pendant le run (agent, sans toucher au bench) : tail du log
  superviseur, `nvidia-smi -i 1`, `systemctl is-active ollama-v100`, et au
  moins une fois la lecture du bench en cours (aucun kill).

## 6. Ce que l'A/B NE PRÉTEND PAS

- n=2/bras (W4R4=1) et n=1/bras (W4R4=0) : exploratoire. Critère comparatif
  (cf. §4 protocole RUN9B) : écart inter-bras < max(écart-type intra-bras, 2 %)
  → INDÉCIS.
- Prompt filler (répétitif, anglais) = régime de §26.5, pas de la prod OWUI
  (français) ; les conclusions de performance se rapportent à **ce régime de
  contexte (~107k)**, pas à tout contexte.
- La divergence de texte (attendue, §29.3) n'est ni un bug ni une régression :
  associativité fp32 (cf. §26.5, §27). Elle est **documentée**, pas tolérée
  silencieusement.

## 7. Interprétations prévues (fixées AVANT, anti-biais)

- **Q1** : Δ H2 vs REF (W4R4=1) : ≤ −10 % → gain confirmé au régime 100k ;
  [−10 %, +2 %] → à investiguer (n_kv effective, MTP) ; > +2 % → H2 en retrait
  au régime 100k (surprenant vs §26.5 → re-vérifier les conditions).
- **Q2** : si REF_OFF ≫ REF_ON et H2_OFF ≈ H2_ON → le gain H2 tient sur le
  kernel fattn BC=8 seul (W4R4 indépendant) ; si les deux s'effondrent →
  l'interaction W4R4×fattn est en cause (à investiguer).
- **Q3** : divergence de texte REF vs H2 attendue (mécanisme fp32, §29.3) ;
  la comparabilité des murs reste valable (T=0, déterminisme intra-bras).
