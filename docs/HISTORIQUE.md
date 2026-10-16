# HISTORIQUE technique — V100 (3 → 16 octobre 2026)

Document d'historicité **honnête** : ce qui est validé, ce qui est expérimental,
les limites connues, et les décisions — avec les données brutes qui permettent à
quiconque de contester ou confirmer.

**Lecture** : §2 = l'état des lieux, §3 = les faits validés (les seuls chiffres à
retentionner en confiance), §4 = l'unique optimisation déployée (H2) en détail,
§5 = ce qui reste expérimental, §6 = limites des mesures (à lire AVANT de contester),
§7 = pistes fermées, §8 = comment reproduire.

---

## 1. Principe directeur (invariants de tous les essais)

1. **La prod n'est jamais modifiée par un test.** Le `.so` déployé (md5), l'unité
   systemd et le GGUF sont vérifiés **identiques avant et après** chaque run (md5
   consigné dans les logs). Un A/B ne change qu'une seule variable :
   `GGML_BACKEND_PATH` sur un **serveur de capture** dédié (port 44901/44911),
   pendant que la prod est stoppée puis restaurée (trap, jamais de « oublier »).
2. **Les seuils de décision sont fixés AVANT la mesure** (ex. : H2b §4.3, audit §7).
3. **La mesure canonique** est la ligne `eval time` du log serveur (pas le mur HTTP).
4. **Chaque run publie ses données brutes** (server.out, réponses JSON, TSV) dans ce
   repo, avec checksums — une conclusion sans données brutes n'a pas de valeur ici.

## 2. État des lieux (fin de période, 16/10)

| Élément | Valeur |
|---|---|
| `.so` déployé | `libggml-cuda-h2.so` **md5 a4b6f440** (= prod known-good + patch H2, 1 fichier recompilé) |
| known-good de référence | **md5 e4afbb4a** (build 2026-10-04, backup `/var/backups/…`) |
| Rollback | `deploy.sh prod` → restauration ~10 s (testée) |
| Service | `ollama-v100` actif, port 11434, API OK |
| Perf de référence (mesurée, long contexte ~107k) | **30,0 → 33,8 tok/s** après H2 (−12,5 % wall) ; acceptation MTP 2,81 → 3,23 |
| Perf courte (< 50k) | ≈ identique (H2 INDÉCIS @41k, +2,1 % dans le bruit) |

## 3. Faits validés (chiffres à confiance)

### 3.1 Le régime du decode long contexte est mem-bound sur les poids (P3, autopsie)
- Débit DRAM des poids **atteint à 84 %** (757/900 GB/s sur lm_head) ; ALU/dp4a ~0.
- ~50 % du cycle (~30–43 ms sur 76 ms) initialement non attribué → c'est la question
  que H2 a traitée (§4.1).
- Données : `autopsy/ETAT.md` (P3), `h2/ETAT-raw.md` §13.

### 3.2 Le decode MTP est quasi saturé GPU (run7 CUPTI v2, 16/10)
- **Occupation GPU 96,11 %** (union des intervalles CUPTI validés), idle 70,7 ms
  structuré (gaps ~1 ms post-`k_get_rows`/`k_set_rows` = sync MTP + ~40 ms de
  micro-gaps de lancement). **Overlap inter-streams = 0 ms** : les 2 streams
  s'exécutent séquentiellement.
- Conséquence : le gain « pipeline » par chevauchement est **nul** ; le seul levier
  est de réduire le temps GPU sérial ou les gaps CPU.
- Données : `h2/ETAT-raw.md` §25 (gates G1–G5, écart union/wall 0,55 %).

### 3.3 Le gain H2 est bien réel et reproductible à ~107k (3 campagnes, 3 ordres)
- **−12,5 ± 0,5 % wall** (RUN9C, RUN9C2, RUN9C3 : spread < 1,5 % par essai,
  W4R4 ON et OFF).
- **Le gain est quasi intégralement un gain d'acceptation MTP** : rounds −13,3 %
  (mean len 2,81 → 3,23 ; pos2 0,576 → 0,721, pos3 0,422 → 0,633) × coût/round +1,75 %
  (le verify H2 est légèrement plus lourd par round) = −11,77 % (≈ Δ mesuré).
- **Déterminisme** : T=0, bit-identique intra-bras (REF et H2) ET **cross-run**
  (h2a1 RUN9C ≡ h2a1 RUN9C3, ~4 h d'écart, service redémarré).
- **Le gain est lié à la longueur de contexte** : @41k H2 ≈ REF (+2,1 %, INDÉCIS) ;
  @107k H2 −12,5 %. Cohérent avec le mécanisme de partitionnement de l'attention
  (cf. §4.4 pour la nuance).
- Données : `h2/nightrun8/ab_run9c3/` (6/6 essais, logs serveurs, réponses JSON),
  `h2/ETAT-raw.md` §29–32.

### 3.4 Le coût KV-q8_0 est le 2ᵉ poste du decode (audit 12/10)
- `ggml_q8v14partial` = **29,8 %** du temps GPU cumulé (pas 17,5 % comme rapporté
  initialement — incohérence trouvée par la revue indépendante, recalculé sur `dur_ns`
  validées). + `quantize_q8_1` 1,8 % ≈ **32 % cumulé** : le cache q8_0 (gain bande
  passante) a un coût fixe de requantisation non négligeable par round.
- Données : `h2/ETAT-audit-v100-12-10.md` §3, `h2/ETAT-cupath-revue-12-10.md`.

### 3.5 W4R4 Q6_K est bénéfique et indépendant de H2
- ON : **+4,1 % REF / +3,3 % H2** (n=1/bras en OFF → indicatif). Texte **identique**
  ON/OFF (la variable quant n'affecte ni la séquence ni le MTP, seulement la vitesse).
- Données : `h2/ETAT-raw.md` §30.2, §32.3.

### 3.6 nsys / CUPTI 13.x ne fonctionnent pas sur Volta/driver 580
- Verdict CLOS (CUPTI 13.0.85/13.1, nsys 2025.3.2/2025.5.2) : `CUPTI_ERROR_INVALID_DEVICE`.
- **CUPTI 12.9 fonctionne** (shim LD_PRELOAD isolé, `h2/cupath_shim*.c`) : c'est
  l'outil d'observation de ce kit. ncu 2024.1.1 fonctionne aussi.
- Données : `h2/ETAT-raw.md` §1 + §15, `h2/PROTOCOLE.md`.

## 4. H2 — l'unique optimisation déployée (détail)

### 4.1 Contexte
~50 % du cycle MTP long contexte non attribué après l'autopsie P0–P7a. H2 = localiser
puis réduire ce temps. Verdict final de la localisation : **pas de trou GPU massif**
(§3.2), **pas de gain par chevauchement** (overlap = 0), **pas de gain par sync**
(H-SYNC infirmée, 11 % du round). → Le levier restant : le temps GPU sérial.

### 4.2 Ce que change H2 (2 lignes de code, `ggml/src/ggml-cuda/fattn-q8-volta.cuh`)
- `constexpr int BC = 16;` → `8` (taille de tuile des clés KV)
- `for (int sub=0;sub<2;sub++)` → `sub<BC/8` (sous-tuiles dérivées de BC)
- Effet microbench : **−15,5 %** sur le `partial_kernel` isolé (1596 → 1349 µs,
  NCU : smem 33,9 Ko → 25,4 Ko, occupation 12,5 % → 18,75 %, spills 152 B → 0).
- Patch : `h2/patch-h2-fattn-q8-volta.cu.patch` (vérifiable sur la source prod
  md5 36f564e2 fournie dans `h2/e2-src/build/`).

### 4.3 H2b — la variante « propre » (découplage tuile/partitionnement)
- Problème de H2 : `split_count(n_kv)` dépend de BC → au-delà de n_kv ≈ 107 520,
  le découpage des splits change → l'associativité fp32 de `reduce_kernel` change →
  **le texte diverge** (1 ULP, contenu correct).
- H2b : partitionnement forcé à PC=16 (= prod) en gardant la tuile BC=8
  (`h2/patch-h2b-fattn-q8-volta.cu.patch`, 6 sites).
- **Résultats** : (a) **bit-identique à prod** partout (4 n_kv de part et d'autre du
  seuil, dumps complets md5-compilés) ; (b) kernel isolé **aussi rapide que H2**
  (1350 µs) ; (c) **mais A/B applicative : H2b = prod (+0,6 %)** — le gain applicatif
  de H2 vient du **partitionnement** BC=8 (meilleure acceptation MTP), pas de la tuile.
- **Conclusion** : le gain de H2 est indissociable de la divergence textuelle. H2b
  existe comme variante de déploiement sans risque de changement de texte (aucun gain).
- Données : `h2/ETAT-raw.md` §26.5–27.10, `h2/ab_results_h2b_v1/`, `ab_results_h2b_control/`.

### 4.4 La divergence textuelle — ce qu'elle est (et ce qu'elle n'est pas)
- **Factuel** : à T=0, REF et H2 produisent des séquences différentes (première
  divergence @41k : token 190 ; @107k : char ~1086 sur prompt filler).
- **Nature** : changement d'associativité fp32 dans `reduce_kernel` (regroupement
  des splits). Ce n'est **pas** une perte de précision de l'attention (les deux
  sorties sont exactes à un ULP près, c'est l'ordre d'addition qui change).
- **Évaluation qualité** (suite 11 prompts : factuels vérifiables, raisonnement,
  JSON strict, needle-in-haystack 41k, multi-tours) : **aucune régression
  macroscopique** ; les divergences sont des reformulations correctes.
- **Ce que ça ne prouve PAS** : l'équivalence formelle des garanties MTP. Un
  second mécanisme de sensibilité à n_kv < 107 521 existe (non caractérisé —
  dump pacc/pm/pl à 41k non fait).
- **Décision** : H2 déployée en **production expérimentale** (l'utilisateur a
  accepté le changement de texte en échange du gain) ; rollback en 1 commande,
  surveillance des contextes > 107,5k (le régime du gain).

## 5. Expérimental / non consolidé

| Élément | Statut | Pourquoi pas « validé » |
|---|---|---|
| Gain H2 sur du **contenu réel** (français, conversations OWUI) | Non mesuré | Tous les A/B sont sur prompt filler (récit répéteur) — régime de mesure, pas de la prod. Le gain est mesuré en ms/tok (indépendant du contenu), mais l'acceptation MTP varie avec le contenu. |
| W4R4 OFF (n=1/bras) | Indicatif | Pas de puissance statistique sur l'OFF ; le ON est l'état par défaut. |
| H2 @41k « INDÉCIS » | Non tranché | +2,1 % dans le bruit (écart intra-bras ~1 %) — ni gain ni perte significative démontrée. |
| Le ~40 ms de micro-gaps (run7) | Non réduit | Caractérisé (latence de lancement, 41k gaps médiane 0,64 µs) mais aucun levier identifié au-dessus du bruit. |
| `p5_ktrace.so` (interpose CUDA) | Outil de veille | Invisible en replay CUDA graph — non utilisé pour les verdicts. |
| Patch Go `speculative_n_max` par requête | Staging | Prêt, jamais déployé (prochain build). |

## 6. Limites des mesures (à lire avant de contester)

1. **Prompt filler répéteur** : acceptance MTP élevée (~0,8–1,0) vs prod réelle
   (~0,6–0,8) ; le mur par cycle est renormalisé par `1000 × mean_len / tps`.
   La **structure** (phases, familles de kernels, gaps) est indépendante du contenu,
   le **niveau absolu** dépend du contenu.
2. **V100 partagée sur certaines missions** (H2-PROD §28) : des mesures y sont
   **entachées** (concurrency GPU non contrôlée) et marquées comme telles dans
   `ETAT-raw.md` §28.4. Les A/B de référence (run9b/c/c2/c3) sont en **session
   isolée** (V100 < 1000 MiB avant chaque essai, vérifié dans les logs).
3. **Compteurs MTP cumulatifs par processus**, réinitialisés au rechargement du
   modèle : la fenêtre de mesure = **delta des 2 dernières lignes** `spec
   common_specu` (documenté, implémenté dans le harnais de référence).
4. **CUPTI : durées valides, timestamps non validés** (avant run7) : les affirmations
   d'« occupation » ne valent qu'avec le shim v2 (run7, gates G1–G5). Le §15.1
   (premier rapport) contenait des incohérences internes (us/rd faux, % sous-estimés)
   — corrigées par la revue §3 de l'audit ; **ne pas citer le §15.1 tel quel**.
5. **Pas de verrouillage de clocks GPU** pendant les A/B : le bruit réplicats mesuré
   est < 1,5 % par essai (3 campagnes) — considéré comme suffisant, documenté ici
   pour quiconque veut un protocole encore plus serré.
6. **Le mur HTTP du bench n'est PAS la mesure** : c'est la ligne `eval time` du log
   serveur (le mur HTTP inclut le transport + le sampling ; un bug de harnais de
   RUN9C l'a démontré — §30.1, cause racine documentée : le gate `error` matchait
   la sous-chaîne dans le texte généré).
7. **Toutes les mesures sont mono-machine** (V100-PCIE-32GB, driver 580.178.04,
   CUDA 12.6, gcc 13.4, nvcc 12.6.85). La reproductibilité **inter-machine** n'est
   pas démontrée ; les chiffres sont des **régimes**, pas des absolus.

## 7. Pistes fermées (ne pas rouvrir sans nouveau signal)

| Piste | Verdict | Preuve |
|---|---|---|
| QPN / Tensor Core pour le decode T=4 | NO-GO (port byte-exact 2,7× plus lent que NInfer, régime PCIe/L2 ; TC > dp4a seulement à T≥16) | `autopsy/ETAT.md` P0–P2 |
| Kernel Q6_K par patch local (P7/P7b) | Clôturé 8/10 : +1,1 % < seuil 2 ms ; le déficit 495 vs 692 GB/s est **structurel** (VDR=1, dp4a émulé, 96 regs) | `p7/bench/ETAT.md` |
| ncols=8 (B8) en prod | Structurellement inatteignable (MTP 3+1, -np 1 → ne1∈{1,4}) | `autopsy/ETAT.md` P6 |
| H-SYNC (syncs/launches dominants) | Infirmée (11 % du round) | `h2/ETAT-raw.md` §13 |
| Gain par chevauchement inter-streams | Null (overlap = 0 ms) | `h2/ETAT-raw.md` §25 |
| nsys / CUPTI 13.x sur Volta | CLOS (driver 580) | `h2/PROTOCOLE.md`, §1 ETAT-raw |
| torch.compile / Triton / XLA | Non applicable (llama.cpp, pas PyTorch) | `h2/ETAT-audit-v100-12-10.md` |
| TensorRT / vLLM / TGI | Non applicable (modèle hybrid SSM+FA + MTP + kernels custom) | idem |
| AWQ / GPTQ / FP8 / INT4-TC | Autre famille / non supporté sur sm_70 ; l'effet utile (octets) est déjà H-BW | idem |
| H2c (décendance H2b2, H2c…) | Aucune ouverte sans nouvelle décision | §27.8 ETAT-raw |

## 8. Reproduction — ce qui est nécessaire

| Besoin | Fourni dans le kit ? |
|---|---|
| Prompts des A/B (107k / 41k) | ✅ `h2/prompt1200.json`, `h2/prefill-107k.json` (sha256 dans `h2/prompts.sha256`) |
| Libs REF / H2 / H2b | ✅ `bigfiles/` + sha256 (hors git : 130 Mo chacune) ; REF = backup prod à créer chez vous |
| Patches source | ✅ `h2/patch-h2-*.patch` + source prod `fattn-q8-volta.cuh` (md5 36f564e2) |
| Harnais A/B de référence | ✅ `h2/nightrun8/run9c3_bench.py` (+ protocoles run9b/c/c2/c3) |
| Microbench kernel attention | ✅ `h2/e2-src/` (e1_bench.cu/cuh, géométrie run6, seeds fixés) |
| Trace CUPTI | ✅ `h2/cupath_shim_v2.c` + `run7_v2.sh` (build ~2 min, CUPTI 12.9 en prefix isolé) |
| Fork complet (llama-volta-merged) | ❌ **Non fourni** — voir `NOTICE.md` (licence MIT, base b11081 + merge v100-optimized) ; le kit ne contient que ce qui est nécessaire aux A/B |
| Modèle qwen3.8-27b-q6-mtp | ❌ Non fourni (23 Go) — à re-quantiser (Unsloth Q6_K + MTP draft) |
| Shortlist MTP (131 072 i32) | ❌ Non fourni — à régénérer (script dans le fork, data/mtp-shortlists) |

**Le minimum reproductible sans le fork ni le modèle** : le microbench `e1_bench`
(kernel attention isolé, données aléatoires seed fixe → reproductible bit-à-bit
de la sortie du kernel), et l'analyse des traces CUPTI (les scripts + les
données de run7 sont là).

**Le reproductible complet** (A/B applicatif) requiert : une V100 32 Go, le fork
(§NOTICE), le modèle, le shortlist — et ~2 h de montage (build ~30 min + 6 essais).
