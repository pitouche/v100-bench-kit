# Protocole 02 — A/B REF vs H2 : confirmer ou contester le résultat central

**Le résultat à (re)produire** : à ~107k tokens de contexte, la lib H2 fait
**−12,5 ± 0,5 % de wall** vs le .so prod (REF), gain quasi intégralement porté par
l'acceptation MTP (mean len 2,81 → 3,23). À ~41k : INDÉCIS (+2,1 %, dans le bruit).

**Le harnais de référence** est `h2/nightrun8/run9c3_bench.py` (version H5 :
gate d'erreur API en racine du JSON, sauvegarde systématique re-lue, mesure
canonique `eval time` du log serveur). Les protocoles des itérations sont dans
`h2/nightrun8/PROTOCOL-run9{b,c,c2,c3}.md`.

## 1. Ce que le protocole garantit (et comment)

| Menace | Contrôle |
|---|---|
| Dérive thermique / horloge | Ordre **alterné** REF/H2/REF/H2 (+ 1 paire W4R4 OFF), 2 réplicats ; spread intra-bras mesuré (< 1,5 % sur 3 campagnes) |
| GPU occupé par autre chose | Gate : `nvidia-smi memory.used < 1000 MiB` sur la V100 **avant chaque essai** (consigné dans results.jsonl) |
| Mauvaise lib chargée | Gate `G_LIB` : md5 de la lib == md5 attendu (e4afbb4a / a4b6f440) **avant** de démarrer le serveur |
| Mauvais GPU | Gate `G_START` : la ligne `using device CUDA…Tesla V100` du log serveur est obligatoire |
| Modèle mal chargé | Gate : `model loaded` + shortlist `CUDA MTP shortlist` présent dans le log |
| Erreur API masquée | Gate H1 : clé `error` en **racine** du JSON OU absence de `choices` → essai INVALID (le faux-positif RUN9C — sous-chaîne « error » dans le texte — est corrigé, cf. §30.1 de `ETAT-raw.md`) |
| Texte perdu | Gate H5 : chaque réponse (warm, prefill1/2, gen) est écrite **avant** tout gate et re-lue après écriture |
| Prod modifiée | Le `.so` déployé est md5-vérifié **avant et après** le run ; le bench ne le touche jamais (seule `GGML_BACKEND_PATH` du serveur de capture change) |
| Service pas restauré | Le bench stoppe/démarre le service dans tous les chemins (y compris en erreur) ; le superviseur (`bench/supervisor_bench.py`) est le filet de sécurité de 2ᵉ niveau |

## 2. Paramètres (figés, à ne pas changer sans nouveau protocole)

```
prompt      : h2/prompt1200.json (107 351 tok, sha256 c38fefee…) — 41k : voir PROTOCOL-run9b.md
decode      : 1200 tokens, temperature=0, top_k=1, ignore_eos
chauffe     : warmup 16 tok + prefill ×2 (2ᵉ passe cached ≈ 107 347 tok)
serveur     : llama-server direct, port 44911, -c 200000 -np 1,
              --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling,
              --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn auto -b 512 -ub 512,
              -lv 4, LLAMA_ARG_FIT_TARGET=1024
env         : GGML_CUDA_VOLTA_Q6_W4R4=1 (essais 1-4) / =0 (essais 5-6),
              GGML_CUDA_QWEN35_MTP_SHORTLIST=<shortlist>, LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1,
              CUDA_VISIBLE_DEVICES=0 (ordre CUDA de la machine de référence — à vérifier)
mesure      : ligne 'eval time = X ms / 1200 tokens' (task de génération) du log serveur
MTP         : delta des 2 dernières lignes 'spec common_specu' (compteurs cumulatifs par process)
```

## 3. Lecture des résultats (règle de décision du protocole)

- **INDÉCIS** si |Δ inter-bras| < max(écart-type intra-bras, 2 %).
- **Gagnant** sinon ; le gain se décompose **obligatoirement** en
  `Δwall = Δcoût/round × Δnombre de rounds` (le §30.3 de `ETAT-raw.md` le fait :
  −13,3 % rounds × +1,75 %/round = −11,77 % ≈ Δ mesuré).
- Le **texte** doit être comparé (premier point de divergence + ratio difflib) —
  la divergence REF/H2 est **attendue** (associativité fp32, §HISTORIQUE 4.4) ;
  ce qui serait un signal d'alerte = une **régression de contenu** (fait erroné,
  JSON invalide, instruction non respectée), pas la divergence elle-même.

## 4. Recommandations pour contester

1. **Reproduire d'abord le run9c3 tel quel** (les artefacts sont là :
   `h2/nightrun8/ab_run9c3/` — on peut recalculer les Δ sans re-lancer).
2. **Contenu réel** : les A/B sont sur prompt filler. Si l'objectif est de
   contester le gain sur de la prod réelle, remplacer `PROMPT_FILE` par un
   prompt long (~100k) représentatif — le protocole reste valable, mais le
   niveau d'acceptation (donc le mur par cycle) changera ; renormaliser par
   `1000 × mean_len / tps`.
3. **Contextes intermédiaires** : 41k (INDÉCIS) et 107k (−12,5 %) bornent le
   phénomène ; le seuil exact n'est **pas** déterminé (l'hypothèse « seuil de
   split_count ~107 520 » est INCONFIRMÉE à 41k, un 2ᵉ mécanisme existe —
   §29.3 de `ETAT-raw.md`). Un sweep 60k/80k/100k trancherait.
4. **Verrouiller les clocks** (non fait dans les campagnes publiées) pour
   réduire le bruit < 1,5 % — si disponible sur la machine.

## 5. Checklist post-run (à consigner)

- [ ] rc=0 du bench ; `supervisor_bench.log` : `=== FIN du run : rc=0 ===`
- [ ] `systemctl is-active ollama-v100` = active
- [ ] md5 du `.so` déployé inchangé
- [ ] 6/6 essais OK (aucun INVALID dans results.jsonl)
- [ ] 6 réponses `gen_*.json` présentes et non vides
- [ ] MTP actif sur les 6 (lignes `draft acceptance` dans les server.out)
- [ ] Δ + décomposition rounds/coût/round + comparaison texte consignés
