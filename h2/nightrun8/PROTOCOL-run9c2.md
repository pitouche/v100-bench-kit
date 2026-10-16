# RUN9C2 — re-run RUN9C avec harnais corrigé (v2), objectif : réponses sauvegardées + vraie comparaison textuelle

**Écrit AVANT tout lancement** (2026-10-15, ~21:00 UTC). Suite immédiate de RUN9C
(`PROTOCOL-run9c.md`, §30). Les conditions de l'A/B sont **inchangées** ; seuls
le harnais et la mesure sont corrigés, comme documenté dans §30.5-4.

## 0. Objectifs

1. **Vraie comparaison textuelle REF vs H2 @107k** : le RUN9C n'a pas pu
   sauvegarder les réponses REF (bug du harnais, §30.1). Re-rouler les 6 essais
   avec un harnais qui **sauvegarde systématiquement** chaque réponse (warmup,
   prefills, gen) → `gen_*.json` des 6 essais → comparaison texte REF vs H2
   (première divergence, recouvrement, contenu) à ~107k, W4R4=1 et W4R4=0.
2. **Ré-vérifier la performance** avec le pipeline de mesure corrigé
   (mesure serveur canonique) et confirmer les écarts RUN9C
   (H2 ≈ −12 % @107k, W4R4 OFF ≈ +4 %/bras, MTP acc 771/828).
3. **Comprendre le faux-positif** : vérifier l'hypothèse « le texte de
   continuation REF contient le mot *error* » (0 occurrence dans le filler et
   dans les 3 textes H2 sauvegardés ; le gate RUN9C cherchait la sous-chaîne
   dans tout le JSON).

## 1. Conditions — identiques à PROTOCOL-run9c.md

Mêmes 6 essais (ref1/h2a1/ref2/h2a2 W4R4=1 + ref3/h2a3 W4R4=0), même prompt
filler 550 000 chars = 107 351 tok, même decode 1200 tok T=0, mêmes flags/env,
même port 44911, même ordre alterné, même chauffe (warmup 16 + 2 prefills),
mêmes gates (G_CTX [100k,115k], G_GEN 1200, G_WALL [30,900] s sur le mur HTTP,
G_MTP log serveur, G_SHORTLIST, G_ISOLATION, G_LIB). Seule différence : le
harnais `run9c2_bench.py` (ci-dessous). Sortie : `ab_run9c2/`.

## 2. Corrections du harnais (documentées, fixées AVANT lancement)

- **H1 — détection d'erreur API robuste** : plus de sous-chaîne. Une réponse
  est une erreur si et seulement si la clé `error` est présente **en racine du
  JSON** OU si `choices` est absent. (Réponse de succès vérifiée : clés racine
  `choices/created/model/system_fingerprint/object/usage/id/timings`, pas de
  `error`.)
- **H2 — sauvegarde systématique** : chaque réponse (warmup, prefill1, prefill2,
  gen) est écrite sur disque **avant** tout gate → aucun texte perdu, même si
  un gate déclenche.
- **H3 — mesure serveur canonique** : en plus du mur HTTP, extraction de la
  ligne `eval time = X ms / 1200 tokens` du log serveur (task de génération)
  → `eval_ms`, `eval_ms_per_tok` dans le record (mesure uniforme, cf. §30.3).
- **H4 — MTP** : le delta des `calls` utilise le champ *g* (calls gén), pas *b*
  (batch) du tuple `#calls(b,g,a)` → `calls` et `mean_acc_len` cohérents.
- **G_WALL** reste sur le mur HTTP (protocole §4 inchangé) ; la mesure canonique
  est le champ `eval_ms`.

## 3. Analyse post-run (à faire après le run)

1. Validité des 6 essais (tous doivent être OK — sinon arrêter et rapporter).
2. Performance : table ms/tok (eval_ms canonique) par bras, écarts vs RUN9C.
3. **Comparaison textuelle** (`compare_texts.py`) : REF vs H2 (W4R4=1) et
   REF vs H2 (W4R4=0) — identité bit-à-bit, longueur du préfixe commun,
   première divergence (position + contexte), recouvrement (difflib),
   occurrences du mot « error » dans chaque texte (vérification de l'hypothèse
   du faux-positif RUN9C), et lecture du contenu (cohérence, régression ?).
4. Déterminisme intra-bras : ref1 vs ref2, h2a1 vs h2a2, etc.
5. Rapport + §31 dans `h2/ETAT.md` + mise à jour de la note RUN9C.

## 4. Protection production

Identique à RUN9C : `.so` déployé jamais modifié (vérifié avant/après),
service stoppé/restauré une fois, superviseur `--detach`, arrêt propre +
signalement si incident. Durée estimée ~30–35 min.
