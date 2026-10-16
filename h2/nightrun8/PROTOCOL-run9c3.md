# RUN9C3 — re-run final A/B REF vs H2 @107k, harnais définitif : réponses gen sauvegardées (v3)

**Écrit AVANT tout lancement** (2026-10-15, ~21:55 UTC). Suite de RUN9C
(§30) et RUN9C2. Objectif unique du re-run : **sauvegarder les 6 réponses gen**
pour la vraie comparaison textuelle REF vs H2 demandée.

## 0. Pourquoi un 3ᵉ run

- **RUN9C** : gate `"error" in json.dumps(...)` = faux-positif → les 3 essais
  REF invalidés à tort, `gen_ref*.json` jamais écrits. Mesure serveur valide
  (§30.2).
- **RUN9C2** (harnais corrigé H1/H3/H4) : **6/6 OK**, performance reproduite
  (eval_ms REF 40 455/40 212/41 832 · H2 35 670/35 558/36 254 ; MTP 771/1281
  vs 828/1112). **Mais** `gen_*.json` absents : 2ᵉ bug du harnais — la
  sauvegarde gen passait un chemin **absolu** à `save()` qui préfixait
  `OUTDIR` → chemin doublé, `[Errno 2]` (voir log superviseur, les 6 lignes
  `!! save ... echec`). Les sauvegardes warm/prefill (noms relatifs) ont
  fonctionné. Le texte gen n'est **pas** dans les logs serveurs (vérifié :
  uniquement des lignes de log) → perdu, re-run nécessaire.

## 1. Conditions — identiques à PROTOCOL-run9c.md / run9c2

Mêmes 6 essais (ref1/h2a1/ref2/h2a2 W4R4=1 + ref3/h2a3 W4R4=0), même prompt
filler 107 351 tok, decode 1200, T=0, mêmes flags/env/port/ordre/chauffe,
mêmes gates de validité (G_CTX/G_GEN/G_WALL/G_MTP/G_SHORTLIST/G_ISOLATION/
G_LIB) et même détection d'erreur H1 (clé `error` en racine OU absence de
`choices`). Sortie : `ab_run9c3/`.

## 2. Corrections H5 (définitives, fixées AVANT lancement)

- **H5a — chemin de sauvegarde** : `save()` reçoit toujours un **nom relatif**
  (`gen_{tag}.json`) ; le chemin absolu ne figure que dans le champ
  `gen_file` du record (métadonnée, non utilisée pour l'écriture).
- **H5b — self-test au démarrage** : avant de stopper le service, le bench
  écrit puis relit `_selftest.json` dans `OUTDIR` ; échec → arrêt immédiat
  (service non stoppé), rc=1.
- **H5c — gate G_SAVE** : après `save(gen…)`, vérification
  `os.path.exists` + taille > 0 + relecture JSON → `gen_saved: true/false`.
  `gen_saved == false` ⇒ essai **INVALID (G_SAVE)** : l'objectif du run est
  la sauvegarde du texte, un essai sans texte est inutile.

## 3. Analyse post-run

1. 6/6 OK **et** `gen_saved == true` (sinon incident signalé).
2. Déterminisme intra-bras : ref1 vs ref2, h2a1 vs h2a2 (texte + MTP), et
   **cross-run** h2a3 vs h2a1-RUN9C (déjà bit-identique entre RUN9C).
3. **Comparaison textuelle REF vs H2** (W4R4=1 et =0) : identité, longueur du
   préfixe commun, première divergence (position + contexte ± 200 chars),
   recouvrement difflib, occurrences du mot « error » dans chaque texte
   (vérification définitive de l'hypothèse du faux-positif RUN9C), lecture du
   contenu.
4. Performance : confirmation RUN9C/RUN9C2 (table eval_ms, écarts).
5. Rapport + §31 (RUN9C2) + §32 (RUN9C3) dans `h2/ETAT.md`, mise à jour note
   canonique RUN9C.

## 4. Protection production

Identique RUN9C : `.so` déployé jamais modifié (vérifié avant/après), service
stoppé/restauré une fois, superviseur `--detach`, arrêt propre + signalement
si incident. Durée estimée ~25–30 min.
