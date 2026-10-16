# Protocole 03 — Reconstruire les libs H2 / H2b (build isolé, prod intouchée)

**Principe** : on ne modifie JAMAIS le `.so` déployé. Une lib de test =
**les 144 autres objets du build prod + 1 seul `fattn.cu.o` recompilé** avec
le patch. Le re-link est vérifié bit-à-bit (Gate A).

**Ce que le kit fournit** :
- `h2/e2-src/build/fattn-q8-volta.cuh.prod-36f564e2` — la source de référence (md5 36f564e2)
- `h2/patch-h2-fattn-q8-volta.cu.patch` — le patch H2 (2 lignes, vérifié : appliqué sur la source = md5 e6901c3c)
- `h2/patch-h2b-fattn-q8-volta.cu.patch` — le patch H2b (6 sites, vérifié : md5 2054bac0)
- `h2/e2-src/h2lib/compile_h2.sh` / `compile_control.sh` / `h2b_build_lib.sh` — les commandes nvcc EXACTES du build prod (déflags FA, arch sm_70, -O3)
- `h2/e2-src/h2lib/link_*.sh` + `*.rsp` (listes des 145 objets) — les commandes de link
- `bigfiles/libggml-cuda-h2.so` (a4b6f440) et `libggml-cuda-h2b.so` (0eb5afbd) — les libs déjà construites (pour A/B sans re-build)

**Ce qu'il faut en plus** (non fourni, voir NOTICE) : le fork complet
(`llama-volta-merged`, md5 du `fattn.cu` de prod = 36f564e2) et sa toolchain
(gcc 13.4 + nvcc 12.6.85) — un build complet « clean » du fork (~30 min)
fournit les 145 objets de référence.

## Séquence (vérifiée, c'est celle qui a produit les libs publiées)

```bash
export LLAMA_ROOT=/chemin/llama-experiments     # où est le fork + le build clean
E2=$LLAMA_ROOT/h2/cupti129-test/e2/h2lib        # (ou le kit)

# 0) Prérequis : un build « clean » du fork a produit 145 .o purs dans
#    build-clean/_deps/llama_cpp-build/ggml/src/ggml-cuda/  (liste dans objects_*.rsp)

# 1) GATE A — recette de link exacte : re-lien des 145 .o purs →
#    doit donner md5 e4afbb4a (le .so prod) — sinon la recette est fausse.
bash $E2/link_gate.sh

# 2) Copier la source de prod, appliquer le patch, recompiler fattn.cu SEUL
#    (commande nvcc exacte = compile_h2.sh, flags identiques au build prod)
cp $LLAMA_ROOT/.../ggml/src/ggml-cuda/fattn-q8-volta.cuh $E2/fork/...
patch -p1 < h2/patch-h2-fattn-q8-volta.cu.patch        # ou patch-h2b-...
bash $E2/compile_h2.sh                                  # → fattn.h2.o
#   ptxas : smem 25408 B / 166 regs (H2b) — contrôler la sortie -v

# 3) Re-link avec 144 .o purs + fattn.h2.o  (objects_h2.rsp = la liste)
bash $E2/link_h2.sh
#    → libggml-cuda-h2.so  md5 attendu a4b6f440 (H2) / 0eb5afbd (H2b)

# 4) Vérifications
md5sum $E2/lib/*.so
nm -D libggml-cuda-h2.so | wc -l    # 1254 symboles dynamiques = prod (aucun diff de nom)
readelf -d libggml-cuda-h2.so       # soname + NEEDED identiques à prod
grep -c "GGML_CUDA_VOLTA_Q6_W4R4" <symboles>   # marqueur W4R4 présent
```

## Gates de validation (ce qui a été fait sur les libs publiées)

| Gate | Contenu | Résultat |
|---|---|---|
| A — bit-identité | 145 .o purs re-liés = e4afbb4a | OK (le 13/10) |
| B — numérique | dumps pacc/pm/pl complets, 4 n_kv de part et d'autre du seuil 107 521, seed 12345 : H2b = prod **bit-identique** ; H2 diverge au seuil | OK |
| C — microbench | e1_bench (géométrie run6) : prod 1598 µs / H2 1508 / H2b 1350 (médianes, 5×30 iters) | OK |
| D — applicative | A/B run9c3 (protocole 02) : H2 −12,5 % @107k ; H2b = prod | OK |

## Numérique : pourquoi la divergence (et pourquoi H2b la supprime)

- `split_count(n_kv)` dérive du nombre de **tuiles** `ceil(n_kv/BC)` : à BC=8,
  le découpage en splits change au-delà de n_kv ≈ 107 520 (160→159 splits,
  frontières multiples de 8 au lieu de 16).
- `reduce_kernel` combine les splits partiels par additions fp32 : le
  **regroupement** change → 1 ULP → l'`argmax` peut basculer sur un token.
- H2b découple : `PC=16` (partitionnement = prod) + `BC=8` (tuile = H2) →
  **mêmes splits, mêmes frontières, même regroupement** → bit-identique,
  mais le gain applicatif (porté par le partitionnement) disparaît avec.

## Microbench standalone (sans le fork, sans le modèle)

Le harnais `h2/e2-src/e1_bench.cu` (+ `.cuh`) est **autonome** : kernel
attention copié, géométrie du decode run6 (n_kv=107 351, kvh=4, G=6, T=4,
D=256, grid 4×160, 128 threads), données aléatoires **seed fixe** → la sortie
du kernel est reproductible bit-à-bit sur la même V100/nvcc.

```bash
nvcc -O2 -arch=sm_70 -ccbin=g++-13 e1_bench.cu -o e1_bench
./e1_bench        # 5 passes × 30 iters, médiane — cf. bigfiles/bin-e1b_*
```

Les binaires `bin-e1b_*` (prod/base/h2/h2b/a3/a4) et `bin-e1c_*` (checker de
sortie) sont dans `bigfiles/` (sha256) si on veut contourner la compilation.
