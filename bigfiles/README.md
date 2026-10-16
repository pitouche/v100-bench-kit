# bigfiles — artefacts binaires du kit (hors git)

Les fichiers binaires du kit (libs + binaires microbench) ne sont pas versionnés
ici (273 Mo) : ils sont fournis par un canal séparé.

Vérification d'intégrité :

```bash
cd bigfiles && sha256sum -c bigfiles.sha256
```

Contenu (273 Mo) :

| Fichier | Md5 | Rôle |
|---|---|---|
| `libggml-cuda-h2.so` | a4b6f440 | lib prod + patch H2 (2 lignes, tuile/partitionnement BC=8) |
| `libggml-cuda-h2b.so` | 0eb5afbd | lib prod + patch H2b (6 sites, bit-identique à prod) |
| `bin-e1b_*` (7) | — | binaires microbench kernel attention (prod/base/h2/h2b/a3/a4) |
| `bin-e1c_*` (4) | — | checkers de sortie du microbench |
| `fattn.h2.o` / `fattn.h2b.o` / `fattn.control.o` | — | objets intermédiaires de build (traces) |
| `bin-devcheck` | — | probe d'ordre CUDA / nvidia-smi (le « piège » documenté) |

Tous ces binaires sont **recompilables** depuis les sources du kit
(`h2/e2-src/`, patches dans `h2/`) sur la machine de référence.
