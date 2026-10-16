# REQUIREMENTS — prérequis, dépendances, pins

Pour reproduire les essais sur une V100.

## 1. Machine (minimum)

| Élément | Valeur de référence | Remarque |
|---|---|---|
| GPU | **Tesla V100-PCIE-32GB** (sm_70, Volta) | Les résultats ne se transposent pas à d'autres archs (kernels custom sm_70) |
| VRAM libre | ≥ 32 Go | Le modèle + KV q8_0 ctx 200k + MTP occupe ~31,5 Go |
| CPU | ≥ 8 cœurs | Le bench et le build en profitent (`JOBS`) |
| RAM | ≥ 30 Go | Build + serveurs de capture |
| OS | Linux x86_64 (tests sur Ubuntu 24.04) | systemd requis (service `ollama-v100`) |
| Disque | ≥ 60 Go libres | Modèle (23 Go) + build (30 Go) + artefacts |

> **⚠️ Piège documenté** : l'ordre `nvidia-smi -L` ≠ l'ordre CUDA. Sur la machine
> de référence : `nvidia-smi` 0=V100, 1=1070, mais **CUDA cuda:0=V100, cuda:1=1070**
> (inversé). Toujours vérifier avec un probe `cuDeviceGet*` avant un run — le
> fit est serré (58 MiB) et un GPU erroné = run qui « réussit » sur la mauvaise carte.

## 2. Ollama / service prod

- Ollama **v0.34.4-5** (commit pin `16b4376a…`) installé et servant sur le
  service systemd `ollama-v100` (port 11434).
- Le service doit être **stoppable** par l'utilisateur (`sudo systemctl
  stop ollama-v100`) sans mot de passe (sudoers) — c'est ce que font les A/B.
- Si vous n'avez pas encore le service : le kit `v100-repro` (machine de ref.)
  le construit depuis zéro (build ~30–60 min, pins dans `manifest.txt`).

## 3. Toolchain (pins de référence)

| Outil | Version | Où |
|---|---|---|
| cmake | 3.31.6 | `tools/cmake/bin/cmake` |
| gcc / g++ | 13.4.0 | `tools/gcc-13-bin/` (pas le gcc système v15) |
| CUDA toolkit | 12.6.85 | `tools/cuda-12.6/usr/local/cuda-12.6` |
| go | 1.26.x | `/usr/bin/go` |
| Nsight Compute | 2024.1.1 | `~/ncu-2024.1.1/` (mesures NCU de `e2/`) |
| python3 | ≥ 3.10 | système. Dépendances tier : **requests** (superviseur, fixture), **PIL/Pillow** (vision synthétique auto_bench), **matplotlib** (plot_bench). Le reste est stdlib. |
| nvidia driver | ≥ 575 (tests : 580.178.04) | `nvidia-smi` |

> Pourquoi pas le gcc système ? Le build de référence est fait avec gcc 13.4 +
> nvcc 12.6 (combo validée, headers CUDA 12.6, binaire Go stable). Le script
> `build_all.sh` (kit v100-repro) émet un WARNING si les versions diffèrent.

## 4. CUPTI / Nsight (observation)

- **CUPTI 13.x et nsys 2025.x NE TROUVENT PAS sur Volta/driver 580**
  (`CUPTI_ERROR_INVALID_DEVICE`) — verdict documenté, ne pas essayer.
- **CUPTI 12.9 fonctionne** : à installer en **prefix isolé** (ne pas polluer le
  système). Le shim `h2/cupath_shim_v2.c` se compile avec nvcc 12.6 et a un
  `rpath` vers le prefix CUPTI 12.9 (cf. `h2/run7_v2.sh`).
- **Nsight Compute 2024.1.1** fonctionne (mesures NCU de l'étape E2).
- **Aucune lib CUDA/NVIDIA n'est redistribuée par ce kit** (cf. `NOTICE.md` §4).

## 5. Modèle + shortlist (à fournir chez vous)

- **Modèle** `qwen3.8-27b-q6-mtp` : GGUF de **23 088 409 504 octets**, sha256
  `493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb`.
  À (re)quantiser : base Qwen3.8-27B → Unsloth **Q6_K_M** + couche MTP
  (draft-mtp n_max=3, `nextn_predict_layers=1`).
- **Shortlist MTP** `qwen38-27b-exact-131072.i32` : 131 072 lignes, couverture
  99,2 %. À régénérer depuis le fork (`data/mtp-shortlists/`).
- **Filler 200k** (`v100-repro/work/filler-200k.txt`, 1,1 Mo) : texte répéteur
  pour les benchs de préfill — à reconstruire si absent (n'importe quel texte
  ~200k tok convient pour le régime, les chiffres absolus dépendront du contenu).

## 6. Variables d'environnement (voir `config/kit.env.example`)

Tous les chemins machine-dépendants sont lus de l'environnement (les défauts
sont ceux de la machine de référence). Minimum pour un A/B :

```bash
export LLAMA_ROOT=/chemin/vers/llama-experiments   # racine du service ollama-v100
export MODEL_GGUF=/chemin/vers/le/blob/gguf
export REF_LIB=/chemin/vers/backup/.so.prod        # le .so known-good (md5 e4afbb4a)
export H2_LIB=$KIT/bigfiles/libggml-cuda-h2.so
export OLLAMA_SERVICE=ollama-v100
export V100_SM_INDEX=<index nvidia-smi de la V100>
```

## 7. Ce qui n'est PAS un prérequis (écarté par design)

- **PyTorch / Triton / torch.compile** : le chemin est llama.cpp, pas PyTorch.
- **TensorRT / vLLM / TGI** : non applicable (modèle hybrid + MTP + kernels custom).
- **MPS** : decode mono-process, pas de contention multi-process à résoudre.

## 8. Vérification rapide avant un premier run

```bash
# 1) La V100 est visible et libre
nvidia-smi -i $V100_SM_INDEX --query-gpu=name,memory.free --format=csv

# 2) Le service prod est actif et sain
systemctl is-active $OLLAMA_SERVICE
md5sum $LLAMA_ROOT/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so

# 3) Le modèle est là et de la bonne taille
stat -c '%s' $MODEL_GGUF   # doit donner 23088409504

# 4) Le .so de test est présent
md5sum $H2_LIB   # doit commencer par a4b6f440
```
