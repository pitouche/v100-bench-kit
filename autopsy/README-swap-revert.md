# swap_revert.sh — swap + rollback du .so CUDA (prod ollama-v100)

## Ce que ça protège
On ne touche **qu'au .so** `cuda_v12/libggml-cuda.so`. Le serveur `ollama` (systemd),
le binaire `ollama`, et le GGUF (23 Go) **ne bougent jamais** dans ce script.
Chaque opération **backe-up le .so en place avant de l'écraser** et **vérifie le hash
après copie**. Le known-good est stocké hors de l'arbre prod :
`reference/libggml-cuda.so.e4afbb4a.known-good` (md5 `e4afbb4a`, sha256 `55bc0a96…`).

## Commandes
```bash
cd /home/guile/llama-experiments/autopsy-20261005/q8-qpn

./swap_revert.sh status                 # hash en place vs known-good, service, VRAM, API   (pas de root)
sudo ./swap_revert.sh swap /chemin/libggml-cuda.so     # installer un .so de test
sudo ./swap_revert.sh revert             # RENTRER EN ARRIÈRE → known-good (rollback unique)
./swap_revert.sh health [--load]         # API + VRAM (+ chargement forcé du modèle 23Go)
```

## Flow `swap`
`stop service` → attend VRAM V100 libre → **backup .so en place** → copie du nouveau .so
→ **vérifie sha256** → `start service` → health.
Si le nouveau .so ne matche pas → ABORT (service redémarré sur l'ancien .so intact).

## Flow `revert` (le bouton de dépannage)
`stop service` → **backup .so en place** (le build de test est préservé) →
**restaure le known-good** (hash vérifié avant ET après copie) → `start service` → health.
Si le .so en place est déjà le known-good → **no-op** (ne stoppe même pas le service).

## Garde-fous
- `swap` refuse (avant tout stop) si le .so proposé == known-good.
- `revert` no-op si déjà known-good.
- Si après un stop la VRAM V100 ne se libère pas → **ALERTE** (liste des PIDs) et
  ABORT, au lieu de `kill` quoi que ce soit.
- Tout arrêt/démarrage via `systemctl` (User=ollama, Restart=always).

## Si le script casse en plein milieu (service arrêté)
Le .so sur disque est soit le nouveau **vérifié**, soit le known-good **restauré**.
Remettre la prod en route, à la main :
```bash
sudo systemctl start ollama-v100.service
./swap_revert.sh status        # confirmer .so == known-good si c'était un revert
```
Le backup du .so écrasé est dans `reference/swap-<horodatage>/`.

## Testé aujourd'hui (2026-10-05, prod intacte)
- `status` OK ; `revert` no-op OK (ne stoppe pas le service) ;
- garde `swap` : proposer le known-good comme « nouveau » → refusé **avant** tout stop.
- Service actif, VRAM 143 MiB free (inchangé), API /api/tags OK.
