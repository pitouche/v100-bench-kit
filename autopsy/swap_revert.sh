#!/usr/bin/env bash
# =============================================================================
# swap_revert.sh — swap / rollback du .so CUDA de la prod ollama-v100
#
# Principe de sécurité : ON NE BOUGE QUE LE .so. Le serveur ollama (systemd),
# le binaire ollama, et le GGUF ne sont JAMAIs touchés par ce script.
#   - `swap <new.so>` : stop service -> backup du .so en place -> install new.so
#                       -> start service -> health
#   - `revert`        : stop service -> backup du .so en place (on ne perd rien)
#                       -> restaure la REF_COPY known-good (hash vérifié)
#                       -> start service -> health
#   - `status`        : hash du .so en place vs known-good, service, VRAM
#   - `health`        : API /api/tags + VRAM + llama-server  (ajoute --load)
#
# Tout arrêt/démarrage passe par systemctl (User=ollama, Restart=always).
# Ce script ne tue JAMAIS de process à la main ; s'il reste du VRAM après un
# stop, il ALERTE (PID en cause) au lieu de kill.
#
# Usage :
#   ./swap_revert.sh status
#   ./swap_revert.sh swap /chemin/vers/nouveau/libggml-cuda.so
#   ./swap_revert.sh revert
#   ./swap_revert.sh health [--load]
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=revert.conf
source "$HERE/revert.conf"

LOG="$HERE/swap_revert.log"
ts()   { date '+%Y-%m-%d %H:%M:%S'; }
say()  { printf '%s %s\n' "$(ts)" "$*"; }
err()  { printf '%s ERROR: %s\n' "$(ts)" "$*" >&2; }
die()  { err "$*"; exit 1; }

# --- helpers d'état ----------------------------------------------------------
vram_free_mib() {  # free VRAM (MiB) de la V100 (smi index)
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$V100_SMI_INDEX" 2>/dev/null \
    | tr -d ' ' | head -1
}
so_md5()    { md5sum    "$SO_PATH" 2>/dev/null | cut -d' ' -f1; }
so_sha256() { sha256sum "$SO_PATH" 2>/dev/null | cut -d' ' -f1; }
svc_active(){ systemctl is-active "$SERVICE" 2>/dev/null || true; }

require_root() {
  [ "$(id -u)" = 0 ] || die "rebooter avec sudo : $ sudo $0 $*"
}

wait_vram_freed() {
  # après un stop, la VRAM V100 doit redescendre ; on attend, sinon on alerte.
  local tries="${1:-60}" i freeb holder
  for ((i=0; i<tries; i++)); do
    freeb="$(vram_free_mib)"; freeb="${freeb:-0}"
    if [ "$freeb" -ge "$V100_MEM_FREE_MIN_MIB" ]; then
      say "V100 VRAM libérée : ${freeb} MiB free"
      return 0
    fi
    sleep 2
  done
  holder="$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null \
            | awk -v idx="$V100_SMI_INDEX" '{print}')"
  err "V100 VRAM toujours occupée après stop (free=$(vram_free_mib) MiB < ${V100_MEM_FREE_MIN_MIB})."
  err "Tenants actuels (ne rien kill à la main, vérifier d'abord) :"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/    /' >&2
  err "Refus de continuer tant que la VRAM n'est pas libre. Réessayer quand c'est libre."
  return 1
}

wait_service_up() {
  local i st
  for ((i=0; i<30; i++)); do
    st="$(svc_active)"
    [ "$st" = "active" ] && { say "service $SERVICE = active"; return 0; }
    sleep 2
  done
  st="$(svc_active)"
  [ "$st" = "active" ] || die "service $SERVICE pas revenu 'active' (état: $st). systemctl status $SERVICE"
}

api_ok() {
  curl -s --max-time 8 "http://127.0.0.1:$PORT/api/tags" 2>/dev/null | grep -q '"models"'
}

backup_current_so() {
  # snapshot horodaté du .so EN PLACE avant de l'écraser (on ne perd jamais un build)
  local stamp cur_md5 dir
  stamp="$(date +%Y%m%d-%H%M%S)"
  cur_md5="$(so_md5)"
  dir="$HERE/reference/swap-$stamp"
  mkdir -p "$dir"
  cp -p "$SO_PATH" "$dir/libggml-cuda.so.${cur_md5:0:8}.in-place"
  say "backup du .so en place (md5 ${cur_md5:0:8}) -> $dir/"
  # marqueur : ce qui est installé maintenant
  {
    echo "timestamp=$stamp"
    echo "installed_md5=$cur_md5"
    echo "installed_sha256=$(so_sha256)"
  } > "$HERE/reference/last_installed.txt"
}

restore_known_good() {
  [ -f "$REF_COPY" ] || die "REF_COPY introuvable : $REF_COPY"
  local ref_sha ref_md5
  ref_sha="$(sha256sum "$REF_COPY" | cut -d' ' -f1)"
  ref_md5="$(md5sum    "$REF_COPY" | cut -d' ' -f1)"
  [ "$ref_sha" = "$SHA256_KNOWN_GOOD" ] || die "REF_COPY corrompue ! sha256=$ref_sha != $SHA256_KNOWN_GOOD"
  [ "$ref_md5" = "$MD5_KNOWN_GOOD" ]    || die "REF_COPY md5 incohérent ($ref_md5 != $MD5_KNOWN_GOOD)"
  say "restauration known-good (md5 ${MD5_KNOWN_GOOD:0:8}) depuis $REF_COPY"
  cp -p "$REF_COPY" "$SO_PATH"
  chmod 755 "$SO_PATH"
  # re-vérification post-copie
  [ "$(so_sha256)" = "$SHA256_KNOWN_GOOD" ] || die "le .so restauré ne matche PAS le known-good (sha256). ABORT, service pas encore redémarré."
  [ "$(so_md5)"    = "$MD5_KNOWN_GOOD" ]    || die "le .so restauré ne matche PAS le known-good (md5). ABORT."
  say "vérification post-copie OK : .so == known-good byte-identique"
}

# --- commands ----------------------------------------------------------------
cmd_status() {
  echo "==================== swap_revert : status ===================="
  say "service $SERVICE : $(svc_active)"
  say "V100 (smi idx $V100_SMI_INDEX) VRAM free : $(vram_free_mib) MiB"
  say ".so en place  md5 : $(so_md5)"
  say "known-good    md5 : $MD5_KNOWN_GOOD"
  if [ "$(so_md5)" = "$MD5_KNOWN_GOOD" ]; then
    say "==> le .so en place EST le known-good (rien à rollbacker)"
  else
    say "==> le .so en place DIFFERE du known-good (un build de test est actif)"
  fi
  [ -f "$HERE/reference/last_installed.txt" ] && {
    say "dernière opération enregistrée :"
    sed 's/^/    /' "$HERE/reference/last_installed.txt"
  }
  if api_ok; then say "API http://127.0.0.1:$PORT/api/tags : OK"; else say "API : pas de réponse"; fi
}

cmd_health() {
  local load="${1:-}"
  say "service : $(svc_active) | V100 VRAM free : $(vram_free_mib) MiB"
  if api_ok; then
    say "API /api/tags : OK"
    curl -s --max-time 8 "http://127.0.0.1:$PORT/api/tags" 2>/dev/null \
      | python3 -c "import sys,json;[print('    -',m['name']) for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null || true
  else
    say "API : pas de réponse"
  fi
  if [ "$load" = "--load" ]; then
    say "chargement forcé du modèle ($MODEL_TAG) via un /api/generate minimal ..."
    local out
    out="$(curl -s --max-time 600 "http://127.0.0.1:$PORT/api/generate" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_TAG\",\"prompt\":\"ping\",\"stream\":false,\"options\":{\"num_predict\":1}}")" || true
    printf '%s\n' "$out" | head -c 400; echo
    say "après chargement, V100 VRAM free : $(vram_free_mib) MiB (le modèle ~23Go doit être en VRAM)"
  fi
}

stop_service() {
  local st
  st="$(svc_active)"
  [ "$st" = "active" ] || { say "service déjà '$st', rien à stopper"; return 0; }
  say "stop de $SERVICE ..."
  systemctl stop "$SERVICE"
  wait_vram_freed 60 || die "VRAM pas libérée, swap/revert annulé (service reste arrêté)."
  say "service stoppé, VRAM libre."
}

start_service() {
  say "démarrage de $SERVICE ..."
  systemctl start "$SERVICE"
  wait_service_up
}

cmd_swap() {
  local newso="${1:-}"
  [ -n "$newso" ] || die "usage: $0 swap /chemin/vers/libggml-cuda.so"
  [ -f "$newso" ] || die "fichier introuvable : $newso"
  require_root swap
  local new_md5 new_sha
  new_md5="$(md5sum "$newso" | cut -d' ' -f1)"
  new_sha="$(sha256sum "$newso" | cut -d' ' -f1)"
  say "nouveau .so : md5=${new_md5:0:8} sha256=${new_sha:0:16}..."
  if [ "$new_md5" = "$MD5_KNOWN_GOOD" ]; then
    die "le .so proposé EST déjà le known-good ; pas besoin de swap (utilisez status)."
  fi
  say "### SWAP $(ts) — nouveau .so md5 ${new_md5:0:8} ###" | tee -a "$LOG"
  stop_service
  backup_current_so
  cp -p "$newso" "$SO_PATH"
  chmod 755 "$SO_PATH"
  [ "$(so_sha256)" = "$new_sha" ] || die "copie du nouveau .so incohérente (sha256). ABORT, service arrêté."
  say "nouveau .so installé et vérifié"
  start_service
  cmd_health
  say "### SWAP terminé. Pour revenir en arrière :  $0 revert ###" | tee -a "$LOG"
}

cmd_revert() {
  require_root revert
  if [ "$(so_md5)" = "$MD5_KNOWN_GOOD" ]; then
    say "le .so en place EST déjà le known-good (md5 ${MD5_KNOWN_GOOD:0:8})."
    say "Aucun rollback nécessaire. (Service : $(svc_active))"
    cmd_health
    return 0
  fi
  say "### REVERT $(ts) — retour au known-good md5 ${MD5_KNOWN_GOOD:0:8} ###" | tee -a "$LOG"
  stop_service
  backup_current_so            # préserve le build de test avant de l'écraser
  restore_known_good
  start_service
  cmd_health
  say "### REVERT terminé : prod back to known-good (md5 ${MD5_KNOWN_GOOD:0:8}) ###" | tee -a "$LOG"
}

# --- dispatch ----------------------------------------------------------------
case "${1:-}" in
  status)  cmd_status ;;
  health)  cmd_health "${2:-}" ;;
  swap)    cmd_swap   "${2:-}" ;;
  revert)  cmd_revert ;;
  ""|help|-h|--help)
    grep '^# ' "$0" | sed 's/^# \{0,1\}//' | sed -n '1,30p'
    ;;
  *) die "commande inconnue : $1  (voir: $0 help)" ;;
esac
