#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chat_watchdog.py — garde de veille des fins de génération OWUI (cron 5 min).

Missions (uniquement les échecs que le superviseur ne couvre PAS) :
  - le nœud assistant courant du chat agent est FINI (done=True) avec une
    erreur de DÉPASSEMENT DE CONTEXTE (ex. "request (405240 tokens) exceeds
    the available context size (200192 tokens)" — un appel outil qui renvoie
    trop de données, ou un contexte qui a atteint sa limite) ;
  - sinon (pas d'erreur, ou erreur hors périmètre : "Ollama: Cannot
    connect…", modèle absent, OOM, …) : RIEN — la reprise de ce cas est le
    rôle du superviseur.

Si le cas se présente :
  1. COMPACTER le contexte du chat via `POST /api/v1/chats/<id>/compact`
     (OWUI 0.12.2 : le résumé de branche est écrit sur le nœud courant sous
     `contextSummary` ; à la prochaine génération le contexte est replacé
     par le résumé + les messages récents — c'est le mécanisme interne du
     "/COMPACT" de l'UI, appelé ici côté serveur),
  2. puis RELAUNCHER l'inférence avec exactement la même mécanique que le
     superviseur (nœuds déterministes `kolbo-wd:` / `kolbo-wd-a:`, commit
     partiel, tir non bloquant, réinjection terminal_id, JAMAIS de clé
     `tools`).

Aware du superviseur : même verrou que lui (`/tmp/kolbo-supervisor-bench.lock`,
flock EX). Le watchdog ne tente le verrou qu'une seule fois, NON bloquant :
s'il est tenu (superviseur en cours, de la fin du bench à la reprise), il
sort sans rien faire. C'est le seul point de collision possible — il est
fermé par construction. Idempotence : les IDs de nœud dépendent uniquement
du nœud en erreur → aucune reprise ne peut être doublée.

USAGE (cron, toutes les 5 min) :
  * * * * * cd <LLAMA_ROOT>/bench && python3 chat_watchdog.py >> <LLAMA_ROOT>/bench/chat_watchdog.cron.log 2>&1

LOG : bench/chat_watchdog.log (+ miroir cron.log pour le cron)
Env : OWUI_URL, OWUI_CHAT_ID (override chat), OWUI_TERMINAL_ID (override
terminal), WATCHDOG_MAX_RUN_S (safety, défaut 300).
"""
import fcntl
import json
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supervisor_bench as sb  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(D, "chat_watchdog.log")

# Safety globale (surtout pour les timeouts de la compaction).
MAX_RUN_S = int(os.getenv("WATCHDOG_MAX_RUN_S", "300"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("chat-watchdog")

WD_PREFIX = "kolbo-wd"
WD_PROMPT = (
    "REPRISE AUTOMATIQUE (watchdog) : ta génération a échoué par dépassement "
    "de contexte (l'appel a renvoyé trop de données, ou le contexte a atteint "
    "sa limite). Le contexte a été compressé automatiquement avant cette "
    "reprise. Reprends EXACTEMENT l'action en cours là où elle s'est arrêtée "
    "— ne rejoue PAS les étapes déjà accomplies, ne redemande PAS "
    "d'information, et continue sans préambule sur l'interruption."
)

# Signatures d'erreur de dépassement de contexte (périmètre du watchdog).
# Le reste (Ollama: Cannot connect, modèle absent, OOM, …) n'est PAS du
# watchdog : le superviseur le reprend après le run, ou l'opérateur intervient.
CTX_SIGNATURES = (
    "exceed_context_size_error",
    "exceeds the available context size",
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "prompt too long",
)


def _alarm(_signum, _frame):
    raise TimeoutError("watchdog : timeout global dépassé")


def error_text(msg):
    """Texte de l'erreur d'un nœud (dict {'content': …}, str, …)."""
    err = msg.get("error")
    if err in (None, False, ""):
        return ""
    if isinstance(err, dict):
        return json.dumps(err, ensure_ascii=False)
    return str(err)


def is_context_error(text):
    t = text.lower()
    return any(sig in t for sig in CTX_SIGNATURES)


def acquire_watchdog_lock():
    """Tente UNE fois le verrou du superviseur, NON bloquant. Retourne le
    fd (gouverner jusqu'à la fin) ou None (superviseur en cours → inaction)."""
    lock = sb.LOCK_FILE.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return None
    return lock


def compact_chat(chat_id, timeout=180):
    """Compaction de la branche courante (mécanisme '/COMPACT' côté serveur).
    L'appel peut durer plusieurs minutes (le résumé est généré par le
    modèle) → timeout long, mais borné par le safety global."""
    r = sb.session.post(
        sb.OWUI_URL + f"/api/v1/chats/{chat_id}/compact",
        json={}, timeout=(sb.CONNECT_TIMEOUT, timeout))
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        return {}


def has_summary_after(chat_id):
    """Le nœud courant porte-t-il un contextSummary (compaction effective) ?"""
    chat = sb.api_get(f"/api/v1/chats/{chat_id}")
    history = chat.get("chat", {}).get("history", {})
    nodes = history.get("messages", {})
    cur = history.get("currentId")
    node = nodes.get(str(cur)) or {}
    summary = node.get("contextSummary")
    return isinstance(summary, str) and bool(summary.strip()), str(cur)


def main():
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(MAX_RUN_S)

    lock = acquire_watchdog_lock()
    if lock is None:
        log.info("Verrou superviseur tenu — superviseur en cours, "
                 "rien à faire.")
        return 0
    try:
        return watch()
    finally:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


def watch():
    if not sb.load_token():
        log.error("Token OWUI absent — sortie.")
        return 2
    sb.session.headers["Authorization"] = f"Bearer {sb.OWUI_TOKEN}"

    chat_id = sb.select_chat()
    if not chat_id:
        log.info("Aucun chat agent trouvé — rien à faire.")
        return 0

    chat = sb.api_get(f"/api/v1/chats/{chat_id}")
    try:
        message_id, msg = sb.extract_current(chat)
    except (ValueError, KeyError) as e:
        log.warning("Chat %s : %s — rien à faire.", chat_id[:8], e)
        return 0

    # Périmètre 1 : le nœud doit être FINI (une génération en cours,
    # done=False, n'est jamais touchée).
    if msg.get("done") is not True:
        log.info("Chat %s : nœud courant %s en cours (done=%r) — rien à "
                 "faire.", chat_id[:8], message_id[:8], msg.get("done"))
        return 0

    # Périmètre 2 : il faut une erreur, et une erreur de CONTEXTE.
    text = error_text(msg)
    if not text:
        log.info("Chat %s : nœud courant %s sans erreur — rien à faire.",
                 chat_id[:8], message_id[:8])
        return 0
    if not is_context_error(text):
        log.info("Chat %s : erreur hors périmètre contexte sur %s — "
                 "rien à faire (périmètre superviseur) : %.120s",
                 chat_id[:8], message_id[:8], text)
        return 0

    log.warning("Chat %s : nœud %s FINI avec erreur de CONTEXTE : %.200s",
                chat_id[:8], message_id[:8], text)

    # 1) Compaction forcée (mécanisme '/COMPACT' côté serveur).
    try:
        res = compact_chat(chat_id)
        log.info("Compaction : %s", json.dumps(res, ensure_ascii=False)[:300])
    except Exception as e:
        log.error("Compaction en échec : %s — pas de relance (un tir sans "
                  "compaction échouerait de la même façon).", e)
        return 1

    ok, cur_id = has_summary_after(chat_id)
    if not ok:
        log.error("Pas de contextSummary sur le nœud courant %s après "
                  "compaction — pas de relance.", cur_id[:8])
        return 1
    log.info("Compaction effective (contextSummary sur le nœud courant %s).",
             cur_id[:8])

    # 2) Relance — même mécanique que le superviseur (prefix kolbo-wd :
    #    les IDs n'entrent jamais en collision avec kolbo-resume).
    assistant_id = sb.send_resume(chat_id, chat, message_id, msg,
                                  prefix=WD_PREFIX, prompt=WD_PROMPT)
    log.info("Reprise watchdog envoyée (nœud assistant %s…) — OWUI prend "
             "la main.", assistant_id[:8])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TimeoutError as e:
        log.error("%s", e)
        sys.exit(1)
    except Exception as e:
        log.exception("Erreur inattendue : %s", e)
        sys.exit(1)
