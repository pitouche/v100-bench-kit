#!/usr/bin/env python3
"""Fixture de test pour chat_watchdog.py : crée un chat test avec un nœud
assistant FINI portant une erreur de dépassement de contexte (le cas que le
watchdog doit traiter). Supprime le chat de test à la fin."""
import json
import pathlib
import sys
import time
import uuid

import requests

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import supervisor_bench as sb  # noqa: E402


def main():
    if not sb.load_token():
        print("token absent")
        return 2
    sb.session.headers["Authorization"] = f"Bearer {sb.OWUI_TOKEN}"

    # 1) Nouveau chat vide
    chat = sb.api_post("/api/v1/chats/new", {
        "name": "WD-TEST fixture (à supprimer)",
        "chat": {"history": {"currentId": None, "messages": {}}},
    }).json()
    cid = chat["id"]
    print("chat test créé :", cid)

    # 2) Nœud user + nœud assistant ERREUR CONTEXTE
    ts = int(time.time())
    uid = str(uuid.uuid4())
    aid = str(uuid.uuid4())
    model = "kolbo5"
    user_node = {
        "id": uid, "role": "user",
        "content": "fixture watchdog : tâche fictive en cours (test compaction+reprise)",
        "parentId": None, "childrenIds": [aid],
        "timestamp": ts, "models": [model],
    }
    asst_node = {
        "id": aid, "role": "assistant", "content": "",
        "parentId": uid, "childrenIds": [],
        "model": model, "modelName": model, "modelIdx": 0,
        "done": True, "timestamp": ts + 1,
        "error": {"content": json.dumps({
            "error": {
                "code": 400,
                "message": "request (245312 tokens) exceeds the available "
                           "context size (200192 tokens), try increasing it",
                "type": "exceed_context_size_error",
                "n_prompt_tokens": 245312, "n_ctx": 200192,
            }}, ensure_ascii=False)},
    }
    sb.api_post(f"/api/v1/chats/{cid}", {
        "chat": {"history": {
            "currentId": aid,
            "messages": {uid: user_node, aid: asst_node},
        }}})
    print("nœud user   :", uid)
    print("nœud erreur :", aid)
    print("CHAT_TEST_ID=" + cid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
