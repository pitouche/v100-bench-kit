#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assainir les scripts shell du kit : chemins machine -> variables d'environnement.

Insert un préambule de configuration (défauts = machine de référence) juste
après la première ligne 'set -', puis convertit les constantes de tête :
    VAR=chemin          ->  VAR="${VAR:-chemin}"
Aucune logique n'est modifiée.
"""
import re, sys

PREAMBLE = '''
# --- Configuration (défauts = machine de référence; voir config/kit.env.example) ---
: "${LLAMA_ROOT:=/home/guile/llama-experiments}"
: "${OLLAMA_ROOT:=$LLAMA_ROOT/ollama-v100}"
: "${NSYS_BIN:=/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys}"
: "${MODEL_GGUF:=/usr/share/ollama/.ollama/models/blobs/sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb}"
: "${FILLER_FILE:=$LLAMA_ROOT/v100-repro/work/filler-200k.txt}"
'''

# conversions (nom variable, regex de la ligne)
CONST = {
    "BASE": (r"^BASE=/home/guile/llama-experiments$", 'BASE="${LLAMA_ROOT}"'),
    "TARGET": (r"^TARGET=\$BASE/ollama-v100$", 'TARGET="${OLLAMA_ROOT}"'),
    "NSYS": (r"^NSYS=/opt/nvidia/[^\"']*$", 'NSYS="${NSYS_BIN}"'),
    "NSYS_BIN": (r"^NSYS=/opt/nvidia/[^\"']*$", 'NSYS="${NSYS_BIN}"'),
    "SERVICE": (r"^SERVICE=ollama-v100$", 'SERVICE="${OLLAMA_SERVICE:-ollama-v100}"'),
    "MODEL_GGUF": (r"^MODEL_GGUF=/usr/share/ollama/[^\"']*$", 'MODEL_GGUF="${MODEL_GGUF}"'),
    "FILLER": (r"^FILLER=\$BASE/v100-repro/work/filler-200k\.txt$", 'FILLER="${FILLER_FILE}"'),
}

def process(path):
    with open(path) as f:
        lines = f.readlines()
    changed = 0
    ins_at = None
    for i, line in enumerate(lines):
        if line.startswith("set -") and ins_at is None:
            ins_at = i + 1
    if ins_at is None:
        print("%s: pas de 'set -' -> non traité" % path)
        return
    # conversions
    for i, line in enumerate(lines):
        for name, (pat, repl) in CONST.items():
            if re.match(pat, line):
                lines[i] = repl + "\n"
                changed += 1
                break
    lines[ins_at:ins_at] = [PREAMBLE]
    with open(path, "w") as f:
        f.writelines(lines)
    print("%s: %d constante(s) convertie(s)" % (path.split("/")[-1], changed))

if __name__ == "__main__":
    for p in sys.argv[1:]:
        process(p)
