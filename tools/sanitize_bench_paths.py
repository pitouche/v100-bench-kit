#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assainir les harnais run9* : chemins machine -> variables d'environnement.

Politique : chaque valeur machine-dépendante devient
    os.environ.get("NOM", "<defaut machine de reference>")
Les défauts restent ceux de la machine de référence (reproductibilité des
artefacts publiés) ; sur une autre machine, définir les variables
(cf. config/kit.env.example) avant de lancer.
Aucune logique n'est modifiée — uniquement les constantes de configuration.
Idempotent : si la ligne est déjà sous forme environ.get, on ne touche pas.
"""
import os, re, sys

RULES = [
    (r'^ROOT = "/home/guile/llama-experiments"$',
     'ROOT = os.environ.get("LLAMA_ROOT", "/home/guile/llama-experiments")'),

    (r'^MODEL_GGUF = "/usr/share/ollama/[^"]*"$',
     'MODEL_GGUF = os.environ.get("MODEL_GGUF",\n'
     '    "/usr/share/ollama/.ollama/models/blobs/sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")'),

    (r'^REF_LIB = "/var/backups/[^"]*"$',
     'REF_LIB = os.environ.get("REF_LIB",\n'
     '    "/var/backups/llama-experiments/libggml-cuda.so.prod-20261014-134252")'),

    (r'^DEPLOYED = .*libggml-cuda\.so.*$',
     'DEPLOYED = os.environ.get("DEPLOYED_LIB",\n'
     '    os.path.join(ROOT, "ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"))'),

    (r'^H2_LIB = os\.path\.join\(ROOT, [^\n]*$',
     'H2_LIB = os.environ.get("H2_LIB", os.path.join(ROOT, "h2/cupti129-test/e2/h2lib/lib/libggml-cuda-h2.so"))'),
    (r'^H2B_LIB = os\.path\.join\(ROOT, [^\n]*$',
     'H2B_LIB = os.environ.get("H2B_LIB", os.path.join(ROOT, "h2/cupti129-test/e2/h2lib/lib/libggml-cuda-h2b.so"))'),

    (r'^_PROMPT = json\.load\(open\(os\.path\.join\(ROOT, "h2/runs/run-20261009-152723/prompt\.json"\)\)\)$',
     'PROMPT_FILE = os.environ.get("PROMPT_FILE",\n'
     '    os.path.join(ROOT, "h2/runs/run-20261009-152723/prompt.json"))\n'
     '_PROMPT = json.load(open(PROMPT_FILE))'),

    # index nvidia-smi (query mémoire)
    (r'"--query-gpu=memory\.used", "--format=csv,noheader,nounits", "-i", "1"\]',
     '"--query-gpu=memory.used", "--format=csv,noheader,nounits",\n'
     '                   "-i", os.environ.get("V100_SM_INDEX", "1")]'),

    # service systemd
    (r'"stop", "ollama-v100"\]',
     '"stop", os.environ.get("OLLAMA_SERVICE", "ollama-v100")]'),
    (r'"start", "ollama-v100"\]',
     '"start", os.environ.get("OLLAMA_SERVICE", "ollama-v100")]'),
    (r'"is-active", "ollama-v100"\]',
     '"is-active", os.environ.get("OLLAMA_SERVICE", "ollama-v100")]'),

    # port prod vérifié
    (r'urlopen\("http://127\.0\.0\.1:11434/api/version"',
     'urlopen("http://127.0.0.1:" + os.environ.get("PROD_PORT", "11434") + "/api/version"'),
]

def process(path):
    with open(path) as f:
        lines = f.readlines()
    changed = 0
    for i, line in enumerate(lines):
        for pat, repl in RULES:
            if re.search(pat, line) and "environ.get" not in line:
                lines[i] = re.sub(pat, repl, line)
                changed += 1
                break
    if changed:
        with open(path, "w") as f:
            f.writelines(lines)
    print("%s: %d ligne(s) modifiée(s)" % (os.path.basename(path), changed))

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    for fn in sorted(os.listdir(d)):
        if fn.startswith("run9") and fn.endswith(".py"):
            process(os.path.join(d, fn))
