#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RUN8 — wrapper Python pour supervisor_bench.py (qui lance `python3 <bench>`).

Exécute le script bash run8_capture.sh (capture CUPTI v2 @ ctx ≈ 41 500 tokens,
dérivé de run7_v2.sh) et renvoie SON code de sortie tel quel (transparence).

Usage : python3 run8_bench.py [nb_tokens]   (défaut 64)
"""
import os
import subprocess
import sys

D = os.path.dirname(os.path.abspath(__file__))
NTOK = sys.argv[1] if len(sys.argv) > 1 else "64"
SCRIPT = os.path.join(D, "run8_capture.sh")

print(f"RUN8 wrapper : bash {SCRIPT} {NTOK}", flush=True)
rc = subprocess.run(["bash", SCRIPT, NTOK]).returncode
print(f"RUN8 wrapper : rc={rc}", flush=True)
sys.exit(rc)
