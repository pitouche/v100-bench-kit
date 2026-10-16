#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RUN9 — A/B contrôlé REF (e4afbb4a) vs H2 (a4b6f440), contexte fixe ~41k.

Protocole : PROTOCOL-run9-ab.md (écrit avant lancement).
Sélection de lib SANS toucher la production : GGML_BACKEND_PATH par essai sur un
llama-server dédié. Ordre alterné : REF, H2, REF, H2. 2 essais/bras, 600 tok.
Le service prod est stoppé au début, restauré à la fin (fin de cycle unique).
"""
import json, os, re, shutil, subprocess, sys, time, urllib.request

ROOT = os.environ.get("LLAMA_ROOT", "/home/guile/llama-experiments")
OUTDIR = os.path.join(ROOT, "h2/cupti129-test/nightrun8/ab_run9")
os.makedirs(OUTDIR, exist_ok=True)
RESULTS = os.path.join(OUTDIR, "results.jsonl")

PORT = 44911
API = f"http://127.0.0.1:{PORT}"
MODEL_TAG = "qwen3.8-27b-q6-mtp:latest"
MODEL_GGUF = os.environ.get("MODEL_GGUF",
    "/usr/share/ollama/.ollama/models/blobs/sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
SHORTLIST = os.path.join(ROOT, "src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32")
L1 = json.load(open(os.path.join(ROOT, "h2/h2prod/suite_prompts.json")))
L1PROMPT = [p for p in L1 if p["id"] == "L1"][0]["prompt"]
INSTR = "Ecris un texte continu en francais sur l'importance de la maintenance preventive, developpe longuement chaque idee."
PROMPT = L1PROMPT + "\n\n" + INSTR

REF_LIB = os.environ.get("REF_LIB",
    "/var/backups/llama-experiments/libggml-cuda.so.prod-20261014-134252")
H2_LIB = os.environ.get("H2_LIB", os.path.join(ROOT, "h2/cupti129-test/e2/h2lib/lib/libggml-cuda-h2.so"))
DEPLOYED = os.environ.get("DEPLOYED_LIB",
    os.path.join(ROOT, "ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"))

# Ordre alterné (protocole)
TRIALS = [("ref1", REF_LIB, "e4afbb4a"),
          ("h2a1", H2_LIB, "a4b6f440"),
          ("ref2", REF_LIB, "e4afbb4a"),
          ("h2a2", H2_LIB, "a4b6f440")]

log = lambda *a: print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def md5(p):
    import hashlib
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def gpu_mem():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                   "-i", os.environ.get("V100_SM_INDEX", "1")],
                       capture_output=True, text=True, timeout=15)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def wait_gpu(max_wait=120, force=False):
    for i in range(max_wait):
        u = gpu_mem()
        if 0 <= u < 1000:
            return u
        if force and i % 10 == 9:
            subprocess.run(["pkill", "-9", "-f", "port " + str(PORT)], capture_output=True)
        time.sleep(1)
    return gpu_mem()


def post(payload, timeout=1800):
    req = urllib.request.Request(API + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


def base_payload(mx):
    return {"model": MODEL_TAG, "stream": False, "temperature": 0, "top_k": 1,
            "ignore_eos": True, "max_tokens": mx,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": PROMPT}]}


def run_trial(tag, lib, expected_md5):
    rec = {"tag": tag, "lib": lib, "lib_md5": md5(lib)[:8],
           "lib_expected": expected_md5, "t_start": time.strftime("%H:%M:%S", time.gmtime())}
    if rec["lib_md5"] != expected_md5:
        rec["status"] = "INVALID (lib md5)"
        return rec
    sout = f"/tmp/run9_server_{tag}.out"
    env = dict(os.environ)
    env.update({
        "LD_LIBRARY_PATH": f"{ROOT}/ollama-v100/lib/ollama:{ROOT}/ollama-v100/lib/ollama/cuda_v12:{ROOT}/tools/cuda-12.6/usr/local/cuda-12.6/lib64:{ROOT}/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
        "GGML_BACKEND_PATH": lib,
        "GGML_CUDA_VOLTA_W4R4": "1",
        "GGML_CUDA_QWEN35_MTP_SHORTLIST": SHORTLIST,
        "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "LLAMA_ARG_FIT_TARGET": "1024",
    })
    cmd = [f"{ROOT}/ollama-v100/lib/ollama/llama-server", "--model", MODEL_GGUF,
           "--port", str(PORT), "--host", "127.0.0.1", "--no-webui",
           "-c", "200000", "-np", "1", "-lv", "4", "--no-log-prefix", "--no-log-timestamps",
           "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
           "--spec-draft-backend-sampling", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
           "--flash-attn", "auto", "-b", "512", "-ub", "512"]
    with open(sout, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)

    def tail(n=15):
        try:
            return "".join(open(sout).readlines()[-n:])
        except Exception:
            return ""

    # gates
    up = False
    for i in range(300):
        if proc.poll() is not None:
            rec["status"] = "INVALID (server mort au démarrage)"; rec["tail"] = tail(); return rec
        try:
            urllib.request.urlopen(API + "/v1/models", timeout=2); up = True; break
        except Exception:
            time.sleep(1)
    if not up:
        rec["status"] = "INVALID (port non up)"; rec["tail"] = tail(); return rec
    fit = False
    for i in range(900):
        if proc.poll() is not None:
            rec["status"] = "INVALID (mort avant load)"; rec["tail"] = tail(); return rec
        if "out of memory" in tail(40) or "fit_params: failed" in tail(40):
            rec["status"] = "INVALID (OOM/FIT)"; rec["tail"] = tail(); return rec
        if "llama_server: model loaded" in tail(5):
            fit = True; break
        time.sleep(1)
    if not fit:
        rec["status"] = "INVALID (model non chargé)"; rec["tail"] = tail(); return rec
    txt = open(sout).read()
    if not any("using device CUDA" in l and "Tesla V100" in l for l in txt.splitlines()):
        rec["status"] = "INVALID (device != V100)"; rec["tail"] = tail(); return rec
    if "CUDA MTP shortlist" not in txt:
        # chargement lazy au 1er decode : on laisse le warmup le charger
        pass

    # chauffe
    warm = {"model": MODEL_TAG, "stream": False, "temperature": 0, "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Say OK."}]}
    wok = False
    for w in range(5):
        try:
            d, _ = post(warm, timeout=600)
            if "error" not in json.dumps(d).lower() and d.get("usage"):
                wok = True; break
        except Exception:
            pass
        time.sleep(10)
    if not wok:
        rec["status"] = "INVALID (warmup)"; rec["tail"] = tail(); return rec
    prefill = base_payload(16)
    for k in range(2):
        try:
            d, w = post(prefill, timeout=1200)
            rec[f"prefill{k+1}_s"] = round(w, 2)
            rec[f"prefill{k+1}_cached"] = d.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")
            if "error" in json.dumps(d).lower():
                rec["status"] = "INVALID (prefill)"; rec["tail"] = tail(); return rec
        except Exception as e:
            rec["status"] = f"INVALID (prefill exc: {type(e).__name__})"; rec["tail"] = tail(); return rec

    # mesure
    gen = base_payload(600)
    d, wall = post(gen, timeout=1800)
    if "error" in json.dumps(d).lower():
        rec["status"] = "INVALID (gen erreur)"; rec["tail"] = tail(); return rec
    txt2 = open(sout).read()
    acc_lines = [l for l in txt2.splitlines() if "draft acceptance =" in l]
    stat_lines = [l for l in txt2.splitlines() if "spec common_specu: statistics" in l]
    acc = None; mean_len = None
    if acc_lines:
        m = re.search(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s+([0-9.]+)", acc_lines[-1])
        if m:
            acc = {"acc_rate": float(m.group(1)), "accepted": int(m.group(2)),
                   "generated": int(m.group(3)), "mean_len": float(m.group(4))}
    mtp_cum = None
    if stat_lines:
        m = re.search(r"#gen tokens =\s+(\d+), #acc tokens =\s+(\d+), #mean acc len = ([0-9.]+)", stat_lines[-1])
        if m:
            mtp_cum = {"gen": int(m.group(1)), "acc": int(m.group(2)), "mean_acc_len": float(m.group(3))}
    usage = d.get("usage", {})
    prompt_ms = d.get("prompt_ms", 0.0)
    dec_ms = wall * 1000 - prompt_ms
    nt = usage.get("completion_tokens", 0)
    rec.update({
        "status": "OK",
        "wall_s": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
        "prompt_ms": round(prompt_ms, 1),
        "decode_ms": round(dec_ms, 1),
        "gen_tokens": nt,
        "ms_per_tok": round(dec_ms / max(1, nt), 2),
        "tok_per_s": round(nt / max(0.001, dec_ms / 1000), 2),
        "finish": d.get("choices", [{}])[0].get("finish_reason"),
        "draft_n": d.get("draft_n"), "draft_n_accepted": d.get("draft_n_accepted"),
        "mtp_request": acc, "mtp_cum": mtp_cum,
    })
    # gates post
    if not (38000 <= (rec.get("prompt_tokens") or 0) <= 45000):
        rec["status"] = "INVALID (G_CTX)"
    if nt != 600:
        rec["status"] = "INVALID (G_GEN)"
    if not (10 <= wall <= 120):
        rec["status"] = "INVALID (G_WALL)"
    if not (d.get("draft_n") or 0) > 0:
        rec["status"] = "INVALID (G_MTP)"
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    return rec


def main():
    log(f"=== RUN9 A/B start : {TRIALS} ===")
    log("lib ref md5 =", md5(REF_LIB)[:8], "| lib h2 md5 =", md5(H2_LIB)[:8],
        "| deployé (doit rester a4b6f440) =", md5(DEPLOYED)[:8])
    subprocess.run(["sudo", "systemctl", "stop", os.environ.get("OLLAMA_SERVICE", "ollama-v100")], capture_output=True)
    u = wait_gpu(120, force=True)
    log(f"prod stoppé, V100 = {u} MiB")
    if u not in (0,) and u >= 1000:
        log("ERREUR: V100 pas libre — arrêt (aucun essai lancé, on restaure)")
        subprocess.run(["sudo", "systemctl", "start", os.environ.get("OLLAMA_SERVICE", "ollama-v100")], capture_output=True)
        return 1
    records = []
    for tag, lib, exp in TRIALS:
        u = gpu_mem()
        log(f"--- essai {tag} (lib {exp[:8]}) : V100 au démarrage = {u} MiB ---")
        if u >= 1000:
            log("G_ISOLATION echec — essai INVALID")
            records.append({"tag": tag, "status": "INVALID (G_ISOLATION)"})
            continue
        rec = run_trial(tag, lib, exp)
        records.append(rec)
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"--- {tag} : {rec.get('status')} wall={rec.get('wall_s')}s "
            f"ms/tok={rec.get('ms_per_tok')} tok/s={rec.get('tok_per_s')} "
            f"acc_len={rec.get('mtp_request', {}).get('mean_len')} "
            f"ctx={rec.get('prompt_tokens')} ---")
        wait_gpu(60, force=True)
    # restauration du service prod (fin de cycle unique)
    subprocess.run(["sudo", "systemctl", "start", os.environ.get("OLLAMA_SERVICE", "ollama-v100")], capture_output=True)
    for i in range(120):
        if subprocess.run(["systemctl", "is-active", os.environ.get("OLLAMA_SERVICE", "ollama-v100")], capture_output=True, text=True).stdout.strip() == "active":
            break
        time.sleep(1)
    svc = subprocess.run(["systemctl", "is-active", os.environ.get("OLLAMA_SERVICE", "ollama-v100")], capture_output=True, text=True).stdout.strip()
    dep = md5(DEPLOYED)[:8]
    ref = md5(REF_LIB)[:8]
    log(f"=== FIN : service={svc} | .so déployé md5={dep} (attendu a4b6f440) | lib ref md5={ref} (attendu e4afbb4a) ===")
    if svc != "active" or dep != "a4b6f440":
        log("⚠️ INCIDENT : restauration incertaine — SIGNALER, aucune conclusion")
        return 3
    ok = [r for r in records if r.get("status") == "OK"]
    if len(ok) < 4:
        log(f"⚠️ {4 - len(ok)} essai(s) invalidé(s) — voir results.jsonl")
    log("results :", RESULTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
