#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RUN9C — A/B REF (e4afbb4a) vs H2 (a4b6f440), ~107k, + condition W4R4 OFF.

Protocole : PROTOCOL-run9c.md (écrit avant lancement, 2026-10-15 ~18:50 UTC).
Suite de RUN9B : même harnais, prompt du régime §26.5 (filler 107 351 tok),
decode 1200 tok, 6 essais (4 en W4R4=1 + 1 paire en W4R4=0), ordre alterné.

Écarts par rapport à run9b_bench.py (tous figés dans le protocole §3) :
  - prompt = h2/runs/run-20261009-152723/prompt.json (550 000 chars du filler,
    107 351 tokens) — même prompt que l'A/B §26.5.
  - max_tokens = 1200 (mesure et prefills 32).
  - 6 essais : ref1/h2a1/ref2/h2a2 (W4R4=1), ref3/h2a3 (W4R4=0).
  - G_CTX [100000, 115000], G_GEN 1200, G_WALL [30, 900].
  - Pas de logprobs (inutilisables sur ce binaire, §29.1-3).

Sélection de lib SANS toucher la production : GGML_BACKEND_PATH par essai.
Le service prod est stoppé au début, restauré à la fin (cycle unique).
"""
import json, os, re, subprocess, sys, time, urllib.request

ROOT = os.environ.get("LLAMA_ROOT", "/home/guile/llama-experiments")
OUTDIR = os.path.join(ROOT, "h2/cupti129-test/nightrun8/ab_run9c")
os.makedirs(OUTDIR, exist_ok=True)
RESULTS = os.path.join(OUTDIR, "results.jsonl")

PORT = 44911
API = f"http://127.0.0.1:{PORT}"
MODEL_TAG = "qwen3.8-27b-q6-mtp:latest"
MODEL_GGUF = os.environ.get("MODEL_GGUF",
    "/usr/share/ollama/.ollama/models/blobs/sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
SHORTLIST = os.path.join(ROOT, "src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32")

# prompt du régime §26.5 (identique mot pour mot à l'A/B h2_ab_lib.sh)
PROMPT_FILE = os.environ.get("PROMPT_FILE",
    os.path.join(ROOT, "h2/runs/run-20261009-152723/prompt.json"))
_PROMPT = json.load(open(PROMPT_FILE))
PROMPT = _PROMPT["messages"][0]["content"]

REF_LIB = os.environ.get("REF_LIB",
    "/var/backups/llama-experiments/libggml-cuda.so.prod-20261014-134252")
H2_LIB = os.environ.get("H2_LIB", os.path.join(ROOT, "h2/cupti129-test/e2/h2lib/lib/libggml-cuda-h2.so"))
DEPLOYED = os.environ.get("DEPLOYED_LIB",
    os.path.join(ROOT, "ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"))

# (tag, lib, md5 attendu, W4R4)
TRIALS = [("ref1", REF_LIB, "e4afbb4a", "1"),
          ("h2a1", H2_LIB, "a4b6f440", "1"),
          ("ref2", REF_LIB, "e4afbb4a", "1"),
          ("h2a2", H2_LIB, "a4b6f440", "1"),
          ("ref3", REF_LIB, "e4afbb4a", "0"),
          ("h2a3", H2_LIB, "a4b6f440", "0")]

N_DEC = 1200

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


def wait_gpu(max_wait=180, force=False):
    for i in range(max_wait):
        u = gpu_mem()
        if 0 <= u < 1000:
            return u
        if force and i % 10 == 9:
            subprocess.run(["pkill", "-9", "-f", "port " + str(PORT)], capture_output=True)
        time.sleep(1)
    return gpu_mem()


def post(payload, timeout=3600):
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


ACC_RE = re.compile(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s+([0-9.]+)")
SPEC_RE = re.compile(r"spec common_specu: statistics\s+draft-mtp: #calls\(b,g,a\) =\s+(\d+)\s+(\d+)\s+(\d+), #gen drafts =\s+(\d+), #acc drafts =\s+(\d+), #gen tokens =\s+(\d+), #acc tokens =\s+(\d+), #mean acc len = ([0-9.]+), #acc rate/pos = \(([0-9.]+), ([0-9.]+), ([0-9.]+)\), dur\(b,g,a\) = ([0-9.]+), ([0-9.]+), ([0-9.]+) ms")


def parse_mtp(sout):
    """Fenêtre MTP = delta des 2 dernières lignes (prefill2 chaud → decode)."""
    acc_lines = [l for l in sout.splitlines() if "draft acceptance =" in l]
    spec_lines = [l for l in sout.splitlines() if "spec common_specu: statistics" in l]
    mtp_req = None
    if acc_lines:
        m = ACC_RE.search(acc_lines[-1])
        if m:
            mtp_req = {"acc_rate": float(m.group(1)), "accepted": int(m.group(2)),
                       "generated": int(m.group(3)), "mean_len": float(m.group(4))}
    mtp_delta = None
    if len(spec_lines) >= 2:
        m0 = SPEC_RE.search(spec_lines[-2])
        m1 = SPEC_RE.search(spec_lines[-1])
        if m0 and m1:
            g0, a0, c0 = int(m0.group(6)), int(m0.group(7)), int(m0.group(1))
            g1, a1, c1 = int(m1.group(6)), int(m1.group(7)), int(m1.group(1))
            dgen, dacc, dcalls = g1 - g0, a1 - a0, c1 - c0
            mtp_delta = {"gen": dgen, "acc": dacc, "calls": dcalls,
                         "acc_rate": (dacc / dgen) if dgen else None,
                         "mean_acc_len": (dacc / dcalls) if dcalls else None,
                         "cum_pos": [float(m1.group(9)), float(m1.group(10)), float(m1.group(11))],
                         "cum_mean_len": float(m1.group(8)),
                         "dur_g_ms": float(m1.group(13))}
    return mtp_req, mtp_delta


def run_trial(tag, lib, expected_md5, w4r4):
    rec = {"tag": tag, "lib": lib, "lib_md5": md5(lib)[:8],
           "lib_expected": expected_md5, "w4r4": w4r4,
           "t_start": time.strftime("%H:%M:%S", time.gmtime())}
    if rec["lib_md5"] != expected_md5:
        rec["status"] = "INVALID (G_LIB)"
        return rec
    sout = f"{OUTDIR}/server_{tag}.out"
    genpath = f"{OUTDIR}/gen_{tag}.json"
    env = dict(os.environ)
    env.update({
        "LD_LIBRARY_PATH": f"{ROOT}/ollama-v100/lib/ollama:{ROOT}/ollama-v100/lib/ollama/cuda_v12:{ROOT}/tools/cuda-12.6/usr/local/cuda-12.6/lib64:{ROOT}/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
        "GGML_BACKEND_PATH": lib,
        "GGML_CUDA_VOLTA_Q6_W4R4": w4r4,     # "1" = parité prod/§26.5 ; "0" = condition OFF
        "GGML_CUDA_QWEN35_MTP_SHORTLIST": SHORTLIST,
        "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "LLAMA_ARG_FIT_TARGET": "1024",
    })
    rec["env"] = {k: env[k] for k in ("GGML_BACKEND_PATH", "GGML_CUDA_VOLTA_Q6_W4R4",
                                      "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT", "CUDA_VISIBLE_DEVICES",
                                      "LLAMA_ARG_FIT_TARGET")}
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

    # gates de démarrage (G_START)
    up = False
    for i in range(300):
        if proc.poll() is not None:
            rec["status"] = "INVALID (G_START: server mort au démarrage)"; rec["tail"] = tail(); return rec
        try:
            urllib.request.urlopen(API + "/v1/models", timeout=2); up = True; break
        except Exception:
            time.sleep(1)
    if not up:
        rec["status"] = "INVALID (G_START: port non up)"; rec["tail"] = tail(); return rec
    fit = False
    for i in range(900):
        if proc.poll() is not None:
            rec["status"] = "INVALID (G_START: mort avant load)"; rec["tail"] = tail(); return rec
        if "out of memory" in tail(40) or "fit_params: failed" in tail(40):
            rec["status"] = "INVALID (G_START: OOM/FIT)"; rec["tail"] = tail(); return rec
        if "llama_server: model loaded" in tail(5):
            fit = True; break
        time.sleep(1)
    if not fit:
        rec["status"] = "INVALID (G_START: model non chargé)"; rec["tail"] = tail(); return rec
    txt = open(sout).read()
    if not any("using device CUDA" in l and "Tesla V100" in l for l in txt.splitlines()):
        rec["status"] = "INVALID (G_START: device != V100)"; rec["tail"] = tail(); return rec
    rec["shortlist_loaded"] = "CUDA MTP shortlist" in txt
    if not rec["shortlist_loaded"]:
        rec["shortlist_loaded"] = None  # lazy : re-vérifié après warmup

    # chauffe (G_WARM/PF)
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
        rec["status"] = "INVALID (G_WARM)"; rec["tail"] = tail(); return rec
    txt = open(sout).read()
    rec["shortlist_loaded"] = rec.get("shortlist_loaded") or ("CUDA MTP shortlist" in txt)
    prefill = base_payload(32)
    for k in range(2):
        try:
            d, w = post(prefill, timeout=1800)
            rec[f"prefill{k+1}_s"] = round(w, 2)
            rec[f"prefill{k+1}_cached"] = d.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")
            if "error" in json.dumps(d).lower():
                rec["status"] = f"INVALID (G_PF: prefill{k+1})"; rec["tail"] = tail(); return rec
        except Exception as e:
            rec["status"] = f"INVALID (G_PF: prefill exc {type(e).__name__})"; rec["tail"] = tail(); return rec

    # mesure (pas de logprobs — inutilisables, §29.1-3)
    gen = base_payload(N_DEC)
    try:
        d, wall = post(gen, timeout=3600)
    except Exception as e:
        rec["status"] = f"INVALID (gen exc {type(e).__name__})"; rec["tail"] = tail()
        proc.terminate()
        return rec
    if "error" in json.dumps(d).lower():
        rec["status"] = "INVALID (gen erreur)"; rec["tail"] = tail()
        proc.terminate()
        return rec
    with open(genpath, "w") as f:
        json.dump(d, f, ensure_ascii=False)

    txt2 = open(sout).read()
    mtp_req, mtp_delta = parse_mtp(txt2)
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
        "mtp_request": mtp_req,
        "mtp_delta": mtp_delta,
        "gen_file": genpath,
        "server_log": sout,
    })
    # gates post (G_CTX / G_GEN / G_WALL / G_MTP / G_SHORTLIST)
    if not (100000 <= (rec.get("prompt_tokens") or 0) <= 115000):
        rec["status"] = "INVALID (G_CTX)"
    if nt != N_DEC:
        rec["status"] = "INVALID (G_GEN)"
    if not (30 <= wall <= 900):
        rec["status"] = "INVALID (G_WALL)"
    g_mtp = (mtp_req and mtp_req["generated"] > 0) or (mtp_delta and mtp_delta["gen"] > 0)
    if not g_mtp:
        rec["status"] = "INVALID (G_MTP)"
    if rec.get("shortlist_loaded") is not True:
        rec["status"] = "INVALID (G_SHORTLIST)"
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    return rec


def main():
    log(f"=== RUN9C A/B start : {[(t, e[:8], 'w4r4='+w) for t, _, e, w in TRIALS]} ===")
    log("prompt chars =", len(PROMPT), "(attendu 550000, régime §26.5, 107351 tok)")
    log("lib ref md5 =", md5(REF_LIB)[:8], "| lib h2 md5 =", md5(H2_LIB)[:8],
        "| deployé (doit rester a4b6f440) =", md5(DEPLOYED)[:8])
    if md5(REF_LIB)[:8] != "e4afbb4a" or md5(H2_LIB)[:8] != "a4b6f440":
        log("ERREUR: md5 libs inattendus — arrêt avant stop du service")
        return 1
    subprocess.run(["sudo", "systemctl", "stop", os.environ.get("OLLAMA_SERVICE", "ollama-v100")], capture_output=True)
    u = wait_gpu(180, force=True)
    log(f"prod stoppé, V100 = {u} MiB")
    if not (0 <= u < 1000):
        log("ERREUR: V100 pas libre — arrêt (aucun essai lancé, on restaure)")
        subprocess.run(["sudo", "systemctl", "start", os.environ.get("OLLAMA_SERVICE", "ollama-v100")], capture_output=True)
        return 1
    records = []
    for tag, lib, exp, w4r4 in TRIALS:
        u = gpu_mem()
        log(f"--- essai {tag} (lib {exp[:8]}, w4r4={w4r4}) : V100 au démarrage = {u} MiB ---")
        if u >= 1000:
            log("G_ISOLATION echec — essai INVALID")
            records.append({"tag": tag, "status": "INVALID (G_ISOLATION)"})
            continue
        rec = run_trial(tag, lib, exp, w4r4)
        records.append(rec)
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"--- {tag} : {rec.get('status')} wall={rec.get('wall_s')}s "
            f"ms/tok={rec.get('ms_per_tok')} tok/s={rec.get('tok_per_s')} "
            f"acc_req={rec.get('mtp_request', {}).get('acc_rate')} "
            f"acc_delta={rec.get('mtp_delta', {}).get('acc_rate')} "
            f"ctx={rec.get('prompt_tokens')} ---")
        wait_gpu(120, force=True)
    subprocess.run(["sudo", "systemctl", "start", os.environ.get("OLLAMA_SERVICE", "ollama-v100")], capture_output=True)
    for i in range(120):
        if subprocess.run(["systemctl", "is-active", os.environ.get("OLLAMA_SERVICE", "ollama-v100")], capture_output=True, text=True).stdout.strip() == "active":
            break
        time.sleep(1)
    svc = subprocess.run(["systemctl", "is-active", os.environ.get("OLLAMA_SERVICE", "ollama-v100")], capture_output=True, text=True).stdout.strip()
    dep = md5(DEPLOYED)[:8]
    ref = md5(REF_LIB)[:8]
    api_ok = False
    try:
        urllib.request.urlopen("http://127.0.0.1:" + os.environ.get("PROD_PORT", "11434") + "/api/version", timeout=5)
        api_ok = True
    except Exception:
        pass
    log(f"=== FIN : service={svc} api={api_ok} | .so déployé md5={dep} (attendu a4b6f440) | lib ref md5={ref} (attendu e4afbb4a) ===")
    if svc != "active" or dep != "a4b6f440":
        log("⚠️ INCIDENT : restauration incertaine — SIGNALER, aucune conclusion")
        return 3
    ok = [r for r in records if r.get("status") == "OK"]
    if len(ok) < 6:
        log(f"⚠️ {6 - len(ok)} essai(s) invalidé(s) — voir results.jsonl")
    log("results :", RESULTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
