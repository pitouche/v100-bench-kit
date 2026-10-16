// (c) 2026 guile — shim du kit V100, MIT (voir LICENSE + NOTICE.md a la racine).
// API CUPTI (NVIDIA) : les libs CUPTI ne sont pas redistribuees par ce kit.
// libcupath_v2.so — shim LD_PRELOAD CUPTI 12.9, FORMAT v2 (mission Kolbo E3)
//
// Objectif : exposer des INTERVALLES GPU réels (début/fin) pour calculer union,
// occupation, chevauchements — ce que le format v1 (seuls dur_ns + t d'émission
// CPU) ne permettait pas.
//
// Format binaire (fichier CUPATH_TRACE, PUR binaire, 56 octets/record, little-endian) :
//   offset 0   u64 seq           (0-based, continu, strictement incrémental)
//   offset 8   u64 gpu_start     (ns, domaine CUPTI — horloge GPU)
//   offset 16  u64 gpu_end       (ns, domaine CUPTI)
//   offset 24  u64 dur_ns        (DOIT == gpu_end - gpu_start ; gate de validation)
//   offset 32  u64 corr          (CUPTI correlationId ; 0 pour RESET/MARK)
//   offset 40  u32 stream        (0 = non applicable)
//   offset 44  u32 op            (0=KER 1=CPY 2=RT 3=RESET 4=MARK)
//   offset 48  u64 aux           (RT: cbid ; MARK: real_us ; RESET: 0 ; KER/CPY: 0)
//
// Fichier texte .hdr (1 ligne, métadonnées de l'en-tête de capture) :
//   cupath_v2 pid=%d t0_real_us=%llu t0_mono_ns=%llu cupti=%d.%d.%d driver=%d.%d
//     t0_real_us  : CLOCK_REALTIME au démarrage (ancrage mur)
//     t0_mono_ns  : CLOCK_MONOTONIC au démarrage (ancrage mono)
//
// Fichier texte .names (1 ligne par record KER/CPY/RT — ALIGNEMENT 1:1 avec les
// records binaires d'op 0/1/2, dans l'ordre) :
//   <op> seq=%llu gpu_s=%llu gpu_e=%llu dur_ns=%llu stream=%u corr=%llu t_real_us=%llu name=%s
//   (MARK/RESET n'ont pas de ligne .names ; l'analyseur compte KER+CPY+RT == lignes)
//
// Notes de domaine d'horloges :
//   - start/end de CUPTI (KERNEL, MEMCPY, RUNTIME API) sont dans le MÊME domaine
//     (ns, horloge CUPTI). On vérifie empiriquement (test v2) que ce domaine est
//     cohérent avec CLOCK_MONOTONIC à une constante près (bracket test).
//   - Le champ t_real_us des lignes .names est le temps RÉEL au flush (utilisable
//     comme ancrage grossier), PAS un timestamp GPU — comme en v1, jamais
//     interprété comme tel.
//
// Usage :
//   CUPATH_TRACE=/tmp/trace_v2 LD_PRELOAD=.../libcupath_v2.so <app CUDA>
// Compile :
//   gcc-13 -O2 -fPIC -shared -o libcupath_v2.so cupath_shim_v2.c \
//     -I<prefix>/include -L<prefix>/lib64 -lcupti -Wl,--no-as-needed -Wl,-rpath,<prefix>/lib64
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <time.h>
#include <cuda.h>
#include <cupti_result.h>
#include <cupti_activity.h>

#define REC_BYTES 56
static int g_fd = -1, g_nd = -1, g_nh = -1;
static uint64_t g_seq = 0;
static int g_on = 0;
static uint64_t g_t0_real_us = 0, g_t0_mono_ns = 0;

static inline long long now_mono_ns(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static inline long long now_real_us(void) {
  struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
  return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

static void emit(uint32_t op, uint64_t start, uint64_t end, uint64_t dur,
                 uint64_t corr, uint32_t stream, uint64_t aux) {
  if (g_fd < 0) return;
  uint8_t rec[REC_BYTES];
  memcpy(rec + 0,  &g_seq,   8);
  memcpy(rec + 8,  &start,   8);
  memcpy(rec + 16, &end,     8);
  memcpy(rec + 24, &dur,     8);
  memcpy(rec + 32, &corr,    8);
  memcpy(rec + 40, &stream,  4);
  memcpy(rec + 44, &op,      4);
  memcpy(rec + 48, &aux,     8);
  (void)write(g_fd, rec, REC_BYTES);
  g_seq++;
}

static void emit_name(uint32_t op, uint64_t seq, uint64_t start, uint64_t end,
                      uint64_t dur, uint32_t stream, uint64_t corr, const char* name) {
  if (g_nd < 0) return;
  const char* tag = op == 0 ? "KER" : op == 1 ? "CPY" : "RT";
  (void)dprintf(g_nd, "%s seq=%llu gpu_s=%llu gpu_e=%llu dur_ns=%llu stream=%u corr=%llu t_real_us=%llu name=%s\n",
                tag, (unsigned long long)seq, (unsigned long long)start,
                (unsigned long long)end, (unsigned long long)dur,
                stream, (unsigned long long)corr, (unsigned long long)now_real_us(),
                name ? name : "?");
}

static void on_buf_req(uint8_t** buffer, size_t* size, size_t* maxRecords) {
  (void)maxRecords;
  if (buffer) *buffer = (uint8_t*)malloc(16u * 1024u * 1024u);
  if (size) *size = 16u * 1024u * 1024u;
}

static void on_buf_done(CUcontext ctx, uint32_t streamId, uint8_t* buffer,
                        size_t size, size_t validSize) {
  (void)ctx; (void)streamId; (void)size;
  CUpti_Activity* rec = NULL;
  while (cuptiActivityGetNextRecord(buffer, validSize, &rec) == CUPTI_SUCCESS && rec) {
    uint64_t seq_slot = g_seq;  // valeur si ce record est émis
    switch (rec->kind) {
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
        const CUpti_ActivityKernel9* k = (const CUpti_ActivityKernel9*)rec;
        uint64_t s = k->start, e = k->end;
        uint64_t d = e > s ? e - s : 0;
        uint64_t corr = k->correlationId;
        emit(0, s, e, d, corr, k->streamId, 0);
        emit_name(0, seq_slot, s, e, d, k->streamId, corr, k->name);
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMCPY: {
        const CUpti_ActivityMemcpy6* m = (const CUpti_ActivityMemcpy6*)rec;
        uint64_t s = m->start, e = m->end;
        uint64_t d = e > s ? e - s : 0;
        uint64_t corr = m->correlationId;
        emit(1, s, e, d, corr, m->streamId, 0);
        char buf[160];
        snprintf(buf, sizeof buf, "memcpy kind=%d bytes=%llu", (int)m->copyKind,
                 (unsigned long long)m->bytes);
        emit_name(1, seq_slot, s, e, d, m->streamId, corr, buf);
        break;
      }
      case CUPTI_ACTIVITY_KIND_RUNTIME: {
        const CUpti_ActivityAPI* a = (const CUpti_ActivityAPI*)rec;
        uint64_t s = a->start, e = a->end;
        uint64_t d = e > s ? e - s : 0;
        uint64_t corr = a->correlationId;
        uint64_t seq_slot = g_seq;
        emit(2, s, e, d, corr, 0, (uint32_t)a->cbid);
        emit_name(2, seq_slot, s, e, d, 0, corr, "cudaLaunchKernel*");
        break;
      }
      default:
        break;
    }
  }
  free(buffer);
}

static void* drain_thread(void* arg) {
  (void)arg;
  while (g_on) {
    usleep(500 * 1000);
    if (!g_on) break;
    cuptiActivityFlushAll(0);
  }
  return NULL;
}

static void cupath_shutdown(void) {
  if (!g_on) return;
  g_on = 0;
  cuptiActivityFlushAll(1);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
  if (g_fd >= 0) { emit(3, 0, 0, 0, 0, 0, 0); close(g_fd); g_fd = -1; }
  if (g_nd >= 0) { close(g_nd); g_nd = -1; }
  if (g_nh >= 0) { close(g_nh); g_nh = -1; }
}

__attribute__((constructor))
static void cupath_init(void) {
  const char* p = getenv("CUPATH_TRACE");
  if (!p || !*p) return;
  // llama-server fork/exec : chaque processus trace dans <prefix>.<pid>.*
  char full[650];
  snprintf(full, sizeof full, "%s.%d", p, (int)getpid());
  g_fd = open(full, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (g_fd < 0) return;
  char np[650], hp[650];
  snprintf(np, sizeof np, "%s.names", full);
  snprintf(hp, sizeof hp, "%s.hdr", full);
  g_nd = open(np, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  g_nh = open(hp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  g_t0_real_us = (uint64_t)now_real_us();
  g_t0_mono_ns = (uint64_t)now_mono_ns();

  int dv = 0;
  (void)cuDriverGetVersion(&dv);
  if (g_nh >= 0) {
    (void)dprintf(g_nh, "cupath_v2 pid=%d t0_real_us=%llu t0_mono_ns=%llu driver=%d\n",
                  (int)getpid(), (unsigned long long)g_t0_real_us,
                  (unsigned long long)g_t0_mono_ns, dv);
  }

  CUptiResult r = cuptiActivityRegisterCallbacks(on_buf_req, on_buf_done);
  if (r != CUPTI_SUCCESS) {
    fprintf(stderr, "[cupath_v2] RegisterCallbacks: %d (shim inactif)\n", (int)r);
    close(g_fd); g_fd = -1;
    if (g_nd >= 0) close(g_nd), g_nd = -1;
    if (g_nh >= 0) close(g_nh), g_nh = -1;
    return;
  }
  r = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (r != CUPTI_SUCCESS) {
    fprintf(stderr, "[cupath_v2] enable KERNEL: %d (shim inactif)\n", (int)r);
    close(g_fd); g_fd = -1;
    if (g_nd >= 0) close(g_nd), g_nd = -1;
    return;
  }
  (void)cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  (void)cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
  g_on = 1;
  atexit(cupath_shutdown);
  static pthread_t th;
  (void)pthread_create(&th, NULL, drain_thread, NULL);
  pthread_detach(th);
  fprintf(stderr, "[cupath_v2] tracing v2 actif -> %s\n", full);
}
