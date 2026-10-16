// libcupath.so — shim LD_PRELOAD : tracing CUPTI 12.9 in-process (préfixe isolé)
// Objectif : observer les kernels (nom, grid, stream, durées) d'un processus CUDA
// existant SANS le modifier. Cible de validation : llama-server (verify MTP, ggml-cuda).
//
// Mécanisme : constructeur LD_PRELOAD -> cuptiActivityRegisterCallbacks + enable
// CONCURRENT_KERNEL/MEMCPY. Thread daemon : flush toutes les 500 ms (drain non-forcé).
// Sortie (fichier H2, lignes 16 octets, O_APPEND, immédiat) :
//   u64 seq | u64 dur_ns | u32 stream | u32 op
//   op: 0=kernel 1=memcpy 2=reset ; les noms des kernels (clairs) sont dans le
//   fichier CUPATH_NAMES (une ligne texte par record kernel/memcpy).
//
// Usage :
//   CUPATH_TRACE=/tmp/trace.bin LD_PRELOAD=/path/libcupath.so <app CUDA>
// Compile : gcc-13 -O2 -fPIC -shared -o libcupath.so cupath_shim.c \
//   -I <prefix>/include -L <prefix>/lib64 -lcupti -Wl,--no-as-needed -Wl,-rpath,<prefix>/lib64
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

static int  g_fd = -1;
static int  g_nd = -1;
static uint64_t g_seq = 0;
static int  g_on = 0;

static inline long long now_ns(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// horloge REELLE (us) pour correlater avec date/TS du script
static inline long long now_real_us(void) {
  struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
  return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

static void emit(uint32_t op, uint64_t dur_ns, uint32_t stream) {
  if (g_fd < 0) return;
  uint8_t rec[24];
  memcpy(rec + 0, &g_seq, 8);
  memcpy(rec + 8, &dur_ns, 8);
  memcpy(rec + 16, &stream, 4);
  uint32_t o = op;
  memcpy(rec + 20, &o, 4);
  (void)write(g_fd, rec, 24);
  g_seq++;
}

static void emit_name(const char* kind, uint64_t dur_ns, uint32_t stream, const char* name) {
  if (g_nd < 0) return;
  (void)dprintf(g_nd, "%s t=%lld stream=%u dur_ns=%llu name=%s\n", kind,
                (long long)now_real_us(), stream, (unsigned long long)dur_ns, name ? name : "?");
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
    switch (rec->kind) {
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
        const CUpti_ActivityKernel9* k = (const CUpti_ActivityKernel9*)rec;
        uint64_t d = k->end > k->start ? k->end - k->start : 0;
        emit(0, d, k->streamId);
        emit_name("KER", d, k->streamId, k->name);
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMCPY: {
        const CUpti_ActivityMemcpy6* m = (const CUpti_ActivityMemcpy6*)rec;
        uint64_t d = m->end > m->start ? m->end - m->start : 0;
        emit(1, d, 0);
        char buf[128];
        snprintf(buf, sizeof buf, "memcpy kind=%d bytes=%llu", (int)m->copyKind,
                 (unsigned long long)m->bytes);
        emit_name("CPY", d, 0, buf);
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
    cuptiActivityFlushAll(0); // non-forcé : les records complets
  }
  return NULL;
}

static void cupath_shutdown(void) {
  if (!g_on) return;
  g_on = 0;
  cuptiActivityFlushAll(1); // forcé : les records incomplets
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
  if (g_fd >= 0) { emit(2, 0, 0); close(g_fd); g_fd = -1; }
  if (g_nd >= 0) { close(g_nd); g_nd = -1; }
}

__attribute__((constructor))
static void cupath_init(void) {
  const char* p = getenv("CUPATH_TRACE");
  if (!p || !*p) return;
  g_fd = open(p, O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (g_fd < 0) return;
  char np[512];
  snprintf(np, sizeof np, "%s.names", p);
  g_nd = open(np, O_WRONLY | O_CREAT | O_APPEND, 0644);

  CUptiResult r = cuptiActivityRegisterCallbacks(on_buf_req, on_buf_done);
  if (r != CUPTI_SUCCESS) {
    fprintf(stderr, "[cupath] RegisterCallbacks: %d (shim inactif)\n", (int)r);
    close(g_fd); g_fd = -1;
    if (g_nd >= 0) close(g_nd), g_nd = -1;
    return;
  }
  r = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (r != CUPTI_SUCCESS) {
    fprintf(stderr, "[cupath] enable KERNEL: %d (shim inactif)\n", (int)r);
    close(g_fd); g_fd = -1;
    if (g_nd >= 0) close(g_nd), g_nd = -1;
    return;
  }
  (void)cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  g_on = 1;
  atexit(cupath_shutdown);
  static pthread_t th;
  (void)pthread_create(&th, NULL, drain_thread, NULL);
  pthread_detach(th);
  char hdr[256];
  int n = snprintf(hdr, sizeof hdr, "[cupath] pid=%d t=%lld trace=%s\n",
                   (int)getpid(), (long long)now_ns(), p);
  (void)write(g_fd, hdr, n);
  fprintf(stderr, "[cupath] tracing actif -> %s\n", p);
}
