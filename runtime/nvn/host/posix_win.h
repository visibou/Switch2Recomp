/* The subset of POSIX the ported nxnative files use, over Win32, so they stay close to the originals:
   pthread mutexes (SRWLOCK), condition variables (CONDITION_VARIABLE), once (INIT_ONCE), keys (FLS, with destructors),
   threads (_beginthreadex, identified by thread id), clock_gettime, nanosleep, localtime_r/gmtime_r, strlcpy.
   Mutexes are not recursive, as nxnative's default pthread mutexes are not. */
#ifndef NXRT_POSIX_WIN_H
#define NXRT_POSIX_WIN_H
#include <windows.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "plat.h"

typedef SRWLOCK pthread_mutex_t;
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
static inline int pthread_mutex_init(pthread_mutex_t *m, const void *a) { (void)a; InitializeSRWLock(m); return 0; }
static inline int pthread_mutex_destroy(pthread_mutex_t *m) { (void)m; return 0; }
static inline int pthread_mutex_lock(pthread_mutex_t *m) { AcquireSRWLockExclusive(m); return 0; }
static inline int pthread_mutex_trylock(pthread_mutex_t *m) { return TryAcquireSRWLockExclusive(m) ? 0 : EBUSY; }
static inline int pthread_mutex_unlock(pthread_mutex_t *m) { ReleaseSRWLockExclusive(m); return 0; }

typedef CONDITION_VARIABLE pthread_cond_t;
#define PTHREAD_COND_INITIALIZER CONDITION_VARIABLE_INIT
static inline int pthread_cond_init(pthread_cond_t *c, const void *a) { (void)a; InitializeConditionVariable(c); return 0; }
static inline int pthread_cond_destroy(pthread_cond_t *c) { (void)c; return 0; }
static inline int pthread_cond_signal(pthread_cond_t *c) { WakeConditionVariable(c); return 0; }
static inline int pthread_cond_broadcast(pthread_cond_t *c) { WakeAllConditionVariable(c); return 0; }
static inline int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) { SleepConditionVariableSRW(c, m, INFINITE, 0); return 0; }

#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
#undef clock_gettime
#define clock_gettime nxrt_clock_gettime
static inline int clock_gettime(int clk, struct timespec *ts) {
    uint64_t ns = clk == CLOCK_MONOTONIC ? plat_now_ns() : plat_realtime_ns();
    ts->tv_sec = (time_t)(ns / 1000000000u); ts->tv_nsec = (long)(ns % 1000000000u);
    return 0;
}
static inline uint64_t clock_gettime_nsec_np(int clk) { return clk == CLOCK_MONOTONIC ? plat_now_ns() : plat_realtime_ns(); }
/* Absolute CLOCK_REALTIME deadline, as POSIX; waits in whole milliseconds (the process runs at a 1 ms timer resolution) */
static inline int pthread_cond_timedwait(pthread_cond_t *c, pthread_mutex_t *m, const struct timespec *abs) {
    uint64_t dl = (uint64_t)abs->tv_sec * 1000000000u + (uint64_t)abs->tv_nsec, now = plat_realtime_ns();
    if (dl <= now) return ETIMEDOUT;
    uint64_t ms = (dl - now + 999999) / 1000000;
    if (!SleepConditionVariableSRW(c, m, ms >= INFINITE ? INFINITE - 1 : (DWORD)ms, 0)) return GetLastError() == ERROR_TIMEOUT ? ETIMEDOUT : EINVAL;
    return 0;
}
#define nanosleep nxrt_nanosleep
static inline int nanosleep(const struct timespec *ts, struct timespec *rem) {
    (void)rem; plat_sleep_ns((uint64_t)ts->tv_sec * 1000000000u + (uint64_t)ts->tv_nsec); return 0;
}
static inline int usleep(unsigned us) { plat_sleep_ns((uint64_t)us * 1000u); return 0; }
static inline unsigned sleep(unsigned s) { Sleep(s * 1000u); return 0; }
static inline int sched_yield(void) { SwitchToThread(); return 0; }
static inline int getpagesize(void) { return 4096; }

typedef INIT_ONCE pthread_once_t;
#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT
static BOOL CALLBACK pthread_once_tramp_(PINIT_ONCE o, PVOID fn, PVOID *ctx) { (void)o; (void)ctx; ((void (*)(void))fn)(); return TRUE; }
static inline int pthread_once(pthread_once_t *o, void (*fn)(void)) { InitOnceExecuteOnce(o, pthread_once_tramp_, (PVOID)fn, NULL); return 0; }

/* Keys are fiber-local slots: FlsAlloc takes a destructor, which runs at thread exit as a pthread key destructor does */
typedef DWORD pthread_key_t;
static inline int pthread_key_create(pthread_key_t *k, void (*dtor)(void *)) {
    *k = FlsAlloc((PFLS_CALLBACK_FUNCTION)(void *)dtor); return *k == FLS_OUT_OF_INDEXES ? EAGAIN : 0;
}
static inline void *pthread_getspecific(pthread_key_t k) { return FlsGetValue(k); }
static inline int pthread_setspecific(pthread_key_t k, const void *v) { return FlsSetValue(k, (PVOID)v) ? 0 : EINVAL; }

/* Threads are named by their thread id */
typedef uint32_t pthread_t;
typedef struct { size_t stack; int detached; } pthread_attr_t;
#define PTHREAD_CREATE_DETACHED 1
static inline int pthread_attr_init(pthread_attr_t *a) { a->stack = 0; a->detached = 0; return 0; }
static inline int pthread_attr_destroy(pthread_attr_t *a) { (void)a; return 0; }
static inline int pthread_attr_setstacksize(pthread_attr_t *a, size_t n) { a->stack = n; return 0; }
static inline int pthread_attr_setdetachstate(pthread_attr_t *a, int d) { a->detached = d; return 0; }
typedef struct { void *(*fn)(void *); void *arg; } pthread_start_;
static unsigned pthread_tramp_(void *p) {
    pthread_start_ s = *(pthread_start_ *)p; HeapFree(GetProcessHeap(), 0, p); s.fn(s.arg); return 0;
}
static inline int pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*fn)(void *), void *arg) {
    pthread_start_ *s = HeapAlloc(GetProcessHeap(), 0, sizeof *s);
    if (!s) return EAGAIN;
    s->fn = fn; s->arg = arg;
    uint32_t id = plat_thread_start(pthread_tramp_, s, a && a->stack ? a->stack : (1u << 20));
    if (!id) { HeapFree(GetProcessHeap(), 0, s); return EAGAIN; }
    *t = id; return 0;
}
static inline int pthread_detach(pthread_t t) { (void)t; return 0; }
static inline int pthread_join(pthread_t t, void **res) {
    if (res) *res = NULL;
    HANDLE h = OpenThread(SYNCHRONIZE, FALSE, t);
    if (!h) return 0;   /* already gone */
    WaitForSingleObject(h, INFINITE); CloseHandle(h); return 0;
}
static inline pthread_t pthread_self(void) { return GetCurrentThreadId(); }
static inline int pthread_setname_np(const char *name) { plat_thread_name(name); return 0; }

static inline struct tm *localtime_r(const time_t *t, struct tm *out) { return localtime_s(out, t) ? NULL : out; }
static inline struct tm *gmtime_r(const time_t *t, struct tm *out) { return gmtime_s(out, t) ? NULL : out; }
static inline time_t timegm(struct tm *tm) { return _mkgmtime(tm); }
static inline size_t strlcpy(char *d, const char *s, size_t n) {
    size_t l = strlen(s);
    if (n) { size_t c = l < n - 1 ? l : n - 1; memcpy(d, s, c); d[c] = 0; }
    return l;
}
#define flockfile _lock_file
typedef unsigned useconds_t;
static inline int setenv(const char *n, const char *v, int overwrite) { if (!overwrite && getenv(n)) return 0; return _putenv_s(n, v) ? -1 : 0; }
static inline int unsetenv(const char *n) { return _putenv_s(n, "") ? -1 : 0; }
/* BSD's mergesort: a stable sort (hle_hid.c's DM_KEYS steps) */
static inline int mergesort(void *base, size_t n, size_t sz, int (*cmp)(const void *, const void *)) {
    if (n < 2) return 0;
    unsigned char *a = (unsigned char *)base, *t = (unsigned char *)malloc(n * sz);
    if (!t) return -1;
    for (size_t w = 1; w < n; w *= 2) {
        for (size_t lo = 0; lo < n; lo += 2 * w) {
            size_t mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, i = lo, j = mid, k = lo;
            while (i < mid && j < hi) { if (cmp(a + j * sz, a + i * sz) < 0) memcpy(t + k++ * sz, a + j++ * sz, sz); else memcpy(t + k++ * sz, a + i++ * sz, sz); }
            while (i < mid) memcpy(t + k++ * sz, a + i++ * sz, sz);
            while (j < hi) memcpy(t + k++ * sz, a + j++ * sz, sz);
        }
        memcpy(a, t, n * sz);
    }
    free(t);
    return 0;
}
#define funlockfile _unlock_file

#endif
