/* Timeline trace (SWR_TRACE=<start s>[,<seconds>]): spans and counters from every thread, the runtime's and Eden's
   (C and C++), plus the host GPU's work (from Vulkan timestamps), written as trace.json next to the exe (Chrome trace
   format: open it in ui.perfetto.dev). Off outside the window: one load of g_swr_trace_on per call.
   Names must be string literals (only the pointer is kept). */
#ifndef SWR_TRACE_H
#define SWR_TRACE_H
#ifdef __cplusplus
extern "C" {
#endif

extern volatile int g_swr_trace_on;

unsigned long long swr_trace_ts(void);                                   /* now, in trace ticks (TSC) */
unsigned long long swr_run_ns(void);                                     /* nanoseconds since the process started */
void swr_trace_span(const char* name, unsigned long long t0);           /* [t0, now] on this thread */
void swr_trace_span_arg(const char* name, unsigned long long t0, unsigned long long arg);   /* arg shown in the name */
void swr_trace_counter(const char* name, long long value);
void swr_trace_thread_name(const char* name);                            /* this thread's track name */
/* host GPU work, on its own track, in GPU nanoseconds; swr_trace_gpu_sync aligns the GPU clock to the trace's:
   a submission made at CPU tick cpu_submit started on the GPU at gpu_start_ns (never earlier) */
void swr_trace_gpu_span(const char* name, unsigned long long gpu_t0_ns, unsigned long long gpu_t1_ns,
                        unsigned long long arg, int track);
void swr_trace_gpu_sync(unsigned long long cpu_submit, unsigned long long gpu_start_ns);
/* starts and stops the window, writes the file once it is over: call it often (each present) */
void swr_trace_poll(void);

#ifdef __cplusplus
}
#endif
#endif
