/* FFmpeg stand-ins for the Eden link test: video decoding (NVDEC) reports "no decoder". */
#include <stddef.h>
#include <string.h>
#define ENOSYS_ (-38)
void* av_buffer_ref(const void* b) { (void)b; return NULL; }
void av_buffer_unref(void** b) { if (b) *b = NULL; }
void* av_frame_alloc(void) { return NULL; }
void av_frame_free(void** f) { if (f) *f = NULL; }
void av_freep(void* p) { (void)p; }
int av_hwdevice_ctx_create(void** c, int t, const char* d, void* o, int f) { (void)c;(void)t;(void)d;(void)o;(void)f; return ENOSYS_; }
const char* av_hwdevice_get_type_name(int t) { (void)t; return NULL; }
int av_hwdevice_iterate_types(int prev) { (void)prev; return 0; }
int av_hwframe_transfer_data(void* d, const void* s, int f) { (void)d;(void)s;(void)f; return ENOSYS_; }
void* av_mallocz(size_t n) { (void)n; return NULL; }
int av_opt_set(void* o, const char* n, const char* v, int f) { (void)o;(void)n;(void)v;(void)f; return ENOSYS_; }
void* av_packet_alloc(void) { return NULL; }
void av_packet_free(void** p) { if (p) *p = NULL; }
const void* av_pix_fmt_desc_get(int f) { (void)f; return NULL; }
int av_strerror(int e, char* buf, size_t n) { (void)e; if (n) strncpy(buf, "ffmpeg not linked", n - 1), buf[n - 1] = 0; return 0; }
void* avcodec_alloc_context3(const void* c) { (void)c; return NULL; }
const void* avcodec_find_decoder(int id) { (void)id; return NULL; }
void avcodec_free_context(void** c) { if (c) *c = NULL; }
const void* avcodec_get_hw_config(const void* c, int i) { (void)c;(void)i; return NULL; }
int avcodec_open2(void* c, const void* codec, void** o) { (void)c;(void)codec;(void)o; return ENOSYS_; }
int avcodec_receive_frame(void* c, void* f) { (void)c;(void)f; return ENOSYS_; }
int avcodec_send_packet(void* c, const void* p) { (void)c;(void)p; return ENOSYS_; }
