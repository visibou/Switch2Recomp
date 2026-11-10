// Presentation: a swapchain on the SDL3 window (FIFO by default, mailbox or immediate on request) and a present thread that shows each game
// present at the game's pace, scaled to the window with the aspect kept (resolution independent). gfx_present / gfx_present_ext end here.
#pragma once
#include "vk_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { VKP_FIFO = 0, VKP_MAILBOX = 1, VKP_IMMEDIATE = 2 };
// What a present shows: the image's view in the layout it rests in between submissions, its size, whether its content is sRGB-encoded in a UNORM view
typedef struct { VkImageView view; VkImageLayout layout; uint32_t w, h; } vkp_source;
typedef int (*vkp_resolve_fn)(uint64_t serial, vkp_source *out);   // 1 = the serial has a host image with content (vk_tex.c)

int vkp_init(int mode);                     // after vk_dev_open and vkw_open; creates the swapchain and starts the present thread
void vkp_shutdown(void);
void vkp_set_resolver(vkp_resolve_fn fn);
// Enqueue one present; returns its ticket at once. serial 0 = nothing to show (the test pattern); wait_value: GPU timeline value the frame's work signals;
// period: the least time the previous image stays on screen (s, 0 = none); crop: x, y, w, h in the texture (NULL = all of it); flip: GL's lower-left origin
uint64_t vkp_enqueue(uint64_t serial, double period, uint64_t wait_value, const int crop[4], int flip);
void vkp_wait(uint64_t ticket);             // until that present has been handed to the GPU (or dropped)
void vkp_flush(void);                       // until every enqueued present has
uint64_t vkp_count(void);                   // presents handed to the GPU so far
uint64_t vkp_dropped(void);                 // presents dropped (window minimized or occluded, swapchain unusable)
double vkp_presented_time(uint64_t n);      // seconds since start at which present n (from 1) was handed to the GPU; 0 = not yet, -1 = dropped
void vkp_set_stall(uint64_t ns);            // unit tests: an extra pause before each present
// Tests: copy the next presented frame (after scaling) into rgba (BGRA or RGBA as the swapchain format says; *bgr tells); 0 on success
int vkp_capture_next(uint32_t **pixels, int *w, int *h, int *bgr, uint32_t timeout_ms);
void vkp_info(int *w, int *h, int *mode, VkFormat *fmt);
void vkp_stats_line(char *out, size_t n);

#ifdef __cplusplus
}
#endif
