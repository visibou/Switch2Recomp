// The flight recorder and the control channel (docs/DEBUG_TOOLS.md).
// Flight recorder: the last frames as shown, and what each frame drew, kept in memory all the time. F5, "gamectl flight" or a timed trigger writes them to <exe folder>/flight/<time>/,
// read with tools/flight.py. NXRT_VK_FLIGHT=0 turns it off (benchmarks).
// Control: a named pipe, \\.\pipe\swr_ctl, one command per connection (tools/gamectl.py): pause, step, hide a shader, change a switch, take a RenderDoc capture, read memory.
#pragma once
#include "vk_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { FL_DRAWN = 1, FL_INDEXED = 2, FL_INDIRECT = 4, FL_MDI = 8, FL_MESH = 16, FL_RC_SKIP = 32, FL_HIDDEN = 64, FL_MDI_CUT = 128, FL_DISPATCH = 256 };
// One draw or dispatch. vs and ps: shader keys (a mesh draw's mesh shader in vs, a dispatch's compute shader in vs). a: the arguments as the CPU saw them (an indirect draw's record in guest memory;
// a dispatch's x, y, z in a[0..2]). g: an indirect draw's record and count as the GPU read them, filled in when the recording is written (flags tell whether they are there)
typedef struct {
    uint64_t vs, ps;
    uint32_t pass, flags, rt, mdi_i, mdi_cnt, binst, args_i, pad;
    uint32_t a[5], pad2;
    uint32_t g[6];
} vkf_row;

uint64_t vkf_key(const char *key);                                 // a shader's key as a number
vkf_row *vkf_row_next(void);                                       // NULL while off or being written
void vkf_note_args(vkf_row *r, uint64_t ind, uint32_t len, uint64_t cnt);   // read this record and count on the GPU when the pass ends
void vkf_args_flush(VkCommandBuffer cb);                           // no pass open
void vkf_frame(uint64_t frame);                                    // the rows so far are this frame's
void vkf_poll(void);                                               // present thread, between presents
int vkf_present(VkCommandBuffer cb, VkImage src, VkExtent2D ext, VkFormat fmt, uint64_t frame);   // src in TRANSFER_SRC_OPTIMAL; 1 when it recorded the copy
void vkf_trigger(const char *label);                               // any thread
const char *vkf_last(uint64_t *serial);                            // the last folder written and how many have been
int vkf_wants(void);
int vkf_flashes(long frames, char *out, size_t cap, int min_blocks);                 // the frames among the last ones where much of the picture changed for one frame and came back: nothing is written to disk

// The pass viewer ("gamectl passview N"): a thumbnail of what each pass and each dispatch wrote, for the next N frames, written to <exe folder>/flight/pv_<time>/ (tools/flight.py --pv)
struct vkt_image;
int vkv_on(void);
void vkv_image(VkCommandBuffer cb, struct vkt_image *im, uint32_t level, uint32_t layer, uint32_t w, uint32_t h, int slot, uint64_t key, int compute);
void vkv_frame(uint64_t frame);
void vkv_request(long frames);

// "gamectl watch N KEY...": for N frames, each uniform buffer a watched shader reads, as guest memory holds it when the draw is recorded and as the GPU had it when the pass ended; differences in the log
int vkw_wants(uint64_t key);
void vkw_note(VkCommandBuffer cb_unused, uint64_t key, int stage, int reg, uint64_t addr, uint32_t size);
void vkw_flush(VkCommandBuffer cb);
void vkw_frame(uint64_t frame);
void vkw_request(long frames, const uint64_t *keys, int n);

// "gamectl drawstats N": how many pixel shader runs each draw of the next N frames made, on the GPU; a line per draw in <exe folder>/flight/ds_<time>.txt (tools/flight.py --ds)
int vkq_on(void);
void vkq_begin(VkCommandBuffer cb, uint64_t vs, uint64_t ps, uint32_t count, int32_t base_v, uint64_t ibuf, uint32_t pass_i, uint32_t rt, int occ);   // occ: also counts samples (no game occlusion query is open), which sees depth-only draws
void vkq_close(VkCommandBuffer cb);
void vkq_frame(uint64_t frame);
void vkq_request(long frames);
void vkq_reset_if(VkCommandBuffer cb);   // outside a pass
int vkq_resetting(void);

void vkc_start(void);                                              // the pipe's thread
int vkc_int(const char *name, int def);                            // a switch: its value from "gamectl set", else NXRT_VK_<NAME>, else def
int vkc_any_hidden(void);
int vkc_hidden(uint64_t key);                                      // "gamectl hide": a shader whose draws are not made
void vkc_gate(uint64_t frame);                                     // the game's present passes here: "gamectl pause" holds it
void vkc_rdoc_load(void);                                          // SWR_RDOC=1: before the Vulkan instance

#ifdef __cplusplus
}
#endif
