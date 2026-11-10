/* C bridge to the Eden GPU backend (runtime/eden, SWR_GPU=eden). Only linked when build/eden exists
   (switchrecomp.py defines SWR_HAVE_EDEN then). The runtime calls these through gpu_eden.c. */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

/* Start Eden's GPU (Vulkan renderer) on the game window. guest_mem/guest_size: the game's address space (g_mem);
   data_dir: folder for Eden's log and caches. Returns 1 on success. */
int eden_gpu_start(void* hwnd, int width, int height, void* guest_mem, unsigned long long guest_size,
                   const char* data_dir, unsigned long long program_id);
/* internal render scale (0.25 .. 2), before eden_gpu_start */
void eden_gpu_set_resolution(float scale);
/* GMMU: GPU address -> guest address (4 KiB pages) */
void eden_gpu_map(unsigned long long gva, unsigned long long cpu, unsigned long long size);
void eden_gpu_unmap(unsigned long long gva, unsigned long long size);
/* GPU-only memory (sparse regions): the CPU cannot write it, so CPU writes are not looked for there */
void eden_gpu_gpu_only(unsigned long long cpu, unsigned long long size, int on);
/* GPU-accessible memory created (1) or destroyed (0) by the game */
void eden_gpu_memory(unsigned long long start, unsigned long long size, int created);
/* GPFIFO entries of a runtime channel (keyed by its syncpoint); incs increments of that syncpoint after the work,
   each raising the runtime's counter when done; wait_id ~0u: no fence to wait for */
void eden_gpu_submit(const unsigned long long* entries, unsigned n, unsigned syncpt, unsigned incs,
                     unsigned wait_id, unsigned wait_val);
/* before a present: wait until Eden processed the previous frame (and this one when it is behind) */
void eden_gpu_wait_ns(unsigned long long* front, unsigned long long* back);
void eden_gpu_present_wait(unsigned fence_id, unsigned fence_value);
/* a syncpoint the CPU side incremented (Eden keeps the runtime's values) */
void eden_gpu_syncpt_cpu_incr(unsigned id);
/* show a presented guest image (Switch 2: RGBA8, block linear) */
void eden_gpu_present_image(unsigned long long addr, unsigned w, unsigned h, unsigned pitch, unsigned layout,
                            unsigned bhl, unsigned fmt, int crop_right, int crop_bottom);
void eden_gpu_present(void);
/* the next composited frame as <path>.ppm; returns a hash of it (0: none within 10 s) */
unsigned long long eden_gpu_screenshot(const char* path);
void eden_gpu_resize(int width, int height);
void eden_gpu_stop(void);

#ifdef __cplusplus
}
#endif
