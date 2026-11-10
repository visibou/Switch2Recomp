// The game window (SDL3, DESIGN D14): resizable, high-DPI aware, on a thread of its own that pumps its events. The renderer draws into it at any size.
#pragma once
#include "vk_fn.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int vkw_open(const char *title, int width, int height, int hidden);   // starts the window thread; 0 = the window exists, -1 = no SDL / no display (logged)
void vkw_close(void);
int vkw_is_open(void);
const char *const *vkw_instance_exts(uint32_t *n);       // what the Vulkan instance needs for this window system
int vkw_create_surface(VkInstance inst, VkSurfaceKHR *out);
void vkw_pixel_size(int *w, int *h);                     // the drawable size in pixels
uint32_t vkw_resize_serial(void);                        // changes whenever the drawable size changes
int vkw_minimized(void);                                 // minimized or occluded: presents are dropped, not waited for
int vkw_display_hz(void);                                // refresh rate of the window's display (0 = unknown)
void vkw_request_size(int w, int h);                     // asks the window thread for a new client size (tests, resolution checks)
void vkw_request_fullscreen(int on);                     // borderless fullscreen on the window's display (Alt+Enter and F11 toggle it too)
void vkw_set_close_handler(void (*fn)(void));            // called on the window thread when the user closes the window (default: ExitProcess(0))
void vkw_set_title(const char *title);

#ifdef __cplusplus
}
#endif
