/* vk_window.h on the runtime's own window (backend_win32.c), in place of pokopia-native's SDL3 window. The window thread, its messages and the input stay ours. */
#include "vk_window.h"
#include "vk_internal.h"
#include <windows.h>
#include <stdio.h>
static HWND g_h; static volatile LONG g_serial; static int g_w, g_h_px;
void nvn_window_set(void *hwnd) { g_h = (HWND)hwnd; }
void nvn_window_resized(int w, int h) { g_w = w; g_h_px = h; InterlockedIncrement(&g_serial); }
int vkw_open(const char *title, int width, int height, int hidden) {
    (void)title; (void)width; (void)height; (void)hidden;
    if (!g_h) { VKRED("no window"); return -1; }
    RECT r; GetClientRect(g_h, &r); g_w = r.right - r.left; g_h_px = r.bottom - r.top;
    VKLOG("window %dx%d pixels", g_w, g_h_px);
    return 0;
}
void vkw_close(void) {}
int vkw_is_open(void) { return g_h != NULL; }
const char *const *vkw_instance_exts(uint32_t *n) {
    static const char *const e[] = {VK_KHR_SURFACE_EXTENSION_NAME, "VK_KHR_win32_surface"};
    *n = 2; return e;
}
int vkw_create_surface(VkInstance inst, VkSurfaceKHR *out) {
    VkWin32SurfaceCreateInfoKHR c = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    c.hinstance = GetModuleHandleW(NULL); c.hwnd = g_h;
    PFN_vkCreateWin32SurfaceKHR f = (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(inst, "vkCreateWin32SurfaceKHR");
    if (!f || f(inst, &c, NULL, out) != VK_SUCCESS) { VKRED("vkCreateWin32SurfaceKHR failed"); return -1; }
    return 0;
}
void vkw_pixel_size(int *w, int *h) { *w = g_w; *h = g_h_px; }
uint32_t vkw_resize_serial(void) { return (uint32_t)g_serial; }
int vkw_minimized(void) { return IsIconic(g_h) != 0; }
int vkw_display_hz(void) { DEVMODEA m; memset(&m, 0, sizeof m); m.dmSize = sizeof m; return EnumDisplaySettingsA(NULL, ENUM_CURRENT_SETTINGS, &m) ? (int)m.dmDisplayFrequency : 0; }
void vkw_request_size(int w, int h) { (void)w; (void)h; }
void vkw_request_fullscreen(int on) { (void)on; }
void vkw_set_close_handler(void (*fn)(void)) { (void)fn; }
void vkw_set_title(const char *title) { (void)title; }
