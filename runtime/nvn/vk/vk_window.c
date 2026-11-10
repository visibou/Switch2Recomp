// The SDL3 window and its event thread (vk_window.h). Written from SDL3's public headers (zlib licence; SDL3.dll is loaded on first use, /DELAYLOAD).
// M6: the thread also starts SDL's controller subsystem and hands every event to the front end (front/front_sdl.c: keyboard, mouse look, controllers).
#include "vk_window.h"
#include "vk_internal.h"
#include "front_sdl.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <process.h>
#include <stdio.h>
#include <string.h>

static struct {
    SDL_Window *win;
    HANDLE thread, ready;
    _Atomic int open, minimized, closing, w, h, hz;
    _Atomic uint32_t resize_serial;
    uint32_t ev_req;                 // our user event: data1/data2 = width/height, code = 1 resize, 2 fullscreen, 3 title, 4 quit
    void (*on_close)(void);
    int fail;
    char title[128];
    int req_w, req_h, hidden, front;
} g_w;

static void refresh_size(void) {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(g_w.win, &w, &h);
    if (w != atomic_load(&g_w.w) || h != atomic_load(&g_w.h)) { atomic_store(&g_w.w, w); atomic_store(&g_w.h, h); atomic_fetch_add(&g_w.resize_serial, 1); }
    SDL_DisplayID d = SDL_GetDisplayForWindow(g_w.win);
    const SDL_DisplayMode *m = d ? SDL_GetCurrentDisplayMode(d) : NULL;
    atomic_store(&g_w.hz, m && m->refresh_rate > 0 ? (int)(m->refresh_rate + 0.5f) : 0);
}

static unsigned __stdcall window_main(void *arg) {
    (void)arg;
    SDL_SetHint(SDL_HINT_VIDEO_ALLOW_SCREENSAVER, "0");
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");   // the name box takes the focus and still hears the controller; the game itself gets nothing while unfocused (front_sdl.c's input_on)
    front_sdl_lock(); int vok = SDL_Init(SDL_INIT_VIDEO); front_sdl_unlock();
    if (!vok) { VKRED("SDL_Init(VIDEO) failed: %s", SDL_GetError()); g_w.fail = 1; SetEvent(g_w.ready); return 0; }
    if (!SDL_Vulkan_LoadLibrary(NULL)) { VKRED("SDL cannot load the Vulkan loader: %s", SDL_GetError()); g_w.fail = 1; SetEvent(g_w.ready); return 0; }
    SDL_WindowFlags fl = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | (g_w.hidden ? SDL_WINDOW_HIDDEN : 0);
    g_w.win = SDL_CreateWindow(g_w.title, g_w.req_w, g_w.req_h, fl);
    if (!g_w.win) { VKRED("SDL_CreateWindow failed: %s", SDL_GetError()); g_w.fail = 1; SetEvent(g_w.ready); return 0; }
    SDL_SetWindowMinimumSize(g_w.win, 320, 180);
    g_w.ev_req = SDL_RegisterEvents(1);
    if (!g_w.hidden) {   // the game's window (the tests' hidden ones take no input)
        front_sdl_lock(); int gok = SDL_InitSubSystem(SDL_INIT_GAMEPAD); front_sdl_unlock();
        if (!gok) VKRED("SDL cannot start its controller subsystem: %s (keyboard and mouse only)", SDL_GetError());
        front_sdl_start(g_w.win); g_w.front = 1;
    }
    refresh_size();
    atomic_store(&g_w.open, 1);
    SetEvent(g_w.ready);
    while (!atomic_load(&g_w.closing)) {
        SDL_Event e;
        if (g_w.front) front_sdl_tick();
        if (!SDL_WaitEventTimeout(&e, 50)) continue;
        do {
            if (g_w.front) front_sdl_event(&e);
            switch (e.type) {
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: case SDL_EVENT_WINDOW_RESIZED: case SDL_EVENT_WINDOW_DISPLAY_CHANGED: refresh_size(); break;
            case SDL_EVENT_WINDOW_MINIMIZED: atomic_store(&g_w.minimized, 1); break;   // covered (occluded) windows keep presenting: DWM still composes them
            case SDL_EVENT_WINDOW_RESTORED: case SDL_EVENT_WINDOW_EXPOSED: case SDL_EVENT_WINDOW_SHOWN: case SDL_EVENT_WINDOW_MAXIMIZED:
                atomic_store(&g_w.minimized, 0); refresh_size(); break;
            case SDL_EVENT_KEY_DOWN:
                if ((e.key.key == SDLK_RETURN && (e.key.mod & SDL_KMOD_ALT)) || e.key.key == SDLK_F11)
                    SDL_SetWindowFullscreen(g_w.win, !(SDL_GetWindowFlags(g_w.win) & SDL_WINDOW_FULLSCREEN));
                break;
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED: case SDL_EVENT_QUIT:
                VKLOG("window closed by the user");
                if (g_w.on_close) g_w.on_close(); else ExitProcess(0);
                break;
            default:
                if (e.type == g_w.ev_req) {
                    if (e.user.code == 1) { SDL_RestoreWindow(g_w.win); SDL_SetWindowFullscreen(g_w.win, false); SDL_SetWindowSize(g_w.win, (int)(intptr_t)e.user.data1, (int)(intptr_t)e.user.data2); SDL_SyncWindow(g_w.win); refresh_size(); }
                    else if (e.user.code == 2) { SDL_SetWindowFullscreen(g_w.win, e.user.data1 != NULL); SDL_SyncWindow(g_w.win); refresh_size(); }
                    else if (e.user.code == 3) SDL_SetWindowTitle(g_w.win, g_w.title);
                }
            }
        } while (SDL_PollEvent(&e));
    }
    atomic_store(&g_w.open, 0);
    SDL_DestroyWindow(g_w.win); g_w.win = NULL; g_w.front = 0;
    front_sdl_lock(); SDL_Quit(); front_sdl_unlock();
    return 0;
}

int vkw_open(const char *title, int width, int height, int hidden) {
    if (atomic_load(&g_w.open)) return 0;
    snprintf(g_w.title, sizeof g_w.title, "%s", title ? title : "pokopia-native");
    g_w.req_w = width > 0 ? width : 1280; g_w.req_h = height > 0 ? height : 720; g_w.hidden = hidden; g_w.fail = 0;
    // SDL3.dll is delay-loaded: a missing DLL raises an exception here instead of stopping nxrt.exe at start
    __try { SDL_GetVersion(); } __except (EXCEPTION_EXECUTE_HANDLER) { VKRED("SDL3.dll is missing next to the executable: no window"); return -1; }
    g_w.ready = CreateEventW(NULL, TRUE, FALSE, NULL);
    g_w.thread = (HANDLE)_beginthreadex(NULL, 0, window_main, NULL, 0, NULL);
    if (!g_w.thread) return -1;
    WaitForSingleObject(g_w.ready, INFINITE);
    if (g_w.fail) { WaitForSingleObject(g_w.thread, 5000); CloseHandle(g_w.thread); g_w.thread = NULL; return -1; }
    VKLOG("window %dx%d pixels (%s), display %d Hz", atomic_load(&g_w.w), atomic_load(&g_w.h), hidden ? "hidden" : "shown", atomic_load(&g_w.hz));
    return 0;
}

void vkw_close(void) {
    if (!g_w.thread) return;
    atomic_store(&g_w.closing, 1);
    SDL_Event e; memset(&e, 0, sizeof e); e.type = g_w.ev_req; e.user.code = 4; SDL_PushEvent(&e);
    WaitForSingleObject(g_w.thread, 5000); CloseHandle(g_w.thread); g_w.thread = NULL; atomic_store(&g_w.closing, 0);
}

int vkw_is_open(void) { return atomic_load(&g_w.open); }
const char *const *vkw_instance_exts(uint32_t *n) { Uint32 c = 0; const char *const *e = SDL_Vulkan_GetInstanceExtensions(&c); *n = c; return e; }
int vkw_create_surface(VkInstance inst, VkSurfaceKHR *out) {
    if (!SDL_Vulkan_CreateSurface(g_w.win, inst, NULL, out)) { VKRED("SDL_Vulkan_CreateSurface failed: %s", SDL_GetError()); return -1; }
    return 0;
}
void vkw_pixel_size(int *w, int *h) { *w = atomic_load(&g_w.w); *h = atomic_load(&g_w.h); }
uint32_t vkw_resize_serial(void) { return atomic_load(&g_w.resize_serial); }
int vkw_minimized(void) { return atomic_load(&g_w.minimized); }
int vkw_display_hz(void) { return atomic_load(&g_w.hz); }
static void post(int code, intptr_t a, intptr_t b) {
    if (!atomic_load(&g_w.open)) return;
    SDL_Event e; memset(&e, 0, sizeof e); e.type = g_w.ev_req; e.user.code = code; e.user.data1 = (void *)a; e.user.data2 = (void *)b; SDL_PushEvent(&e);
}
void vkw_request_size(int w, int h) { post(1, w, h); }
void vkw_request_fullscreen(int on) { post(2, on, 0); }
void vkw_set_close_handler(void (*fn)(void)) { g_w.on_close = fn; }
void vkw_set_title(const char *title) { snprintf(g_w.title, sizeof g_w.title, "%s", title); post(3, 0, 0); }
