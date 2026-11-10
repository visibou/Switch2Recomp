/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* SwitchRecomp tool: the recompiler's front end and the place to play the games, drawn like the console's HOME
   menu but made for the PC: mouse first, keyboard and controller too.
   - recompiles a dump (exefs/ + romfs/) into games\
   - puts the latest runtime (bin\runtime_win64_win32.exe) into every game already made
   - edits each game's options (the .ini) and starts it
   build: see build_tool.sh */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shlobj.h>
#include <shellapi.h>
#include <commdlg.h>
#define EXE_SUFFIX ".exe"
#define CODE_SUFFIX "_code.dll"
#define RUNTIME_FILE "bin\\runtime_win64_win32.exe"
#else
#include "posix_compat.h"   /* linux: the same calls on posix */
#define EXE_SUFFIX ""
#define CODE_SUFFIX "_code.so"
#define RUNTIME_FILE "bin/runtime_linux_x11"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <SDL.h>
#include <SDL_image.h>
#ifndef _WIN32
#define IMG_Load(p) IMG_Load(px(p))
#define IMG_SavePNG(s, p) IMG_SavePNG(s, px(p))
#endif
#include "gfx.h"
#include "music.h"

#define SCREEN_W 1280
#define SCREEN_H 720
#define MAX_GAMES 256

/* ------------------------------------------------------------------ state */

typedef struct {
    Color bg, text, text_dim, accent, accent_glow, line, slot, slot_edge, button, icon, dim;
} Theme;

typedef struct {
    char name[128];     /* file name: games\<name>.exe + <name>_code.dll */
    char title[128];
    SDL_Texture *icon;
    int up_to_date;     /* exe is the current runtime */
    int tcc;            /* code built with TCC (slow) */
    char src[MAX_PATH]; /* dump it was made from, if we know */
    char dir[MAX_PATH]; /* its folder, with a trailing backslash */
    int custom_icon;    /* the player picked the picture */
} Game;

typedef enum { SCREEN_HOME, SCREEN_LIST, SCREEN_PROGRESS } ScreenId;

typedef struct {
    int left, right, up, down, a, b, opts;    /* edge or key repeat */
    /* mouse, in 1280x720 coordinates */
    float mx, my;
    int moved, click, rclick, back, wheel;
} Input;

static struct {
    Theme theme;
    int dark, clock_24h, gcc, native;
    Game games[MAX_GAMES];
    int ngames;
    double now;
    ScreenId screen;
    int quit;
    char here[MAX_PATH];        /* folder of this exe, with a trailing backslash */
    char version[32];           /* runtime version in bin\ */
    char games_dir[MAX_PATH];   /* where games are made and looked for, trailing backslash */
} app;

static void theme_apply(int dark)
{
    app.dark = dark;
    if (!dark)
        app.theme = (Theme){ rgb(235, 235, 235), rgb(45, 45, 45), rgb(130, 130, 130), rgb(8, 175, 200), rgb(60, 228, 238),
            rgb(45, 45, 45), rgb(238, 238, 238), rgb(246, 246, 246), rgb(255, 255, 255), rgb(110, 120, 120), rgb(205, 205, 205) };
    else
        app.theme = (Theme){ rgb(45, 45, 45), rgb(255, 255, 255), rgb(160, 160, 160), rgb(0, 200, 225), rgb(90, 240, 250),
            rgb(255, 255, 255), rgb(52, 52, 52), rgb(62, 62, 62), rgb(70, 70, 70), rgb(225, 225, 225), rgb(80, 80, 80) };
}

static Color pulse_color(void)
{
    float k = 0.5f + 0.5f * sinf((float)app.now * 3.6f);
    return color_mix(app.theme.accent, app.theme.accent_glow, k * 0.55f);
}

static float ease_out(float t)
{
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    float u = 1 - t;
    return 1 - u * u * u;
}

static int inside(float x, float y, float rx, float ry, float rw, float rh)
{
    return x >= rx && x < rx + rw && y >= ry && y < ry + rh;
}

/* ------------------------------------------------------------------ files */

#ifdef _WIN32
static int file_exists(const char *p) { return GetFileAttributesA(p) != INVALID_FILE_ATTRIBUTES; }
#else
static int file_exists(const char *p) { return access(px(p), F_OK) == 0; }
#endif

static void path_join(char *out, size_t n, const char *a, const char *b) { snprintf(out, n, "%s%s", a, b); }

/* key=value .ini, comments with ';' or '#' */
static int ini_get(const char *file, const char *key, char *val, size_t n)
{
    FILE *f = fopen(file, "r");
    val[0] = 0;
    if (!f) return 0;
    char line[1024];
    size_t kl = strlen(key);
    int found = 0;
    while (fgets(line, sizeof line, f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (!strncmp(p, key, kl)) {
            char *q = p + kl;
            while (*q == ' ' || *q == '\t') q++;
            if (*q != '=') continue;
            q++;
            while (*q == ' ' || *q == '\t') q++;
            char *e = q + strcspn(q, "\r\n;");
            while (e > q && (e[-1] == ' ' || e[-1] == '\t')) e--;
            snprintf(val, n, "%.*s", (int)(e - q), q);
            found = 1;
        }
    }
    fclose(f);
    return found;
}

static void ini_set(const char *file, const char *key, const char *val)
{
    FILE *f = fopen(file, "r");
    char *buf = NULL;
    size_t len = 0, cap = 0;
    int done = 0;
    char line[1024];
    size_t kl = strlen(key);
    if (f) {
        while (fgets(line, sizeof line, f)) {
            char *p = line;
            while (*p == ' ' || *p == '\t') p++;
            char out[1100];
            const char *w = line;
            if (!strncmp(p, key, kl)) {
                char *q = p + kl;
                while (*q == ' ' || *q == '\t') q++;
                if (*q == '=') {
                    snprintf(out, sizeof out, "%s=%s\n", key, val);
                    w = out;
                    done = 1;
                }
            }
            size_t l = strlen(w);
            if (len + l + 1 > cap) { cap = (len + l + 1) * 2; buf = realloc(buf, cap); }
            memcpy(buf + len, w, l);
            len += l;
        }
        fclose(f);
    }
    if (!done) {
        if (len && buf[len - 1] != '\n') {
            if (len + 2 > cap) { cap = len + 64; buf = realloc(buf, cap); }
            buf[len++] = '\n';
        }
        snprintf(line, sizeof line, "%s=%s\n", key, val);
        size_t l = strlen(line);
        if (len + l + 1 > cap) { cap = len + l + 64; buf = realloc(buf, cap); }
        memcpy(buf + len, line, l);
        len += l;
    }
    f = fopen(file, "w");
    if (f) {
        fwrite(buf, 1, len, f);
        fclose(f);
    }
    free(buf);
}

static void tool_ini(char *out, size_t n) { path_join(out, n, app.here, "SwitchRecomp.ini"); }
static void runtime_path(char *out, size_t n) { path_join(out, n, app.here, RUNTIME_FILE); }
static void game_file(char *out, size_t n, const Game *g, const char *suffix)
{
    snprintf(out, n, "%s%s%s", g->dir, g->name, suffix);
}

static int same_file(const char *a, const char *b)
{
    FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
    int same = fa && fb;
    if (same) {
        fseek(fa, 0, SEEK_END);
        fseek(fb, 0, SEEK_END);
        same = ftell(fa) == ftell(fb);
        fseek(fa, 0, SEEK_SET);
        fseek(fb, 0, SEEK_SET);
        static char x[65536], y[65536];
        while (same) {
            size_t n = fread(x, 1, sizeof x, fa), m = fread(y, 1, sizeof y, fb);
            if (n != m || memcmp(x, y, n)) same = 0;
            if (!n) break;
        }
    }
    if (fa) fclose(fa);
    if (fb) fclose(fb);
    return same;
}

/* the English title in control.nacp, else the first one there is */
static void nacp_title(const char *path, char *out, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return;
    static char nacp[0x3000];
    size_t got = fread(nacp, 1, sizeof nacp, f);
    fclose(f);
    for (int i = 0; i < 16 && (size_t)(i + 1) * 0x300 <= got; i++) {
        const char *t = nacp + i * 0x300;
        if (t[0]) {
            snprintf(out, n, "%.*s", 0x200, t);
            return;
        }
    }
}

/* icon_*.dat in the control data are JPEGs; the English one first */
static int find_icon_in(const char *dir, int depth, char *out, size_t n)
{
    static const char *const names[] = { "icon_AmericanEnglish.dat", "icon.jpg", "icon.png", "icon_BritishEnglish.dat", "icon_Japanese.dat" };
    char p[MAX_PATH];
    for (unsigned i = 0; i < sizeof names / sizeof names[0]; i++) {
        snprintf(p, sizeof p, "%s\\%s", dir, names[i]);
        if (file_exists(p)) { snprintf(out, n, "%s", p); return 1; }
    }
    snprintf(p, sizeof p, "%s\\icon_*.dat", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(p, &fd);
    if (h != INVALID_HANDLE_VALUE) {
        snprintf(out, n, "%s\\%s", dir, fd.cFileName);
        FindClose(h);
        return 1;
    }
    if (depth <= 0) return 0;
    /* control/ and similar folders, not the big romfs */
    snprintf(p, sizeof p, "%s\\*", dir);
    h = FindFirstFileA(p, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    int found = 0;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || fd.cFileName[0] == '.') continue;
        if (!_stricmp(fd.cFileName, "romfs") || !_stricmp(fd.cFileName, "exefs") || !_stricmp(fd.cFileName, "save")) continue;
        char sub[MAX_PATH];
        snprintf(sub, sizeof sub, "%s\\%s", dir, fd.cFileName);
        found = find_icon_in(sub, depth - 1, out, n);
    } while (!found && FindNextFileA(h, &fd));
    FindClose(h);
    return found;
}

/* the game has no icon in <name>_data, so take one from the data folder or the dump and keep it as
   <name>_data\icon.jpg */
static void extract_icon(Game *g)
{
    char have[MAX_PATH], p[MAX_PATH], dst[MAX_PATH];
    static const char *const known[] = { "_data\\icon.jpg", "_data\\icon.png" };
    for (int i = 0; i < 2; i++) {
        game_file(p, sizeof p, g, known[i]);
        if (file_exists(p)) return;
    }
    game_file(p, sizeof p, g, "_data");
    int ok = find_icon_in(p, 2, have, sizeof have);
    if (!ok && g->src[0]) ok = find_icon_in(g->src, 2, have, sizeof have);
    if (!ok) return;
    game_file(dst, sizeof dst, g, "_data\\icon.jpg");
    if (strlen(have) > 4 && !_stricmp(have + strlen(have) - 4, ".png")) game_file(dst, sizeof dst, g, "_data\\icon.png");
    CopyFileA(have, dst, TRUE);
}

static void load_game(Game *g, SDL_Renderer *r)
{
    char p[MAX_PATH], rt[MAX_PATH];
    snprintf(g->title, sizeof g->title, "%s", g->name);
    game_file(p, sizeof p, g, "_data\\control.nacp");
    nacp_title(p, g->title, sizeof g->title);
    game_file(p, sizeof p, g, ".src");
    g->src[0] = 0;
    FILE *sf = fopen(p, "r");
    if (sf) {
        if (fgets(g->src, sizeof g->src, sf)) g->src[strcspn(g->src, "\r\n")] = 0;
        fclose(sf);
    }
    extract_icon(g);
    game_file(p, sizeof p, g, "_data\\icon_original.jpg");
    g->custom_icon = file_exists(p);
    game_file(p, sizeof p, g, "_data\\icon_original.png");
    g->custom_icon |= file_exists(p);
    game_file(p, sizeof p, g, "_data\\icon_custom");
    g->custom_icon |= file_exists(p);
    static const char *const icons[] = { "_data\\icon.jpg", "_data\\icon.png", "_data\\icon_AmericanEnglish.dat", ".jpg", ".png" };
    g->icon = NULL;
    for (unsigned i = 0; i < sizeof icons / sizeof icons[0] && !g->icon; i++) {
        game_file(p, sizeof p, g, icons[i]);
        if (!file_exists(p)) continue;
        SDL_Surface *s = IMG_Load(p);
        if (s) {
            g->icon = SDL_CreateTextureFromSurface(r, s);
            SDL_FreeSurface(s);
        }
    }
    game_file(p, sizeof p, g, EXE_SUFFIX);
    runtime_path(rt, sizeof rt);
    g->up_to_date = same_file(p, rt);
    /* the runtime says it in the first lines of the game's log */
    g->tcc = 0;
    game_file(p, sizeof p, g, ".log");
    FILE *f = fopen(p, "r");
    if (f) {
        char line[512];
        for (int i = 0; i < 4 && fgets(line, sizeof line, f); i++)
            if (strstr(line, "compiled with TCC")) g->tcc = 1;
        fclose(f);
    }
}

static void unload_games(void)
{
    for (int i = 0; i < app.ngames; i++)
        if (app.games[i].icon) SDL_DestroyTexture(app.games[i].icon);
    app.ngames = 0;
}

static int game_cmp(const void *a, const void *b) { return _stricmp(((const Game *)a)->title, ((const Game *)b)->title); }

/* games are <name>.exe + <name>_code.dll in the folder or any subfolder (one folder per game is fine) */
static void scan_dir(const char *dir, int depth, SDL_Renderer *r)
{
    char pat[MAX_PATH];
    snprintf(pat, sizeof pat, "%s*", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (fd.cFileName[0] == '.') continue;
        size_t l = strlen(fd.cFileName);
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            /* not the games' own data and build folders */
            if (depth <= 0 || (l > 5 && (!_stricmp(fd.cFileName + l - 5, "_data") || (l > 6 && !_stricmp(fd.cFileName + l - 6, "_build")))) ||
                !_stricmp(fd.cFileName, "shader_cache") || !_stricmp(fd.cFileName, "captures")) continue;
            char sub[MAX_PATH];
            snprintf(sub, sizeof sub, "%s%s\\", dir, fd.cFileName);
            scan_dir(sub, depth - 1, r);
            continue;
        }
        size_t cl = strlen(CODE_SUFFIX);
        if (l <= cl || _stricmp(fd.cFileName + l - cl, CODE_SUFFIX) || app.ngames >= MAX_GAMES) continue;
        Game *g = &app.games[app.ngames];
        memset(g, 0, sizeof *g);
        snprintf(g->name, sizeof g->name, "%.*s", (int)(l - cl), fd.cFileName);
        snprintf(g->dir, sizeof g->dir, "%s", dir);
        char exe[MAX_PATH];
        game_file(exe, sizeof exe, g, EXE_SUFFIX);
        if (!file_exists(exe)) continue;
        load_game(g, r);
        app.ngames++;
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

static void load_games(SDL_Renderer *r)
{
    unload_games();
    scan_dir(app.games_dir, 4, r);
    qsort(app.games, app.ngames, sizeof(Game), game_cmp);
}

static void read_version(void)
{
    char p[MAX_PATH];
    path_join(p, sizeof p, app.here, "bin\\runtime_version.txt");
    snprintf(app.version, sizeof app.version, "?");
    FILE *f = fopen(p, "r");
    if (f) {
        if (fgets(app.version, sizeof app.version, f)) app.version[strcspn(app.version, "\r\n")] = 0;
        fclose(f);
    }
}

/* ------------------------------------------------------------------ actions */

static void launch_game(const Game *g)
{
    char exe[MAX_PATH], dir[MAX_PATH];
    game_file(exe, sizeof exe, g, EXE_SUFFIX);
    snprintf(dir, sizeof dir, "%s", g->dir);
#ifdef _WIN32
    ShellExecuteA(NULL, "open", exe, NULL, dir, SW_SHOWNORMAL);
#else
    px_open(exe, dir);
#endif
}

static void open_folder(const Game *g)
{
#ifndef _WIN32
    CreateDirectoryA(app.games_dir, NULL);
    px_open(g ? g->dir : app.games_dir, NULL);
    return;
#else
    if (g) {
        char exe[MAX_PATH], args[MAX_PATH + 16];
        game_file(exe, sizeof exe, g, EXE_SUFFIX);
        snprintf(args, sizeof args, "/select,\"%s\"", exe);
        ShellExecuteA(NULL, "open", "explorer.exe", args, NULL, SW_SHOWNORMAL);
    } else {
        CreateDirectoryA(app.games_dir, NULL);
        ShellExecuteA(NULL, "open", app.games_dir, NULL, NULL, SW_SHOWNORMAL);
    }
#endif
}

/* the runtime is the exe and the game code is the dll next to it, so a new runtime only needs the exe copied */
static int update_game(Game *g)
{
    char exe[MAX_PATH], rt[MAX_PATH];
    game_file(exe, sizeof exe, g, EXE_SUFFIX);
    runtime_path(rt, sizeof rt);
    if (g->up_to_date) return 1;
    if (!CopyFileA(rt, exe, FALSE)) return 0;
    g->up_to_date = 1;
    return 1;
}

static int pick_file(char *out, const char *title)
{
#ifndef _WIN32
    return px_dialog(out, MAX_PATH, title, 0, "--file-filter='Pictures | *.jpg *.jpeg *.png *.bmp *.dat' --file-filter='All files | *'", "'*.jpg *.jpeg *.png *.bmp *.dat|Pictures'") > 0;
#else
    OPENFILENAMEA of;
    memset(&of, 0, sizeof of);
    out[0] = 0;
    of.lStructSize = sizeof of;
    of.lpstrFilter = "Pictures (jpg, png, bmp)\0*.jpg;*.jpeg;*.png;*.bmp;*.dat\0All files\0*.*\0";
    of.lpstrFile = out;
    of.nMaxFile = MAX_PATH;
    of.lpstrTitle = title;
    of.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&of) && out[0];
#endif
}

/* the picked picture becomes <name>_data\icon.png, square 256x256 (cropped to the middle). the game's own icon
   is kept as icon_original.* */
static int set_custom_icon(Game *g, const char *file)
{
    SDL_Surface *s = IMG_Load(file);
    if (!s) return 0;
    SDL_Surface *c = SDL_ConvertSurfaceFormat(s, SDL_PIXELFORMAT_ABGR8888, 0);
    SDL_FreeSurface(s);
    if (!c) return 0;
    int side = c->w < c->h ? c->w : c->h;
    SDL_Rect src = { (c->w - side) / 2, (c->h - side) / 2, side, side };
    SDL_Surface *d = SDL_CreateRGBSurfaceWithFormat(0, 256, 256, 32, SDL_PIXELFORMAT_ABGR8888);
    SDL_SetSurfaceBlendMode(c, SDL_BLENDMODE_NONE);
    SDL_BlitScaled(c, &src, d, NULL);
    SDL_FreeSurface(c);
    char dir[MAX_PATH], p[MAX_PATH], o[MAX_PATH];
    game_file(dir, sizeof dir, g, "_data");
    CreateDirectoryA(dir, NULL);
    static const char *const ext[] = { "jpg", "png" };
    for (int i = 0; i < 2; i++) {
        snprintf(p, sizeof p, "%s\\icon.%s", dir, ext[i]);
        snprintf(o, sizeof o, "%s\\icon_original.%s", dir, ext[i]);
        if (file_exists(p) && !file_exists(o)) MoveFileA(p, o);
        else if (file_exists(p)) DeleteFileA(p);
    }
    snprintf(p, sizeof p, "%s\\icon.png", dir);
    int ok = IMG_SavePNG(d, p) == 0;
    SDL_FreeSurface(d);
    snprintf(o, sizeof o, "%s\\icon_custom", dir);
    FILE *f = fopen(o, "w");
    if (f) fclose(f);
    return ok;
}

/* back to the game's own icon */
static void restore_icon(Game *g)
{
    char dir[MAX_PATH], p[MAX_PATH], o[MAX_PATH];
    game_file(dir, sizeof dir, g, "_data");
    snprintf(p, sizeof p, "%s\\icon.png", dir);
    DeleteFileA(p);
    snprintf(p, sizeof p, "%s\\icon_custom", dir);
    DeleteFileA(p);
    static const char *const ext[] = { "jpg", "png" };
    for (int i = 0; i < 2; i++) {
        snprintf(o, sizeof o, "%s\\icon_original.%s", dir, ext[i]);
        snprintf(p, sizeof p, "%s\\icon.%s", dir, ext[i]);
        if (file_exists(o)) MoveFileA(o, p);
    }
}

static int pick_folder(char *out, const char *title)
{
#ifndef _WIN32
    return px_dialog(out, MAX_PATH, title, 1, "", "") > 0;
#else
    BROWSEINFOA bi;
    memset(&bi, 0, sizeof bi);
    bi.lpszTitle = title;
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    PIDLIST_ABSOLUTE id = SHBrowseForFolderA(&bi);
    if (!id) return 0;
    int ok = SHGetPathFromIDListA(id, out);
    CoTaskMemFree(id);
    return ok && out[0];
#endif
}

/* ------------------------------------------------------------------ recompile job */

static struct {
#ifdef _WIN32
    HANDLE proc, pipe;
#else
    pid_t pid;
    int fd;
#endif
    SDL_Thread *reader;
    SDL_mutex *lock;
    char pending[8192];
    int npending;
    int running, finished, ok, cancelled;
    char lines[10][200];
    int nlines;
    char phase[200];
    float progress, shown;
    char src[MAX_PATH], made[128];
    char header[200];
    double t0;
} J;

static int SDLCALL job_reader(void *arg)
{
    (void)arg;
    char buf[1024];
#ifdef _WIN32
    DWORD n;
    while (ReadFile(J.pipe, buf, sizeof buf, &n, NULL) && n) {
#else
    ssize_t n;
    while ((n = read(J.fd, buf, sizeof buf)) > 0) {
#endif
        SDL_LockMutex(J.lock);
        if (J.npending + (int)n < (int)sizeof J.pending) {
            memcpy(J.pending + J.npending, buf, n);
            J.npending += n;
        }
        SDL_UnlockMutex(J.lock);
    }
    return 0;
}

static int find_python(char *out, size_t n)
{
#ifndef _WIN32
    /* the package's own python first, then the system's */
    char p[MAX_PATH];
    snprintf(p, sizeof p, "%spython/bin/python3", px(app.here));
    if (!access(p, X_OK)) { px_quote(out, n, p); return 1; }
    if (!system("command -v python3 >/dev/null 2>&1")) { snprintf(out, n, "python3"); return 1; }
    return 0;
#else
    char p[MAX_PATH];
    snprintf(p, sizeof p, "%spython\\python.exe", app.here);
    if (file_exists(p)) { snprintf(out, n, "\"%s\"", p); return 1; }
    const char *c[] = { "py.exe", "python.exe", "python3.exe" };
    for (int i = 0; i < 3; i++)
        if (SearchPathA(NULL, c[i], NULL, MAX_PATH, p, NULL)) {
            snprintf(out, n, "%s%s", c[i], i == 0 ? " -3" : "");
            return 1;
        }
    return 0;
#endif
}

static void job_line(const char *s)
{
    if (!s[0]) return;
    if (J.nlines == 10) {
        memmove(J.lines, J.lines + 1, sizeof J.lines[0] * 9);
        J.nlines = 9;
    }
    snprintf(J.lines[J.nlines++], sizeof J.lines[0], "%s", s);
    int a, b;
    const char *d;
    /* what the recompiler prints, turned into a rough progress */
    if (strstr(s, "Analyzing the code")) { snprintf(J.phase, sizeof J.phase, "Analyzing the game code..."); J.progress = 0.05f; }
    else if (strstr(s, "Translating ARM64 to C")) { snprintf(J.phase, sizeof J.phase, "Translating ARM64 to C..."); J.progress = 0.10f; }
    else if (sscanf(s, " C generated: %d/%d", &a, &b) == 2 && b) J.progress = 0.10f + 0.25f * a / b;
    else if (strstr(s, "Copying romfs")) { snprintf(J.phase, sizeof J.phase, "Copying the game data..."); J.progress = 0.36f; }
    else if (strstr(s, "Compiling with")) { snprintf(J.phase, sizeof J.phase, "Compiling to native code..."); J.progress = 0.40f; }
    else if (sscanf(s, " compiled %d/%d", &a, &b) == 2 && b) J.progress = 0.40f + 0.55f * a / b;
    else if (strstr(s, "Linking")) { snprintf(J.phase, sizeof J.phase, "Linking..."); J.progress = 0.96f; }
    else if ((d = strstr(s, "Done: ")) != NULL) {
        const char *base = strrchr(d, '\\');
        const char *b2 = strrchr(d, '/');
        if (!base || (b2 && b2 > base)) base = b2;
        base = base ? base + 1 : d + 6;
        snprintf(J.made, sizeof J.made, "%s", base);
        char *dot = strrchr(J.made, '.');
        if (dot && !_stricmp(dot, ".exe")) *dot = 0;
        J.progress = 1;
    }
}

static int job_start(const char *src)
{
    char py[MAX_PATH], cc[MAX_PATH], cmd[6144];
    memset(&J.pending, 0, sizeof J.pending);
    J.npending = 0;
    J.nlines = 0;
    J.progress = J.shown = 0;
    J.finished = J.ok = J.cancelled = 0;
    J.made[0] = 0;
    snprintf(J.src, sizeof J.src, "%s", src);
    const char *nm = strrchr(src, '\\');
    if (!nm || (strrchr(src, '/') && strrchr(src, '/') > nm)) nm = strrchr(src, '/');
    snprintf(J.header, sizeof J.header, "Recompiling: %s", nm ? nm + 1 : src);
    snprintf(J.phase, sizeof J.phase, "Starting...");
    J.t0 = app.now;
    if (!find_python(py, sizeof py)) {
#ifdef _WIN32
        /* Install.bat gets python (and GCC) into this folder, it opens in its own window */
        char ib[MAX_PATH];
        snprintf(ib, sizeof ib, "%sInstall.bat", app.here);
        if (file_exists(ib) && (INT_PTR)ShellExecuteA(NULL, "open", ib, NULL, app.here, SW_SHOWNORMAL) > 32)
            job_line("Python is not installed yet: Install.bat is downloading it (and GCC) in another window. Recompile again when it says All done.");
        else
            job_line("Python 3 was not found. Double-click Install.bat (needs internet) or install Python from python.org.");
#else
        job_line("Python 3 was not found. Double-click Install.sh (needs internet) or install python3 with your package manager.");
#endif
        J.finished = 1;
        return 0;
    }
#ifndef _WIN32
    /* linux: the GCC that "Install GCC.sh" puts in gcc/, else the system's gcc (or clang). the game comes out as
       <name> + <name>_code.so */
    char lcc_own[MAX_PATH * 2];
    snprintf(cc, sizeof cc, "%sgcc/bin/x86_64-conda-linux-gnu-gcc", px(app.here));
    px_quote(lcc_own, sizeof lcc_own, cc);
    const char *lcc = !access(cc, X_OK) ? lcc_own : !system("command -v gcc >/dev/null 2>&1") ? "gcc" : !system("command -v clang >/dev/null 2>&1") ? "clang" : NULL;
    /* no compiler at all: download GCC first (tool/get_deps.py, into gcc/) and then recompile, all in this job */
    char inst[MAX_PATH * 3] = "";
    if (!lcc) {
        char gp[MAX_PATH], qg[MAX_PATH * 2];
        snprintf(gp, sizeof gp, "%stool/get_deps.py", px(app.here));
        if (!access(gp, R_OK)) {
            px_quote(qg, sizeof qg, gp);
            snprintf(inst, sizeof inst, "%s -u -I %s gcc && ", py, qg);
            lcc = lcc_own;
            job_line("No C compiler found: downloading GCC first (only this time).");
        }
    }
    if (!lcc) {
        job_line("No C compiler found. Double-click Install.sh in the SwitchRecomp folder (needs internet), then Recompile.");
        J.finished = 1;
        return 0;
    }
    char qs[MAX_PATH * 2], qo[MAX_PATH * 2], qh[MAX_PATH * 2];
    char outdir[MAX_PATH];
    snprintf(outdir, sizeof outdir, "%s", px(app.games_dir));
    size_t ol = strlen(outdir);
    if (ol > 1 && outdir[ol - 1] == '/') outdir[ol - 1] = 0;
    px_quote(qs, sizeof qs, px(src));
    px_quote(qo, sizeof qo, outdir);
    char sp[MAX_PATH];
    snprintf(sp, sizeof sp, "%sswitchrecomp.py", px(app.here));
    px_quote(qh, sizeof qh, sp);
    snprintf(cmd, sizeof cmd, "%s%s -u %s %s --target linux --backend x11 --outdir %s --cc %s --jobs %d%s",
        inst, py, qh, qs, qo, lcc, px_nproc(), app.native ? " --native" : "");
    if (inst[0]) {   /* px_spawn_pipe execs the command, so the && chain needs its own shell */
        char ch[sizeof cmd];
        px_quote(ch, sizeof ch, cmd);
        snprintf(cmd, sizeof cmd, "/bin/sh -c %s", ch);
    }
    if (!px_spawn_pipe(cmd, px(app.here), &J.pid, &J.fd)) {
        job_line("Could not start Python.");
        J.finished = 1;
        return 0;
    }
    J.running = 1;
    job_line(strcmp(lcc, "clang") ? "Compiler: GCC" : "Compiler: Clang");
    J.reader = SDL_CreateThread(job_reader, "reader", NULL);
    return 1;
#endif
#ifdef _WIN32
    int gcc = app.gcc;
    snprintf(cc, sizeof cc, "%sgcc\\Library\\bin\\x86_64-w64-mingw32-gcc.exe", app.here);
    if (gcc && !file_exists(cc)) {
        /* GCC is a separate download, building with TCC is better than not building at all */
        job_line("GCC is not installed: using TCC. Double-click Install.bat (needs internet) and use Recompile for a faster game.");
        gcc = 0;
    }
    if (!gcc) snprintf(cc, sizeof cc, "%stcc\\tcc.exe", app.here);
    if (!file_exists(cc)) {
        job_line("The bundled compiler is missing. Unzip the complete package.");
        J.finished = 1;
        return 0;
    }
    /* the compilers' DLLs */
    static char path[16384], np[17000];
    GetEnvironmentVariableA("PATH", path, sizeof path);
    snprintf(np, sizeof np, "%sgcc\\Library\\bin;%stcc;%s", app.here, app.here, path);
    SetEnvironmentVariableA("PATH", np);
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    /* --outdir must not end in a backslash: \" would escape the quote */
    char outdir[MAX_PATH];
    snprintf(outdir, sizeof outdir, "%s", app.games_dir);
    size_t ol = strlen(outdir);
    if (ol > 3 && outdir[ol - 1] == '\\') outdir[ol - 1] = 0;
    snprintf(cmd, sizeof cmd, "%s -u \"%sswitchrecomp.py\" \"%s\" --target win64 --outdir \"%s\" --cc \"%s\" --jobs %d%s",
        py, app.here, src, outdir, cc, (int)si.dwNumberOfProcessors, gcc && app.native ? " --native" : "");
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE rd, wr;
    CreatePipe(&rd, &wr, &sa, 0);
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA st;
    memset(&st, 0, sizeof st);
    st.cb = sizeof st;
    st.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    st.hStdOutput = wr;
    st.hStdError = wr;
    st.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi;
    if (!CreateProcessA(NULL, cmd, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, app.here, &st, &pi)) {
        CloseHandle(rd);
        CloseHandle(wr);
        job_line("Could not start Python.");
        J.finished = 1;
        return 0;
    }
    CloseHandle(wr);
    CloseHandle(pi.hThread);
    J.proc = pi.hProcess;
    J.pipe = rd;
    J.running = 1;
    job_line(gcc ? "Compiler: GCC (takes longer, the game runs faster)" : "Compiler: TCC (fast to build, the game runs slower)");
    J.reader = SDL_CreateThread(job_reader, "reader", NULL);
    return 1;
#endif
}

static void job_poll(void)
{
    if (!J.running) return;
    char local[8192];
    int n;
    SDL_LockMutex(J.lock);
    n = J.npending;
    memcpy(local, J.pending, n);
    J.npending = 0;
    SDL_UnlockMutex(J.lock);
    static char acc[1024];
    static int nacc;
    for (int i = 0; i < n; i++) {
        char ch = local[i];
        if (ch == '\r') continue;
        if (ch == '\n' || nacc == (int)sizeof acc - 1) {
            acc[nacc] = 0;
            job_line(acc);
            nacc = 0;
        } else acc[nacc++] = ch;
    }
#ifdef _WIN32
    if (WaitForSingleObject(J.proc, 0) == WAIT_OBJECT_0) {
#else
    int wst = 0;
    if (waitpid(J.pid, &wst, WNOHANG) == J.pid) {
#endif
        SDL_WaitThread(J.reader, NULL);
        /* whatever came after the last poll */
        SDL_LockMutex(J.lock);
        for (int i = 0; i < J.npending; i++) {
            char ch = J.pending[i];
            if (ch == '\r') continue;
            if (ch == '\n') { acc[nacc] = 0; job_line(acc); nacc = 0; }
            else if (nacc < (int)sizeof acc - 1) acc[nacc++] = ch;
        }
        J.npending = 0;
        SDL_UnlockMutex(J.lock);
        if (nacc) { acc[nacc] = 0; job_line(acc); nacc = 0; }
#ifdef _WIN32
        DWORD code = 1;
        GetExitCodeProcess(J.proc, &code);
        CloseHandle(J.proc);
        CloseHandle(J.pipe);
#else
        int code = WIFEXITED(wst) ? WEXITSTATUS(wst) : 1;
        close(J.fd);
#endif
        J.running = 0;
        J.finished = 1;
        J.ok = code == 0 && !J.cancelled && J.made[0];
        if (J.ok) {
            snprintf(J.phase, sizeof J.phase, "Done. The game is in the games folder.");
            /* remember the dump, for "Recompile" later */
            char p[MAX_PATH];
            snprintf(p, sizeof p, "%s%s.src", app.games_dir, J.made);
            FILE *f = fopen(p, "w");
            if (f) { fprintf(f, "%s\n", J.src); fclose(f); }
        } else if (J.cancelled)
            snprintf(J.phase, sizeof J.phase, "Cancelled.");
        else
            snprintf(J.phase, sizeof J.phase, "It did not work. The last lines below say why.");
    }
}

static void job_cancel(void)
{
    if (!J.running) return;
    J.cancelled = 1;
#ifndef _WIN32
    kill(-J.pid, SIGTERM);   /* its whole process group: python and the compilers */
    return;
#else
    /* python and the compilers it started */
    char cmd[64];
    snprintf(cmd, sizeof cmd, "taskkill /T /F /PID %lu", GetProcessId(J.proc));
    STARTUPINFOA st;
    memset(&st, 0, sizeof st);
    st.cb = sizeof st;
    PROCESS_INFORMATION pi;
    if (CreateProcessA(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &st, &pi)) {
        WaitForSingleObject(pi.hProcess, 5000);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    } else
        TerminateProcess(J.proc, 1);
#endif
}

/* ------------------------------------------------------------------ shared drawing */

static void draw_status(float alpha)
{
    const Theme *th = &app.theme;
    Color text = with_alpha(th->text, alpha);
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    char hm[16];
    float right = 1098;
    if (app.clock_24h)
        snprintf(hm, sizeof hm, "%d:%02d", tm->tm_hour, tm->tm_min);
    else {
        int h = tm->tm_hour % 12 ? tm->tm_hour % 12 : 12;
        snprintf(hm, sizeof hm, "%d:%02d", h, tm->tm_min);
        right -= gfx_text(FONT_CLOCK_SMALL, tm->tm_hour < 12 ? "AM" : "PM", right, 75, ALIGN_RIGHT, text) + 3;
    }
    gfx_text(FONT_CLOCK, hm, right, 75, ALIGN_RIGHT, text);
    gfx_icon(ICON_WIFI, 1136, 63, 27, with_alpha(th->dim, alpha));
    gfx_circle(1136, 72, 3.5f, text);
    int pct = 100, charging = 1;
#ifdef _WIN32
    SYSTEM_POWER_STATUS ps;
    if (GetSystemPowerStatus(&ps) && !(ps.BatteryFlag & 128) && ps.BatteryLifePercent <= 100) {
        pct = ps.BatteryLifePercent;
        charging = ps.ACLineStatus == 1;
    }
#else
    px_battery(&pct, &charging);
#endif
    gfx_battery(1175, 55, pct, charging, text);
}

typedef enum { HINT_A, HINT_B, HINT_PLUS_MINUS } HintButton;
typedef struct { HintButton button; const char *label; } Hint;
/* where the hints were drawn, so a click on one acts like the button */
static struct { float x, w; HintButton b; } g_hint_hit[4];
static int g_nhint_hit;

static void draw_button_glyph(HintButton b, float x, float baseline, Color c, Color bg)
{
    if (b == HINT_PLUS_MINUS) {
        float w = 0;
        gfx_icon(ICON_MINUS, x + 10, baseline - 8, 20, c);
        w += 21;
        w += gfx_text(FONT_HINT, "/", x + w, baseline, ALIGN_LEFT, c) + 1;
        gfx_icon(ICON_PLUS, x + w + 10, baseline - 8, 20, c);
        return;
    }
    gfx_circle(x + 12, baseline - 7, 12, c);
    gfx_text(FONT_BODY_BOLD, b == HINT_A ? "A" : "B", x + 12, baseline - 0.5f, ALIGN_CENTER, bg);
}

static void draw_bottom_bar(const Hint *hints, int n, float alpha, const Input *in)
{
    const Theme *th = &app.theme;
    Color c = with_alpha(th->text, alpha);
    gfx_rect(30, 647, 1220, 1, with_alpha(th->line, alpha));
    gfx_icon(ICON_HANDHELD, 89, 684, 68, with_alpha(app.dark ? th->text : rgb(70, 70, 70), alpha));
    float x = 1218;
    g_nhint_hit = 0;
    for (int i = n - 1; i >= 0; i--) {
        float lw = gfx_text_width(FONT_HINT, hints[i].label);
        float gw = hints[i].button == HINT_PLUS_MINUS ? 62 : 24;
        float start = x - lw - 12 - gw;
        if (in && inside(in->mx, in->my, start - 10, 658, x - start + 20, 50))
            gfx_round_rect(start - 10, 662, x - start + 20, 42, 6, with_alpha(th->dim, alpha * 0.6f));
        draw_button_glyph(hints[i].button, start, 691, c, th->bg);
        gfx_text(FONT_HINT, hints[i].label, x, 691, ALIGN_RIGHT, c);
        if (g_nhint_hit < 4) {
            g_hint_hit[g_nhint_hit].x = start - 10;
            g_hint_hit[g_nhint_hit].w = x - start + 20;
            g_hint_hit[g_nhint_hit].b = hints[i].button;
            g_nhint_hit++;
        }
        x = start - 44;
    }
}

/* a click on a hint of the bottom bar becomes that button */
static void hint_clicks(Input *in)
{
    if (!in->click || in->my < 658 || in->my > 708) return;
    for (int i = 0; i < g_nhint_hit; i++)
        if (in->mx >= g_hint_hit[i].x && in->mx < g_hint_hit[i].x + g_hint_hit[i].w) {
            in->click = 0;
            if (g_hint_hit[i].b == HINT_A) in->a = 1;
            else if (g_hint_hit[i].b == HINT_B) in->b = 1;
            else in->opts = 1;
        }
}

/* ------------------------------------------------------------------ dialog (on top of any screen) */

static struct {
    int open, sel, nbuttons, kind;
    float t;
    char title[200], text[300];
    const char *buttons[2];
} D;
enum { DLG_INFO, DLG_UPDATE_ALL, DLG_CLOSE };

static void dialog_open(int kind, const char *title, const char *text, const char *b0, const char *b1)
{
    D.open = 1;
    D.kind = kind;
    D.t = 0;
    D.sel = b1 ? 1 : 0;
    D.nbuttons = b1 ? 2 : 1;
    snprintf(D.title, sizeof D.title, "%s", title);
    snprintf(D.text, sizeof D.text, "%s", text ? text : "");
    D.buttons[0] = b0;
    D.buttons[1] = b1;
}

static void dialog_rect(float *x, float *y, float *w, float *h)
{
    *w = 680;
    *h = 268;
    *x = (SCREEN_W - *w) / 2;
    *y = (SCREEN_H - *h) / 2 + (1 - ease_out(D.t / 0.18f)) * 20;
}

static void dialog_confirmed(int kind);

static void dialog_update(Input *in, float dt)
{
    D.t += dt;
    float x, y, w, h;
    dialog_rect(&x, &y, &w, &h);
    float by = y + h - 84, bw = w / D.nbuttons;
    if (in->moved && inside(in->mx, in->my, x, by, w, 84)) D.sel = (int)((in->mx - x) / bw);
    if ((in->left || in->right) && D.nbuttons == 2) D.sel ^= 1;
    int pressed = in->a;
    if (in->click) {
        if (inside(in->mx, in->my, x, by, w, 84)) { D.sel = (int)((in->mx - x) / bw); pressed = 1; }
        else if (!inside(in->mx, in->my, x, y, w, h)) { D.open = 0; return; }   /* click outside = cancel */
    }
    if (in->b || in->back) { D.open = 0; return; }
    if (pressed) {
        D.open = 0;
        if (D.nbuttons == 1 || D.sel == 1) dialog_confirmed(D.kind);
    }
}

static void dialog_draw(void)
{
    const Theme *th = &app.theme;
    float t = ease_out(D.t / 0.18f);
    gfx_rect(0, 0, SCREEN_W, SCREEN_H, with_alpha(rgb(0, 0, 0), 0.55f * t));
    float x, y, w, h;
    dialog_rect(&x, &y, &w, &h);
    Color panel = app.dark ? rgb(70, 70, 70) : rgb(255, 255, 255);
    gfx_round_rect(x, y, w, h, 6, with_alpha(panel, t));
    gfx_text_fit(FONT_BODY, D.title, x + w / 2, y + 96, w - 60, ALIGN_CENTER, with_alpha(th->text, t));
    gfx_text_fit(FONT_BODY, D.text, x + w / 2, y + 132, w - 60, ALIGN_CENTER, with_alpha(th->text_dim, t));
    float by = y + h - 84;
    gfx_rect(x, by, w, 1, with_alpha(th->dim, t));
    if (D.nbuttons == 2) gfx_rect(x + w / 2, by, 1, 84, with_alpha(th->dim, t));
    float bw = w / D.nbuttons;
    for (int i = 0; i < D.nbuttons; i++) {
        float bx = x + i * bw;
        if (D.sel == i) gfx_round_frame(bx + 6, by + 6, bw - 12, 72, 4, 4, with_alpha(pulse_color(), t));
        gfx_text(FONT_HINT, D.buttons[i], bx + bw / 2, by + 51, ALIGN_CENTER, with_alpha(th->accent, t));
    }
}

/* ------------------------------------------------------------------ HOME */

#define TILE 256
#define PITCH 270
#define ROW_X 107
#define ROW_Y 194
#define VISIBLE 4
#define BUTTON_Y 545.5f
#define BUTTON_R 43
#define BUTTON_PITCH 108

enum { BTN_UPDATE, BTN_FOLDER, BTN_SETTINGS, BTN_POWER, BTN_COUNT };
static const char *const g_button_names[BTN_COUNT] = { "Update all games", "Games folder", "Settings", "Close" };
static const IconId g_button_icons[BTN_COUNT] = { ICON_REFRESH, ICON_FOLDER, ICON_GEAR, ICON_POWER };

static struct {
    int focus_buttons;
    int sel, first, button;
    float frame_x, scroll, pop, bump, intro;
    int bump_dir;
} H;

static void list_open_game(int game);
static void list_open_settings(void);
static void load_settings(void);
static void recompile(const char *src);

/* tile 0 is "recompile a game", games follow */
static int row_length(void) { return app.ngames + 1; }
static float tile_x(int i) { return ROW_X + (i - H.scroll) * PITCH; }


static void home_enter(void)
{
    if (H.sel >= row_length()) H.sel = row_length() - 1;
    if (H.first > H.sel) H.first = H.sel;
    H.frame_x = tile_x(H.sel);
    H.scroll = (float)H.first;
    H.pop = 1;
    H.bump = -1;
    H.intro = 0;
    app.screen = SCREEN_HOME;
}

static void select_tile(int i)
{
    if (i == H.sel) return;
    H.sel = i;
    if (H.sel < H.first) H.first = H.sel;
    if (H.sel > H.first + VISIBLE - 1) H.first = H.sel - VISIBLE + 1;
    H.pop = 0;
}

static void move_selection(int dir)
{
    int next = H.sel + dir;
    if (next < 0 || next >= row_length()) {
        H.bump = 0;
        H.bump_dir = dir;
        return;
    }
    select_tile(next);
}

static int outdated_games(void)
{
    int n = 0;
    for (int i = 0; i < app.ngames; i++) n += !app.games[i].up_to_date;
    return n;
}

static void activate_tile(void)
{
    if (H.sel == 0) {
        char dir[MAX_PATH];
        if (pick_folder(dir, "Pick the game dump: the folder with exefs and romfs inside")) recompile(dir);
    } else
        launch_game(&app.games[H.sel - 1]);
}

static void activate_button(int b)
{
    char t[200];
    switch (b) {
    case BTN_UPDATE: {
        int n = outdated_games();
        if (!app.ngames)
            dialog_open(DLG_INFO, "There are no games yet.", "Recompile one first: the first tile of the row.", "OK", NULL);
        else if (!n)
            dialog_open(DLG_INFO, "All your games already have the latest runtime.", app.version, "OK", NULL);
        else {
            snprintf(t, sizeof t, "Update %d game%s to runtime %s?", n, n == 1 ? "" : "s", app.version);
            dialog_open(DLG_UPDATE_ALL, t, "Only the .exe changes: saves, options and game code stay.", "Cancel", "Update");
        }
        break;
    }
    case BTN_FOLDER: open_folder(NULL); break;
    case BTN_SETTINGS: list_open_settings(); break;
    case BTN_POWER: dialog_open(DLG_CLOSE, "Close SwitchRecomp?", "Games you already started keep running.", "Cancel", "Close"); break;
    }
}

static void dialog_confirmed(int kind)
{
    if (kind == DLG_CLOSE) app.quit = 1;
    if (kind == DLG_UPDATE_ALL) {
        int ok = 0, fail = 0;
        for (int i = 0; i < app.ngames; i++) {
            if (app.games[i].up_to_date) continue;
            if (update_game(&app.games[i])) ok++;
            else fail++;
        }
        char t[200], s[200];
        snprintf(t, sizeof t, "%d game%s updated to runtime %s.", ok, ok == 1 ? "" : "s", app.version);
        if (fail) snprintf(s, sizeof s, "%d could not be updated: close them and try again.", fail);
        else snprintf(s, sizeof s, "Saves, options and game code were not touched.");
        dialog_open(DLG_INFO, t, s, "OK", NULL);
    }
}

static int button_at(float mx, float my)
{
    float x0 = SCREEN_W / 2.0f - (BTN_COUNT - 1) * BUTTON_PITCH / 2.0f;
    for (int i = 0; i < BTN_COUNT; i++) {
        float dx = mx - (x0 + i * BUTTON_PITCH), dy = my - BUTTON_Y;
        if (dx * dx + dy * dy <= (BUTTON_R + 8) * (BUTTON_R + 8)) return i;
    }
    return -1;
}

static int tile_at(float mx, float my)
{
    if (my < ROW_Y || my >= ROW_Y + TILE) return -1;
    for (int i = 0; i < row_length(); i++) {
        float x = tile_x(i);
        if (mx >= x && mx < x + TILE) return i;
    }
    return -1;
}

static void home_update(Input *in, float dt)
{
    H.intro += dt;
    H.pop += dt;
    if (H.bump >= 0) H.bump = H.bump > 0.5f ? -1 : H.bump + dt;
    float k = 1 - expf(-dt * 22);
    H.frame_x += (tile_x(H.sel) - H.frame_x) * k;
    H.scroll += ((float)H.first - H.scroll) * (1 - expf(-dt * 16));

    /* mouse: hovering selects like touching the screen, a click acts */
    if (in->moved || in->click || in->rclick) {
        int t = tile_at(in->mx, in->my), b = button_at(in->mx, in->my);
        if (t >= 0) {
            float x = tile_x(t);
            int whole = x >= 0 && x + TILE <= SCREEN_W;
            if (whole || in->click || in->rclick) {
                if (H.focus_buttons) { H.focus_buttons = 0; H.pop = 0; }
                select_tile(t);
                if (in->click && whole) activate_tile();
                if (in->rclick && t > 0) list_open_game(t - 1);
            }
        } else if (b >= 0) {
            if (!H.focus_buttons || H.button != b) { H.focus_buttons = 1; H.button = b; H.pop = 0; }
            if (in->click) activate_button(b);
        }
    }
    if (in->wheel) move_selection(in->wheel < 0 ? 1 : -1);

    if (!H.focus_buttons) {
        if (in->left) move_selection(-1);
        if (in->right) move_selection(+1);
        if (in->down) { H.focus_buttons = 1; H.pop = 0; }
        if (in->a) activate_tile();
        if (in->opts && H.sel > 0) list_open_game(H.sel - 1);
    } else {
        if (in->left && H.button > 0) { H.button--; H.pop = 0; }
        if (in->right && H.button < BTN_COUNT - 1) { H.button++; H.pop = 0; }
        if (in->up || in->b) { H.focus_buttons = 0; H.pop = 0; }
        if (in->a) activate_button(H.button);
    }
}

static float pop_scale(void)
{
    if (H.pop > 0.6f) return 1;
    return 1 + 0.045f * expf(-H.pop * 13) * cosf(H.pop * 26);
}

static float bump_offset(void)
{
    if (H.bump < 0) return 0;
    return H.bump_dir * 12 * expf(-H.bump * 11) * sinf(H.bump * 38);
}

static void draw_tile(int i, float x, float y, float alpha)
{
    const Theme *th = &app.theme;
    if (i >= row_length()) {
        gfx_rect(x, y, TILE, TILE, with_alpha(th->slot_edge, alpha));
        gfx_rect(x + 3, y + 3, TILE - 6, TILE - 6, with_alpha(th->slot, alpha));
        return;
    }
    gfx_shadow(x, y, TILE, TILE, 0, 3, with_alpha(rgb(0, 0, 0), alpha * (app.dark ? 0.4f : 0.16f)));
    Color card = app.dark ? rgb(70, 70, 70) : rgb(255, 255, 255);
    if (i == 0) {
        /* "recompile a game": a card with a big plus, like an empty slot you can fill */
        gfx_rect(x, y, TILE, TILE, with_alpha(card, alpha));
        gfx_circle(x + TILE / 2.0f, y + 104, 52, with_alpha(th->accent, alpha));
        gfx_icon(ICON_PLUS, x + TILE / 2.0f, y + 104, 48, with_alpha(rgb(255, 255, 255), alpha));
        gfx_text(FONT_TILE, "Recompile a game", x + TILE / 2.0f, y + 205, ALIGN_CENTER, with_alpha(th->text, alpha));
        return;
    }
    const Game *g = &app.games[i - 1];
    if (g->icon) gfx_texture(g->icon, x, y, TILE, TILE, alpha);
    else {
        gfx_rect(x, y, TILE, TILE, with_alpha(card, alpha));
        gfx_text_fit(FONT_TILE, g->title, x + TILE / 2.0f, y + TILE / 2.0f + 9, TILE - 32, ALIGN_CENTER, with_alpha(th->text, alpha));
    }
    /* a game still on an older runtime gets a small badge */
    if (!g->up_to_date) {
        gfx_circle(x + TILE - 22, y + 22, 13, with_alpha(rgb(255, 255, 255), alpha));
        gfx_circle(x + TILE - 22, y + 22, 10, with_alpha(th->accent, alpha));
    }
}

static void draw_frame(float x, float y, float w, float h, float alpha)
{
    const Theme *th = &app.theme;
    gfx_round_frame(x - 4, y - 4, w + 8, h + 8, 1, 4, with_alpha(app.dark ? th->bg : rgb(255, 255, 255), alpha));
    gfx_round_frame(x - 10, y - 10, w + 20, h + 20, 5, 6, with_alpha(th->accent_glow, alpha * 0.18f));
    gfx_round_frame(x - 9, y - 9, w + 18, h + 18, 4, 5, with_alpha(pulse_color(), alpha));
}

static void home_draw(const Input *in)
{
    const Theme *th = &app.theme;
    gfx_clear(th->bg);
    float alpha = ease_out(H.intro / 0.25f);
    draw_status(alpha);

    float bump = bump_offset();
    int last = (int)H.scroll + VISIBLE + 2;
    if (last < 5) last = 5;
    for (int i = (int)H.scroll - 1; i < last; i++) {
        if (i < 0) continue;
        float t = ease_out((H.intro - 0.04f * (i - (int)H.scroll)) / 0.35f);
        float x = tile_x(i) + (1 - t) * 60 + (i == H.sel ? bump : 0);
        if (x > SCREEN_W || x + TILE < 0) continue;
        draw_tile(i, x, ROW_Y, alpha * t);
    }
    if (!H.focus_buttons) {
        float s = pop_scale();
        float fx = H.frame_x + bump, fw = TILE * s;
        draw_frame(fx - (fw - TILE) / 2, ROW_Y - (fw - TILE) / 2, fw, fw, alpha * ease_out(H.intro / 0.3f));
        char title[200];
        if (H.sel == 0)
            snprintf(title, sizeof title, app.ngames ? "Recompile a game" : "Recompile a game: pick the folder with exefs and romfs");
        else {
            const Game *g = &app.games[H.sel - 1];
            snprintf(title, sizeof title, "%s%s", g->title, !g->up_to_date ? "  -  older runtime, update it" : g->tcc ? "  -  built with TCC (slower)" : "");
        }
        gfx_text_fit(FONT_TITLE, title, H.frame_x + 29, 175, SCREEN_W - H.frame_x - 60, ALIGN_LEFT, with_alpha(th->accent, alpha));
    }

    float x0 = SCREEN_W / 2.0f - (BTN_COUNT - 1) * BUTTON_PITCH / 2.0f;
    for (int i = 0; i < BTN_COUNT; i++) {
        float t = ease_out((H.intro - 0.12f - 0.03f * i) / 0.3f);
        float cx = x0 + i * BUTTON_PITCH, cy = BUTTON_Y + (1 - t) * 24;
        float a = alpha * t;
        int sel = H.focus_buttons && H.button == i;
        float s = sel ? pop_scale() : 1;
        gfx_shadow(cx - BUTTON_R, cy - BUTTON_R + 2, BUTTON_R * 2, BUTTON_R * 2, BUTTON_R, 2, with_alpha(rgb(0, 0, 0), a * 0.10f));
        if (sel) {
            gfx_circle_scaled(cx, cy, BUTTON_R + 9, s, with_alpha(pulse_color(), a));
            gfx_circle_scaled(cx, cy, BUTTON_R + 4, s, with_alpha(th->bg, a));
        }
        gfx_circle_scaled(cx, cy, BUTTON_R, s, with_alpha(th->button, a));
        gfx_icon_scaled(g_button_icons[i], cx, cy, i == BTN_SETTINGS ? 42 : 40, s, with_alpha(th->icon, a));
        if (i == BTN_UPDATE && outdated_games()) {
            /* how many games are waiting for the new runtime */
            char n[8];
            snprintf(n, sizeof n, "%d", outdated_games());
            gfx_circle(cx + 30, cy - 30, 14, with_alpha(th->accent, a));
            gfx_text(FONT_CLOCK_SMALL, n, cx + 30, cy - 23, ALIGN_CENTER, with_alpha(rgb(255, 255, 255), a));
        }
        if (sel) gfx_text(FONT_HINT, g_button_names[i], cx, BUTTON_Y - BUTTON_R - 22, ALIGN_CENTER, with_alpha(th->accent, alpha));
    }

    static const Hint game_hints[] = { { HINT_PLUS_MINUS, "Options" }, { HINT_A, "Start" } };
    static const Hint add_hints[] = { { HINT_A, "Recompile" } };
    static const Hint button_hints[] = { { HINT_A, "OK" } };
    if (H.focus_buttons) draw_bottom_bar(button_hints, 1, alpha, in);
    else if (H.sel == 0) draw_bottom_bar(add_hints, 1, alpha, in);
    else draw_bottom_bar(game_hints, 2, alpha, in);
}

/* ------------------------------------------------------------------ list pages */

#define ROW_H 64
#define LIST_Y 116
#define LIST_X 90
#define LIST_W 1100
#define LIST_ROWS 8         /* rows that fit above the bottom bar */

enum { ACT_NONE, ACT_START, ACT_RECOMPILE, ACT_UPDATE, ACT_FOLDER, ACT_TOOL, ACT_ICON, ACT_ICON_RESET, ACT_GAMES_DIR, ACT_MUSIC };

typedef struct {
    const char *label;
    const char *key;
    const char *const *values;
    const char *const *names;
    int count, def, cur;
    int action;
    char value_text[160];   /* actions show this on the right */
} Row;

static const char *const k_off_on[] = { "0", "1" };
static const char *const n_off_on[] = { "Off", "On" };
static const char *const k_on_off[] = { "1", "0" };
static const char *const n_on_off[] = { "On", "Off" };
static const char *const k_docked[] = { "0", "1" };
static const char *const n_docked[] = { "Handheld (1280x720)", "TV (1920x1080)" };
static const char *const k_res[] = { "0.5", "0.75", "1", "1.5", "2", "3", "4" };
static const char *const n_res[] = { "x0.5", "x0.75", "x1 (native)", "x1.5", "x2", "x3", "x4" };
static const char *const n_fps[] = { "60 FPS", "Unlimited" };
static const char *const k_lang[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14" };
static const char *const n_lang[] = { "Japanese", "English (US)", "French", "German", "Italian", "Spanish", "Chinese", "Korean",
    "Dutch", "Portuguese", "Russian", "Taiwanese", "English (UK)", "French (Canada)", "Spanish (Latin America)" };
static const char *const k_cc[] = { "gcc", "tcc" };
static const char *const n_cc[] = { "GCC (slower to build, faster game)", "TCC (builds in seconds, slower game)" };
static const char *const k_music_vol[] = { "0.1", "0.25", "0.5", "0.75", "1" };
static const char *const n_music_vol[] = { "10%", "25%", "50%", "75%", "100%" };
static const char *const k_memory[] = { "0", "4", "6", "8" };
static const char *const n_memory[] = { "Automatic", "4 GB", "6 GB", "8 GB" };
static const char *const k_theme[] = { "light", "dark" };
static const char *const n_theme[] = { "Basic White", "Basic Black" };
static const char *const n_clock[] = { "12-hour", "24-hour" };

#define ROW(label, key, k, n, def) { label, key, k, n, (int)(sizeof(k) / sizeof(k[0])), def, def, ACT_NONE, "" }
#define ACTION(label, act) { label, NULL, NULL, NULL, 0, 0, 0, act, "" }

static const Row k_game_rows[] = {
    ACTION("Start", ACT_START),
    ACTION("Runtime", ACT_UPDATE),
    ACTION("Recompile the game code", ACT_RECOMPILE),
    ACTION("Open its folder", ACT_FOLDER),
    ACTION("Icon", ACT_ICON),
    ACTION("Use the game's own icon", ACT_ICON_RESET),
    ROW("Console mode", "docked", k_docked, n_docked, 0),
    ROW("Full screen", "fullscreen", k_on_off, n_on_off, 0),
    ROW("Resolution", "resolution", k_res, n_res, 2),
    ROW("Dynamic resolution", "dynamic_resolution", k_off_on, n_off_on, 0),
    ROW("Shadows", "shadows", k_on_off, n_on_off, 0),
    ROW("Frame rate", "fps_limit", k_on_off, n_fps, 0),
    ROW("Frame generation", "lsfg", k_off_on, n_off_on, 0),
    ROW("Language", "language", k_lang, n_lang, 1),
    ROW("Game memory", "memory_gb", k_memory, n_memory, 0),
};

static const Row k_settings_rows[] = {
    ACTION("Games folder", ACT_GAMES_DIR),
    ROW("Compiler", "compiler", k_cc, n_cc, 0),
    ROW("Optimize for this CPU (GCC)", "native", k_off_on, n_off_on, 0),
    ROW("Theme", "theme", k_theme, n_theme, 0),
    ROW("Clock", "clock_24h", k_off_on, n_clock, 0),
    ACTION("Background music file", ACT_MUSIC),
    ROW("Background music", "music_enabled", k_off_on, n_off_on, 0),
    ROW("Music volume", "music_volume", k_music_vol, n_music_vol, 1),
    ACTION("Runtime in this package", ACT_TOOL),
    ACTION("About", ACT_TOOL),
};

static struct {
    Row rows[16];
    int nrows, sel, top, game, closing;  /* game < 0: the tool's settings */
    char header[200], ini[MAX_PATH];
    float t, sel_y, top_y, pop;
} L;

static void list_refresh_values(void)
{
    for (int i = 0; i < L.nrows; i++) {
        Row *r = &L.rows[i];
        if (r->action == ACT_MUSIC) {
            char file[MAX_PATH]; ini_get(L.ini,"music_file",file,sizeof file);
            const char* slash=strrchr(file,'\\'); if(!slash) slash=strrchr(file,'/');
            snprintf(r->value_text,sizeof r->value_text,"%s",file[0]?(slash?slash+1:file):"Choose WAV, OGG or MP3...");
        } else if (r->action == ACT_UPDATE) {
            const Game *g = &app.games[L.game];
            snprintf(r->value_text, sizeof r->value_text, g->up_to_date ? "Latest (%s)" : "Older: update to %s", app.version);
        } else if (r->action == ACT_RECOMPILE) {
            const Game *g = &app.games[L.game];
            snprintf(r->value_text, sizeof r->value_text, "%s%s", g->tcc ? "Built with TCC: " : "", app.gcc ? "with GCC" : "with TCC");
        } else if (r->action == ACT_ICON) {
            snprintf(r->value_text, sizeof r->value_text, "%s", app.games[L.game].custom_icon ? "Your picture: change it" : "Choose a picture...");
        } else if (r->action == ACT_ICON_RESET) {
            snprintf(r->value_text, sizeof r->value_text, "%s", app.games[L.game].custom_icon ? "Restore" : app.games[L.game].icon ? "In use" : "The dump has none");
        } else if (r->action == ACT_GAMES_DIR) {
            snprintf(r->value_text, sizeof r->value_text, "%s", app.games_dir);
        } else if (r->action == ACT_TOOL) {
            if (!strcmp(r->label, "About")) snprintf(r->value_text, sizeof r->value_text, "SwitchRecomp by MZ Labs");
            else snprintf(r->value_text, sizeof r->value_text, "%s", app.version);
        }
    }
}

static void open_page(const Row *rows, int n, int game, const char *header, const char *ini)
{
    memcpy(L.rows, rows, sizeof(Row) * n);
    L.nrows = n;
    L.game = game;
    L.sel = L.top = 0;
    L.t = 0;
    L.pop = 1;
    L.closing = 0;
    L.sel_y = LIST_Y;
    L.top_y = 0;
    snprintf(L.header, sizeof L.header, "%s", header);
    snprintf(L.ini, sizeof L.ini, "%s", ini);
    for (int i = 0; i < n; i++) {
        Row *r = &L.rows[i];
        if (!r->key) continue;
        char v[256];
        ini_get(ini, r->key, v, sizeof v);
        for (int k = 0; k < r->count; k++)
            if (v[0] && !strcmp(v, r->values[k])) r->cur = k;
    }
    list_refresh_values();
    app.screen = SCREEN_LIST;
}

static void list_open_game(int game)
{
    char header[200], ini[MAX_PATH];
    snprintf(header, sizeof header, "Options: %s", app.games[game].title);
    game_file(ini, sizeof ini, &app.games[game], ".ini");
    open_page(k_game_rows, sizeof k_game_rows / sizeof k_game_rows[0], game, header, ini);
}

static void list_open_settings(void)
{
    char ini[MAX_PATH];
    tool_ini(ini, sizeof ini);
    open_page(k_settings_rows, sizeof k_settings_rows / sizeof k_settings_rows[0], -1, "Settings", ini);
}

static void load_settings(void)
{
    char ini[MAX_PATH], v[64];
    tool_ini(ini, sizeof ini);
    ini_get(ini, "compiler", v, sizeof v);
    app.gcc = strcmp(v, "tcc") != 0;
    ini_get(ini, "native", v, sizeof v);
    app.native = !strcmp(v, "1");
    ini_get(ini, "theme", v, sizeof v);
    theme_apply(!strcmp(v, "dark"));
    ini_get(ini, "clock_24h", v, sizeof v);
    app.clock_24h = !strcmp(v, "1");
    char song[MAX_PATH],enabled[32],volume[32];
    ini_get(ini,"music_file",song,sizeof song); ini_get(ini,"music_enabled",enabled,sizeof enabled); ini_get(ini,"music_volume",volume,sizeof volume);
    music_config(song,!strcmp(enabled,"1"),volume[0]?(float)atof(volume):.25f);
    char gd[MAX_PATH];
    if (ini_get(ini, "games_dir", gd, sizeof gd) && gd[0]) {
        size_t l = strlen(gd);
        snprintf(app.games_dir, sizeof app.games_dir, "%s%s", gd, gd[l - 1] == '\\' || gd[l - 1] == '/' ? "" : "\\");
    } else
        snprintf(app.games_dir, sizeof app.games_dir, "%sgames\\", app.here);
#ifndef _WIN32
    px_inplace(app.games_dir);
#endif
}

static void load_games(SDL_Renderer *r);
static void reload_keep_sel(void)
{
    char name[128], dir[MAX_PATH];
    snprintf(name, sizeof name, "%s", app.games[L.game].name);
    snprintf(dir, sizeof dir, "%s", app.games[L.game].dir);
    load_games(gfx_renderer());
    for (int i = 0; i < app.ngames; i++)
        if (!strcmp(app.games[i].name, name) && !strcmp(app.games[i].dir, dir)) L.game = i;
}

static void list_action(Row *r)
{
    Game *g = L.game >= 0 ? &app.games[L.game] : NULL;
    switch (r->action) {
    case ACT_MUSIC: {
        char file[MAX_PATH]={0};
#ifdef _WIN32
        OPENFILENAMEA of={0}; of.lStructSize=sizeof of;
        of.lpstrFilter="Music (WAV, OGG, MP3)\0*.wav;*.ogg;*.mp3\0All files\0*.*\0";
        of.lpstrFile=file; of.nMaxFile=MAX_PATH; of.lpstrTitle="Choose background music"; of.Flags=OFN_FILEMUSTEXIST|OFN_PATHMUSTEXIST|OFN_NOCHANGEDIR;
        if(GetOpenFileNameA(&of)){
#else
        if(px_dialog(file,sizeof file,"Choose background music",0,"--file-filter='Music | *.wav *.ogg *.mp3' --file-filter='All files | *'","'*.wav *.ogg *.mp3|Music'")>0){
#endif
            if(music_load(file)){ ini_set(L.ini,"music_file",file); ini_set(L.ini,"music_enabled","1"); for(int j=0;j<L.nrows;j++) if(L.rows[j].key && !strcmp(L.rows[j].key,"music_enabled")) L.rows[j].cur=1; load_settings(); }
            else dialog_open(DLG_INFO,"Could not play this music file.",music_error(),"OK",NULL);
        }
        break;
    }
    case ACT_START: launch_game(g); break;
    case ACT_FOLDER: open_folder(g); break;
    case ACT_UPDATE:
        if (g->up_to_date) dialog_open(DLG_INFO, "This game already has the latest runtime.", app.version, "OK", NULL);
        else if (update_game(g)) dialog_open(DLG_INFO, "Updated to the latest runtime.", "Saves, options and game code were not touched.", "OK", NULL);
        else dialog_open(DLG_INFO, "Could not replace the .exe.", "Close the game and try again.", "OK", NULL);
        break;
    case ACT_ICON: {
        char file[MAX_PATH];
        if (pick_file(file, "Pick a picture for this game")) {
            if (set_custom_icon(g, file)) { reload_keep_sel(); g = &app.games[L.game]; }
            else dialog_open(DLG_INFO, "That file is not a picture SwitchRecomp can read.", "Use a JPG, PNG or BMP.", "OK", NULL);
        }
        break;
    }
    case ACT_ICON_RESET:
        if (g->custom_icon) { restore_icon(g); reload_keep_sel(); }
        break;
    case ACT_GAMES_DIR: {
        char dir[MAX_PATH];
        if (pick_folder(dir, "Pick the folder where your games are (subfolders count too)")) {
            char ini[MAX_PATH];
            tool_ini(ini, sizeof ini);
            ini_set(ini, "games_dir", dir);
            load_settings();
            load_games(gfx_renderer());
            H.sel = H.first = 0;
        }
        break;
    }
    case ACT_RECOMPILE: {
        char dir[MAX_PATH];
        if (g->src[0] && file_exists(g->src)) recompile(g->src);
        else if (pick_folder(dir, "Pick this game's dump: the folder with exefs and romfs inside")) recompile(dir);
        break;
    }
    }
    list_refresh_values();
}

static void change(int dir)
{
    Row *r = &L.rows[L.sel];
    if (r->action) {
        if (dir > 0) list_action(r);
        L.pop = 0;
        return;
    }
    if (!r->key || r->count < 2) return;
    r->cur = (r->cur + dir + r->count) % r->count;
    ini_set(L.ini, r->key, r->values[r->cur]);
    if (L.game < 0) load_settings();
    list_refresh_values();
    L.pop = 0;
}

static void list_select(int i)
{
    if (i < 0 || i >= L.nrows || i == L.sel) return;
    L.sel = i;
    if (L.sel < L.top) L.top = L.sel;
    if (L.sel > L.top + LIST_ROWS - 1) L.top = L.sel - LIST_ROWS + 1;
    L.pop = 0;
}

static void list_update(Input *in, float dt)
{
    L.t += dt;
    L.pop += dt;
    L.sel_y += (LIST_Y + L.sel * ROW_H - L.sel_y) * (1 - expf(-dt * 22));
    L.top_y += ((float)L.top - L.top_y) * (1 - expf(-dt * 18));
    if (L.closing) {
        if (L.t > 0.16f) home_enter();
        return;
    }
    if (in->moved || in->click || in->rclick) {
        float rel = in->my - LIST_Y + L.top_y * ROW_H;
        if (in->mx >= LIST_X && in->mx < LIST_X + LIST_W && in->my >= LIST_Y && in->my < 640 && rel >= 0) {
            int i = (int)(rel / ROW_H);
            if (i < L.nrows) {
                list_select(i);
                if (in->click) change(+1);
                if (in->rclick) change(-1);
            }
        }
    }
    if (in->wheel) {
        L.top -= in->wheel;
        int maxtop = L.nrows > LIST_ROWS ? L.nrows - LIST_ROWS : 0;
        if (L.top < 0) L.top = 0;
        if (L.top > maxtop) L.top = maxtop;
    }
    if (in->up && L.sel > 0) list_select(L.sel - 1);
    if (in->down && L.sel < L.nrows - 1) list_select(L.sel + 1);
    if (in->left) change(-1);
    if (in->right || in->a) change(+1);
    if (in->b || in->back) {
        L.closing = 1;
        L.t = 0;
    }
}

static void list_draw(const Input *in)
{
    const Theme *th = &app.theme;
    gfx_clear(th->bg);
    float alpha = L.closing ? 1 - ease_out(L.t / 0.16f) : ease_out(L.t / 0.2f);
    float slide = L.closing ? 0 : (1 - ease_out(L.t / 0.25f)) * 30;
    Color text = with_alpha(th->text, alpha);
    gfx_icon(L.game < 0 ? ICON_GEAR : ICON_HANDHELD, 74, 60, L.game < 0 ? 38 : 50, with_alpha(th->icon, alpha));
    gfx_text_fit(FONT_HEADER, L.header, 112, 71, SCREEN_W - 160, ALIGN_LEFT, text);
    gfx_rect(30, 88, 1220, 1, with_alpha(th->line, alpha));

    SDL_Rect clip = { 0, LIST_Y - 8, SCREEN_W, 640 - LIST_Y };
    SDL_RenderSetClipRect(gfx_renderer(), &clip);
    float off = L.top_y * ROW_H;
    for (int i = 0; i < L.nrows; i++) {
        const Row *r = &L.rows[i];
        float y = LIST_Y + i * ROW_H + slide - off;
        if (y < LIST_Y - ROW_H || y > 640) continue;
        if (i > 0) gfx_rect(LIST_X, y, LIST_W, 1, with_alpha(th->dim, alpha));
        gfx_text(FONT_BODY, r->label, LIST_X + 24, y + 41, ALIGN_LEFT, text);
        const char *v = r->action ? r->value_text : r->names[r->cur];
        Color vc = r->action == ACT_TOOL ? th->text_dim : th->accent;
        gfx_text_fit(FONT_BODY, v, LIST_X + LIST_W - 24, y + 41, LIST_W / 2, ALIGN_RIGHT, with_alpha(vc, alpha));
    }
    float s = L.pop > 0.5f ? 0 : 3 * expf(-L.pop * 12) * cosf(L.pop * 24);
    float y = L.sel_y + slide - off;
    gfx_round_frame(LIST_X - 4 - s, y - 2 - s, LIST_W + 8 + 2 * s, ROW_H + 4 + 2 * s, 4, 5, with_alpha(pulse_color(), alpha));
    SDL_RenderSetClipRect(gfx_renderer(), NULL);

    /* more rows below / above */
    if (L.top + LIST_ROWS < L.nrows) gfx_text(FONT_HINT, "...", SCREEN_W / 2.0f, 636, ALIGN_CENTER, with_alpha(th->text_dim, alpha));

    const Row *r = &L.rows[L.sel];
    static const Hint change_hints[] = { { HINT_B, "Back" }, { HINT_A, "Change" } };
    static const Hint ok_hints[] = { { HINT_B, "Back" }, { HINT_A, "OK" } };
    draw_bottom_bar(r->action ? ok_hints : change_hints, 2, alpha, in);
}

/* ------------------------------------------------------------------ progress page */

static void recompile(const char *src)
{
    char ex[MAX_PATH];
    snprintf(ex, sizeof ex, "%s\\exefs", src);
    if (!file_exists(ex) && !(strlen(src) > 4 && !_stricmp(src + strlen(src) - 4, ".nro"))) {
        dialog_open(DLG_INFO, "That folder has no exefs inside.", "Pick the folder that has exefs and romfs (extracted from your own dump).", "OK", NULL);
        return;
    }
    job_start(src);
    app.screen = SCREEN_PROGRESS;
}

static void progress_update(Input *in, float dt)
{
    job_poll();
    J.shown += (J.progress - J.shown) * (1 - expf(-dt * 6));
    if (J.running) {
        if (in->b || in->back) job_cancel();
        return;
    }
    if (in->a || in->b || in->back || in->click) {
        int made = -1;
        load_games(gfx_renderer());
        for (int i = 0; i < app.ngames; i++)
            if (J.ok && !strcmp(app.games[i].name, J.made)) made = i;
        if (made >= 0) {
            H.sel = made + 1;
            H.first = H.sel > VISIBLE - 1 ? H.sel - VISIBLE + 1 : 0;
            H.focus_buttons = 0;
        }
        home_enter();
    }
}

static void progress_draw(const Input *in)
{
    const Theme *th = &app.theme;
    gfx_clear(th->bg);
    Color text = with_alpha(th->text, 1);
    gfx_icon(ICON_REFRESH, 74, 60, 40, th->icon);
    gfx_text_fit(FONT_HEADER, J.header, 112, 71, SCREEN_W - 160, ALIGN_LEFT, text);
    gfx_rect(30, 88, 1220, 1, th->line);

    gfx_text_fit(FONT_TITLE, J.phase, SCREEN_W / 2.0f, 200, 1100, ALIGN_CENTER, J.finished && !J.ok ? th->text : th->accent);
    /* bar */
    float bx = 190, by = 240, bw = 900, bh = 14;
    gfx_round_rect(bx, by, bw, bh, 7, th->dim);
    float f = J.shown < 0 ? 0 : J.shown > 1 ? 1 : J.shown;
    if (f > 0.01f) gfx_round_rect(bx, by, bw * f, bh, 7, J.finished && !J.ok ? th->text_dim : pulse_color());
    char pct[64];
    int secs = (int)(app.now - J.t0);
    snprintf(pct, sizeof pct, "%d%%   %d:%02d", (int)(f * 100 + 0.5f), secs / 60, secs % 60);
    gfx_text(FONT_HINT, pct, SCREEN_W / 2.0f, 290, ALIGN_CENTER, th->text_dim);

    /* what the recompiler says, last lines */
    gfx_round_rect(90, 320, 1100, 300, 6, app.dark ? rgb(58, 58, 58) : rgb(245, 245, 245));
    for (int i = 0; i < J.nlines && i < 9; i++)
        gfx_text_fit(FONT_BODY, J.lines[J.nlines > 9 ? J.nlines - 9 + i : i], 110, 356 + i * 30, 1060, ALIGN_LEFT, th->text_dim);

    static const Hint run_hints[] = { { HINT_B, "Cancel" } };
    static const Hint done_hints[] = { { HINT_A, "OK" } };
    draw_bottom_bar(J.running ? run_hints : done_hints, 1, 1, in);
}

/* ------------------------------------------------------------------ input */

static void read_input(Input *in, float dt, SDL_GameController **pad)
{
    static float hold[4];
    static int held[4];
    int down[4] = { 0 };
    float mx = in->mx, my = in->my;
    memset(in, 0, sizeof *in);
    in->mx = mx;
    in->my = my;
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
        case SDL_QUIT: app.quit = 1; break;
        case SDL_MOUSEMOTION: in->mx = (float)e.motion.x; in->my = (float)e.motion.y; in->moved = 1; break;
        case SDL_MOUSEBUTTONDOWN:
            in->mx = (float)e.button.x;
            in->my = (float)e.button.y;
            if (e.button.button == SDL_BUTTON_LEFT) in->click = 1;
            if (e.button.button == SDL_BUTTON_RIGHT) in->rclick = 1;
            if (e.button.button == SDL_BUTTON_X1) in->back = 1;
            break;
        case SDL_MOUSEWHEEL: in->wheel += e.wheel.y; break;
        case SDL_KEYDOWN: {
            SDL_Keycode k = e.key.keysym.sym;
            int dir = k == SDLK_LEFT ? 0 : k == SDLK_RIGHT ? 1 : k == SDLK_UP ? 2 : k == SDLK_DOWN ? 3 : -1;
            if (dir >= 0) { if (!e.key.repeat) down[dir] = 1; }
            else if (!e.key.repeat) {
                if (k == SDLK_RETURN || k == SDLK_KP_ENTER || k == SDLK_SPACE) in->a = 1;
                if (k == SDLK_ESCAPE || k == SDLK_BACKSPACE) in->b = 1;
                if (k == SDLK_PLUS || k == SDLK_MINUS || k == SDLK_KP_PLUS || k == SDLK_KP_MINUS || k == SDLK_EQUALS) in->opts = 1;
            }
            break;
        }
        case SDL_CONTROLLERDEVICEADDED:
            if (!*pad) *pad = SDL_GameControllerOpen(e.cdevice.which);
            break;
        case SDL_CONTROLLERBUTTONDOWN: {
            int b = e.cbutton.button;
            if (b == SDL_CONTROLLER_BUTTON_A) in->a = 1;
            if (b == SDL_CONTROLLER_BUTTON_B) in->b = 1;
            if (b == SDL_CONTROLLER_BUTTON_START || b == SDL_CONTROLLER_BUTTON_BACK) in->opts = 1;
            if (b == SDL_CONTROLLER_BUTTON_DPAD_LEFT) down[0] = 1;
            if (b == SDL_CONTROLLER_BUTTON_DPAD_RIGHT) down[1] = 1;
            if (b == SDL_CONTROLLER_BUTTON_DPAD_UP) down[2] = 1;
            if (b == SDL_CONTROLLER_BUTTON_DPAD_DOWN) down[3] = 1;
            break;
        }
        }
    }
    /* held directions repeat like the HOME menu */
    const Uint8 *ks = SDL_GetKeyboardState(NULL);
    int now[4] = { ks[SDL_SCANCODE_LEFT], ks[SDL_SCANCODE_RIGHT], ks[SDL_SCANCODE_UP], ks[SDL_SCANCODE_DOWN] };
    if (*pad) {
        const int bt[4] = { SDL_CONTROLLER_BUTTON_DPAD_LEFT, SDL_CONTROLLER_BUTTON_DPAD_RIGHT, SDL_CONTROLLER_BUTTON_DPAD_UP, SDL_CONTROLLER_BUTTON_DPAD_DOWN };
        int ax = SDL_GameControllerGetAxis(*pad, SDL_CONTROLLER_AXIS_LEFTX), ay = SDL_GameControllerGetAxis(*pad, SDL_CONTROLLER_AXIS_LEFTY);
        int st[4] = { ax < -20000, ax > 20000, ay < -20000, ay > 20000 };
        for (int i = 0; i < 4; i++) {
            now[i] |= SDL_GameControllerGetButton(*pad, bt[i]) | st[i];
            if (st[i] && !held[i]) down[i] = 1;
        }
    }
    int *out[4] = { &in->left, &in->right, &in->up, &in->down };
    for (int i = 0; i < 4; i++) {
        if (down[i]) { *out[i] = 1; hold[i] = -0.32f; }
        else if (now[i]) {
            hold[i] += dt;
            if (hold[i] >= 0) { *out[i] = 1; hold[i] -= 0.075f; }
        }
        held[i] = now[i];
    }
}

/* ------------------------------------------------------------------ main */

#include "splash_img.h"
/* the SwitchRECOMP startup screen: logo and copyright notice on black for 2 seconds, then it vanishes (fades out
   while it grows a little). NOTICE.txt (AGPL section 7 terms) says copies and forks keep it */
static void splash_show(SDL_Renderer *ren, double secs)
{
    unsigned char *cov = malloc(SPLASH_W * SPLASH_H);
    Uint32 *px = malloc(SPLASH_W * SPLASH_H * 4);
    SDL_Texture *t = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, SPLASH_W, SPLASH_H);
    if (cov && px && t) {
        splash_unpack(cov);
        for (int i = 0; i < SPLASH_W * SPLASH_H; i++) px[i] = 0x00FFFFFFu | ((Uint32)cov[i] << 24);
        SDL_UpdateTexture(t, NULL, px, SPLASH_W * 4);
        SDL_SetTextureBlendMode(t, SDL_BLENDMODE_BLEND);
        SDL_SetTextureScaleMode(t, SDL_ScaleModeLinear);
        Uint64 t0 = SDL_GetPerformanceCounter();
        double freq = (double)SDL_GetPerformanceFrequency(), fin = 0.25, vstart = secs * 0.7;
        for (;;) {
            SDL_Event ev;
            int quit = 0;
            while (SDL_PollEvent(&ev)) if (ev.type == SDL_QUIT) { SDL_PushEvent(&ev); quit = 1; break; }
            double tt = (double)(SDL_GetPerformanceCounter() - t0) / freq;
            if (quit || tt >= secs) break;
            double al = 1.0, e = 0.0;
            if (tt < fin) al = tt / fin;
            else if (tt > vstart) { e = (tt - vstart) / (secs - vstart); e = e * e * (3.0 - 2.0 * e); al = 1.0 - e; }
            SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
            SDL_RenderClear(ren);
            int w = (int)(SCREEN_W * (1.0 + 0.06 * e)), h = (int)(SCREEN_H * (1.0 + 0.06 * e));
            SDL_Rect d = { (SCREEN_W - w) / 2, (SCREEN_H - h) / 2, w, h };
            SDL_SetTextureAlphaMod(t, (Uint8)(al * 255.0));
            SDL_RenderCopy(ren, t, NULL, &d);
            SDL_RenderPresent(ren);
            SDL_Delay(5);
        }
    }
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
    SDL_RenderClear(ren);
    SDL_RenderPresent(ren);
    if (t) SDL_DestroyTexture(t);
    free(px);
    free(cov);
}

int main(int argc, char **argv)
{
    const char *shot = NULL;
    int shot_screen = 0;
    const char *test_icon = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--shot=", 7)) shot = argv[i] + 7;
        if (!strncmp(argv[i], "--screen=", 9)) shot_screen = atoi(argv[i] + 9);
        if (!strncmp(argv[i], "--test-icon=", 12)) test_icon = argv[i] + 12;   /* checks: picture for game 0 */
    }
    char *base = SDL_GetBasePath();
    snprintf(app.here, sizeof app.here, "%s", base ? base : ".\\");
    SDL_free(base);
#ifdef _WIN32
    SetCurrentDirectoryA(app.here);
    CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    SetProcessDPIAware();
#else
    if (chdir(app.here)) {}
    signal(SIGPIPE, SIG_IGN);
#endif

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER | SDL_INIT_AUDIO) < 0) {
#ifdef _WIN32
        MessageBoxA(NULL, SDL_GetError(), "SwitchRecomp", MB_ICONERROR);
#else
        fprintf(stderr, "SwitchRecomp: %s\n", SDL_GetError());
#endif
        return 1;
    }
    SDL_Window *win = SDL_CreateWindow("SwitchRecomp v190", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_W, SCREEN_H, SDL_WINDOW_RESIZABLE);
    SDL_Renderer *ren = win ? SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC) : NULL;
    if (!ren && win) ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
    char fonts[MAX_PATH];
    path_join(fonts, sizeof fonts, app.here, "tool\\fonts\\");
#ifndef _WIN32
    px_inplace(fonts);
#endif
    if (!ren || !gfx_init(ren, fonts)) {
#ifdef _WIN32
        MessageBoxA(NULL, "Could not start the interface (tool\\fonts missing?). Unzip the complete package.", "SwitchRecomp", MB_ICONERROR);
#else
        fprintf(stderr, "SwitchRecomp: could not start the interface (tool/fonts missing?). Unzip the complete package.\n");
#endif
        return 1;
    }
    SDL_RenderSetLogicalSize(ren, SCREEN_W, SCREEN_H);
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    splash_show(ren, 2.0);
    IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG);
    J.lock = SDL_CreateMutex();

    load_settings();
    CreateDirectoryA(app.games_dir, NULL);
    read_version();
    load_games(ren);
    if (test_icon && app.ngames) {
        if (!strcmp(test_icon, "restore")) restore_icon(&app.games[0]);
        else set_custom_icon(&app.games[0], test_icon);
        load_games(ren);
    }
    home_enter();

    SDL_GameController *pad = NULL;
    Input in;
    memset(&in, 0, sizeof in);
    in.mx = in.my = -1;
    Uint64 last = SDL_GetPerformanceCounter();
    int frames = 0;
    while (!app.quit) {
        Uint64 t = SDL_GetPerformanceCounter();
        float dt = (float)(t - last) / (float)SDL_GetPerformanceFrequency();
        last = t;
        if (dt > 0.1f) dt = 0.1f;
        app.now += dt;
        read_input(&in, dt, &pad);
        if (shot) {
            /* screenshots for checking the look: --shot=file.bmp --screen=0 home, 1 options, 2 settings, 3 dialog, 4 progress */
            static int set;
            if (!set) {
                set = 1;
                if (shot_screen == 1 && app.ngames) list_open_game(0);
                if (shot_screen == 2) list_open_settings();
                if (shot_screen == 3) activate_button(BTN_UPDATE);
                if (shot_screen == 4) {
                    app.screen = SCREEN_PROGRESS;
                    snprintf(J.header, sizeof J.header, "Recompiling: Kirby and the Forgotten Land");
                    J.running = 0;
                    job_line("SwitchRecomp - use your own copy of the game. No games or keys included.");
                    job_line("Translating ARM64 to C...");
                    job_line("  C generated: 40/40 chunks (61s)");
                    job_line("Compiling with x86_64-w64-mingw32-gcc.exe (-O2, 16 at a time)...");
                    job_line("  compiled 25/40 (95s)");
                    J.shown = J.progress;
                    J.t0 = -154;
                }
            }
        }
        if (D.open) {
            Input none = in;
            dialog_update(&in, dt);
            memset(&in.left, 0, sizeof(int) * 7);
            in.moved = 0;
            in.click = in.rclick = in.back = in.wheel = 0;
            (void)none;
        } else
            hint_clicks(&in);
        if (app.screen == SCREEN_HOME) { home_update(&in, dt); home_draw(&in); }
        else if (app.screen == SCREEN_LIST) { list_update(&in, dt); list_draw(&in); }
        else { progress_update(&in, dt); progress_draw(&in); }
        if (D.open) dialog_draw();
        if (shot && ++frames == 90) {
            int w, h;
            SDL_GetRendererOutputSize(ren, &w, &h);
            SDL_Surface *s = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_ARGB8888);
            SDL_RenderReadPixels(ren, NULL, SDL_PIXELFORMAT_ARGB8888, s->pixels, s->pitch);
            SDL_SaveBMP(s, shot);
            SDL_FreeSurface(s);
            break;
        }
        SDL_RenderPresent(ren);
    }
    if (J.running) job_cancel();
    unload_games();
    music_close();
    IMG_Quit();
    gfx_exit();
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
