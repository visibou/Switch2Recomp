/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* the few Windows calls the tool uses, on Linux. main.c builds its paths with backslashes and every call here turns
   them into slashes first. file dialogs go through zenity or kdialog, whichever is installed */
#ifndef POSIX_COMPAT_H
#define POSIX_COMPAT_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <dirent.h>
#include <fnmatch.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

extern char **environ;

#define MAX_PATH 4096
typedef unsigned int DWORD;
typedef void *HANDLE;
#define INVALID_HANDLE_VALUE ((HANDLE)0)
#define FILE_ATTRIBUTE_DIRECTORY 0x10
#define _stricmp strcasecmp
#define TRUE 1
#define FALSE 0

/* backslashes to slashes, in one of a few rotating buffers so two paths can be in use at once */
static const char *px(const char *p)
{
    static char b[4][MAX_PATH];
    static int k;
    char *o = b[k++ & 3];
    size_t i = 0;
    for (; p[i] && i < MAX_PATH - 1; i++) o[i] = p[i] == '\\' ? '/' : p[i];
    o[i] = 0;
    return o;
}
static void px_inplace(char *p) { for (; *p; p++) if (*p == '\\') *p = '/'; }

static FILE *px_fopen(const char *p, const char *m) { return fopen(px(p), m); }
#define fopen(p, m) px_fopen(p, m)

typedef struct { DWORD dwFileAttributes; char cFileName[256]; } WIN32_FIND_DATAA;
typedef struct { DIR *d; char dir[MAX_PATH]; char pat[256]; } PxFind;

static int px_find_next(PxFind *f, WIN32_FIND_DATAA *fd)
{
    struct dirent *e;
    while ((e = readdir(f->d))) {
        if (fnmatch(f->pat, e->d_name, FNM_CASEFOLD)) continue;
        char full[MAX_PATH * 2];
        struct stat st;
        snprintf(full, sizeof full, "%s/%s", f->dir, e->d_name);
        fd->dwFileAttributes = !stat(full, &st) && S_ISDIR(st.st_mode) ? FILE_ATTRIBUTE_DIRECTORY : 0;
        snprintf(fd->cFileName, sizeof fd->cFileName, "%s", e->d_name);
        return 1;
    }
    return 0;
}
static HANDLE FindFirstFileA(const char *pattern, WIN32_FIND_DATAA *fd)
{
    const char *p = px(pattern);
    const char *sl = strrchr(p, '/');
    PxFind *f = calloc(1, sizeof *f);
    if (sl) { snprintf(f->dir, sizeof f->dir, "%.*s", (int)(sl - p), p); snprintf(f->pat, sizeof f->pat, "%s", sl + 1); }
    else { snprintf(f->dir, sizeof f->dir, "."); snprintf(f->pat, sizeof f->pat, "%s", p); }
    if (!f->dir[0]) snprintf(f->dir, sizeof f->dir, "/");
    f->d = opendir(f->dir);
    if (!f->d || !px_find_next(f, fd)) { if (f->d) closedir(f->d); free(f); return INVALID_HANDLE_VALUE; }
    return f;
}
static int FindNextFileA(HANDLE h, WIN32_FIND_DATAA *fd) { return px_find_next((PxFind *)h, fd); }
static int FindClose(HANDLE h) { PxFind *f = h; closedir(f->d); free(f); return 1; }

static int CreateDirectoryA(const char *p, void *sa) { (void)sa; return mkdir(px(p), 0755) == 0; }
static int DeleteFileA(const char *p) { return unlink(px(p)) == 0; }
static int MoveFileA(const char *a, const char *b) { char x[MAX_PATH]; snprintf(x, sizeof x, "%s", px(a)); return rename(x, px(b)) == 0; }
/* copies keep the source's permissions, so a copied runtime stays executable */
static int CopyFileA(const char *a, const char *b, int fail_if_exists)
{
    char x[MAX_PATH], y[MAX_PATH];
    snprintf(x, sizeof x, "%s", px(a));
    snprintf(y, sizeof y, "%s", px(b));
    struct stat st;
    if (stat(x, &st)) return 0;
    if (fail_if_exists && !access(y, F_OK)) return 0;
    int in = open(x, O_RDONLY);
    if (in < 0) return 0;
    char tmp[MAX_PATH + 8];
    snprintf(tmp, sizeof tmp, "%s.new", y);   /* a running game keeps its old file, like a replace on Windows */
    int out = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, st.st_mode & 0777);
    if (out < 0) { close(in); return 0; }
    char buf[65536];
    ssize_t n;
    int ok = 1;
    while ((n = read(in, buf, sizeof buf)) > 0) if (write(out, buf, n) != n) { ok = 0; break; }
    close(in);
    close(out);
    if (!ok || rename(tmp, y)) { unlink(tmp); return 0; }
    return 1;
}

static void px_sh(const char *cmd)   /* fire and forget, detached */
{
    pid_t pid;
    char *argv[] = { "sh", "-c", (char *)cmd, NULL };
    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&at, 0);
    if (!posix_spawn(&pid, "/bin/sh", NULL, &at, argv, environ)) { /* reaped by SIGCHLD=SIG_IGN */ }
    posix_spawnattr_destroy(&at);
}
/* single-quoted for sh */
static void px_quote(char *o, size_t n, const char *s)
{
    size_t k = 0;
    if (k < n - 1) o[k++] = '\'';
    for (; *s && k + 5 < n; s++) {
        if (*s == '\'') { memcpy(o + k, "'\\''", 4); k += 4; }
        else o[k++] = *s;
    }
    if (k < n - 1) o[k++] = '\'';
    o[k] = 0;
}
/* a game: run it from its folder. anything else (a folder): the desktop's file manager */
static void px_open(const char *what, const char *dir)
{
    char q[MAX_PATH * 2], d[MAX_PATH * 2], cmd[MAX_PATH * 5];
    struct stat st;
    const char *w = px(what);
    px_quote(q, sizeof q, w);
    if (!stat(w, &st) && S_ISREG(st.st_mode) && (st.st_mode & 0111)) {
        px_quote(d, sizeof d, dir ? px(dir) : ".");
        snprintf(cmd, sizeof cmd, "cd %s && exec %s >/dev/null 2>&1", d, q);
    } else snprintf(cmd, sizeof cmd, "xdg-open %s >/dev/null 2>&1", q);
    px_sh(cmd);
}

/* zenity, else kdialog. 0 if neither is there or the user cancelled */
static int px_dialog(char *out, size_t n, const char *title, int folder, const char *filter_zenity, const char *filter_kde)
{
    char cmd[2048], t[600];
    px_quote(t, sizeof t, title);
    if (!system("command -v zenity >/dev/null 2>&1"))
        snprintf(cmd, sizeof cmd, "zenity --file-selection %s --title=%s %s 2>/dev/null", folder ? "--directory" : "", t, folder ? "" : filter_zenity);
    else if (!system("command -v kdialog >/dev/null 2>&1"))
        snprintf(cmd, sizeof cmd, "kdialog --title %s %s . %s 2>/dev/null", t, folder ? "--getexistingdirectory" : "--getopenfilename", folder ? "" : filter_kde);
    else {
        fprintf(stderr, "SwitchRecomp: install zenity or kdialog to pick files and folders\n");
        return -1;
    }
    FILE *p = popen(cmd, "r");
    if (!p) return 0;
    out[0] = 0;
    if (!fgets(out, (int)n, p)) out[0] = 0;
    pclose(p);
    out[strcspn(out, "\r\n")] = 0;
    return out[0] != 0;
}

/* the recompiler runs as a child with its output on a pipe, in its own process group so cancel stops the compilers
   too */
static int px_spawn_pipe(const char *cmd, const char *dir, pid_t *pid, int *rd)
{
    int fds[2];
    if (pipe(fds)) return 0;
    posix_spawn_file_actions_t fa;
    posix_spawnattr_t at;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_addclose(&fa, fds[1]);
    posix_spawnattr_init(&at);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&at, 0);
    char full[8192];
    char d[MAX_PATH * 2];
    px_quote(d, sizeof d, dir);
    snprintf(full, sizeof full, "cd %s && exec %s", d, cmd);
    char *argv[] = { "sh", "-c", full, NULL };
    int r = posix_spawn(pid, "/bin/sh", &fa, &at, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&at);
    close(fds[1]);
    if (r) { close(fds[0]); return 0; }
    *rd = fds[0];
    return 1;
}

/* battery for the status bar: the first BAT* in /sys, 100% on AC if there is none */
static int px_battery(int *pct, int *charging)
{
    DIR *d = opendir("/sys/class/power_supply");
    if (!d) return 0;
    struct dirent *e;
    int found = 0;
    while (!found && (e = readdir(d))) {
        if (strncmp(e->d_name, "BAT", 3)) continue;
        char p[512], v[64];
        snprintf(p, sizeof p, "/sys/class/power_supply/%s/capacity", e->d_name);
        FILE *f = fopen(p, "r");
        if (!f) continue;
        if (fgets(v, sizeof v, f)) { *pct = atoi(v); found = 1; }
        fclose(f);
        snprintf(p, sizeof p, "/sys/class/power_supply/%s/status", e->d_name);
        f = fopen(p, "r");
        if (f) { if (fgets(v, sizeof v, f)) *charging = strncmp(v, "Discharging", 11) != 0; fclose(f); }
    }
    closedir(d);
    return found;
}

static int px_nproc(void) { long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 4; }
#endif
