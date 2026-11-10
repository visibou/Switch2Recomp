// DM_CTL's renderer commands (hle_hid.c ctl_main): "shot <path>", "texdump <path>", "passdump <n>". nxnative's Metal renderer defines them (nvn_mtl.m
// dm_gfx_shot_at and the dumps); here the backend chosen at run time fills dm_gfx_ctl when it starts (ctl_standins.c says so when none did).
#pragma once

typedef struct {
    void (*shot_at)(const char *path);         // the next presented frame as a PNG at path
    void (*texdump_next)(const char *path);    // the images the next frame draws into or samples (the backend picks the folder: beside the log)
    void (*passdump_next)(long n);             // the passes of the next n frames, to the log
} dm_gfx_ctl_ops;
extern dm_gfx_ctl_ops dm_gfx_ctl;

void dm_gfx_shot_at(const char *path);
void dm_gfx_texdump_next(const char *path);
void dm_gfx_passdump_next(long n);
