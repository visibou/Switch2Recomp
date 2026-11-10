/* SwitchRecomp by MZ Labs, Copyright 2026 MZ Labs. AGPL-3.0, see LICENSE.txt and NOTICE.txt. */
#ifndef SWR_TOOL_MUSIC_H
#define SWR_TOOL_MUSIC_H
int music_load(const char* path);
void music_config(const char* path,int enabled,float volume);
const char* music_error(void);
void music_close(void);
#endif
