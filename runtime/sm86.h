/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* Switch 2 (Ampere) SASS -> Maxwell SASS for shader.c */
#ifndef SWR_SM86_H
#define SWR_SM86_H
#include "cpu.h"
typedef struct { u32 insts, untranslated, first_bad, bad_op; int maxreg, reg_clash; int tex_bank; } Sm86Info;   /* tex_bank: the cbuf bank its bound texture handles live in, -1 none */
/* has_sph: graphics stages start with the 0x50 byte header. returns a malloc'd Maxwell program (header + code)
   for shader_to_glsl, its size in out_len, and the Ampere program size in alen */
extern unsigned short g_sm86_untr[256]; extern int g_sm86_nuntr;
u8* sm86_lower(const u8* code, u32 max_bytes, int has_sph, u32* out_len, u32* alen, Sm86Info* inf);
#endif
