/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* glibc 2.38 headers turn strtol/sscanf/fscanf into __isoc23_* when _GNU_SOURCE is on, and those only exist in
   glibc 2.38+, so the binary wouldn't start on Ubuntu 22.04 or Debian 12. defining them here means the binary
   brings its own. this file must not define _GNU_SOURCE: inside it the names are the old ones */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

long __isoc23_strtol(const char* s, char** e, int b){ return strtol(s,e,b); }
unsigned long __isoc23_strtoul(const char* s, char** e, int b){ return strtoul(s,e,b); }
long long __isoc23_strtoll(const char* s, char** e, int b){ return strtoll(s,e,b); }
unsigned long long __isoc23_strtoull(const char* s, char** e, int b){ return strtoull(s,e,b); }
int __isoc23_sscanf(const char* s, const char* f, ...){ va_list a; va_start(a,f); int r=vsscanf(s,f,a); va_end(a); return r; }
int __isoc23_fscanf(FILE* fp, const char* f, ...){ va_list a; va_start(a,f); int r=vfscanf(fp,f,a); va_end(a); return r; }
int __isoc23_scanf(const char* f, ...){ va_list a; va_start(a,f); int r=vscanf(f,a); va_end(a); return r; }
int __isoc23_vsscanf(const char* s, const char* f, va_list a){ return vsscanf(s,f,a); }
int __isoc23_vfscanf(FILE* fp, const char* f, va_list a){ return vfscanf(fp,f,a); }

/* fmod and fmodf got a new symbol version in 2.38 too. fmodl did not, and long double holds any double or float
   exactly, so the remainder is the same */
long double fmodl(long double, long double);
double fmod(double x, double y){ return (double)fmodl(x,y); }
float fmodf(float x, float y){ return (float)fmodl(x,y); }
