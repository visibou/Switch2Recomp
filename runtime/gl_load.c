/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* GL function pointer loading */
#define SWR_GL_NOMACROS
#include "glfn.h"
#include <stddef.h>
#define X(T,n) T p_##n;
SWR_GL_FUNCS(X)
#undef X
PFNGLCLIPCONTROLPROC_ p_glClipControl;
PFNGLPROGRAMBINARYPROC p_glProgramBinary; PFNGLGETPROGRAMBINARYPROC p_glGetProgramBinary; PFNGLPROGRAMPARAMETERIPROC p_glProgramParameteri;
int swr_gl_load(void* (*gp)(const char*)){
  int ok=1;
#if defined(_WIN32)
  HMODULE gl=LoadLibraryA("opengl32.dll");
#define GETP(n) (gp(n)? gp(n) : (void*)GetProcAddress(gl,n))
#else
#define GETP(n) gp(n)
#endif
#define X(T,n) p_##n=(T)GETP(#n); if(!p_##n) ok=0;
  SWR_GL_FUNCS(X)
#undef X
  p_glClipControl=(PFNGLCLIPCONTROLPROC_)GETP("glClipControl");
  p_glProgramBinary=(PFNGLPROGRAMBINARYPROC)GETP("glProgramBinary"); p_glGetProgramBinary=(PFNGLGETPROGRAMBINARYPROC)GETP("glGetProgramBinary"); p_glProgramParameteri=(PFNGLPROGRAMPARAMETERIPROC)GETP("glProgramParameteri");
  return ok;
}
