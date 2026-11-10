/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* Switch 2 shaders through Eglantine's Ampere -> Maxwell lowering (runtime/eglantine, GPL-3.0-or-later, Eden
   Emulator Project), the one that draws MKW correctly there. Plain C interface for sm86.c, which
   packs the result into our Maxwell layout and falls back to its own lowering for programs this one rejects.
   Built on its own with -std=gnu++20 (std::span) */
#include <cstring>
#include <cstdlib>
#include "eglantine/program_capture.h"

extern "C" int egl_lower_prog(const unsigned char* code, unsigned max_insts, unsigned long long** out, unsigned* nout,
                              int* tex_bank, unsigned* consumed, const char** err, unsigned* err_index){
  using namespace Shader::Ampere;
  std::vector<Instruction> native;
  ProgramCaptureState capture;
  if(max_insts>ProgramCaptureState::MaximumInstructions) max_insts=ProgramCaptureState::MaximumInstructions;
  /* NVN2 leaves 0x30 zero bytes after the 0x50 byte header: the code starts after them. Kept as NOPs so instruction
     indices (and the Ampere program size) stay those of the program */
  unsigned pad=0; while(pad<3 && pad<max_insts){ static const unsigned char z[16]={0}; if(std::memcmp(code+(size_t)pad*16,z,16)) break; pad++; }
  for(unsigned i=pad;i<max_insts;i++){
    Instruction in; std::memcpy(in.words.data(),code+(size_t)i*16,16);
    native.push_back(in);
    auto st=capture.Observe(in,i);
    if(st==ProgramCaptureState::Result::InvalidTarget){ *err="branch outside the program"; *err_index=i; return 0; }
    if(st==ProgramCaptureState::Result::Complete) break; }
  *consumed=pad+(unsigned)native.size();
  LoweredProgram lp=LowerToMaxwell(native);
  if(!lp){ static thread_local char msg[160]; std::size_t n=lp.error.size()<159? lp.error.size() : 159; std::memcpy(msg,lp.error.data(),n); msg[n]=0;
    *err=msg; *err_index=pad+lp.error_offset/16; return 0; }
  *nout=(unsigned)lp.instructions.size();
  *out=(unsigned long long*)std::malloc(sizeof(unsigned long long)*(lp.instructions.size()+1));
  if(!*out){ *err="out of memory"; *err_index=0; return 0; }
  std::memcpy(*out,lp.instructions.data(),sizeof(unsigned long long)*lp.instructions.size());
  *tex_bank= lp.required_texture_buffer==~0U? -1 : (int)lp.required_texture_buffer;
  return 1; }
