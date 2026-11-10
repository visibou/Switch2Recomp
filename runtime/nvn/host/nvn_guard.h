/* What MSVC-style __try did in the ported files, for GCC: a read of memory that may not be mapped. nvn_guard_cur holds a __builtin_setjmp buffer while
   the guarded code runs; the vectored handler in nvn_host.c continues there on an access violation. */
#pragma once
extern __thread void **nvn_guard_cur;
#define NVN_GUARDED(fail) void *gb_[5]; if (__builtin_setjmp(gb_)) { nvn_guard_cur = 0; fail; } nvn_guard_cur = gb_;
#define NVN_GUARD_END() (nvn_guard_cur = 0)
