/* Our CPU struct (runtime/cpu.h) as the ported NVN layer sees it: same layout up to the vector registers (checked in nvn_hook.c). */
#pragma once
#include <stdint.h>
typedef struct NxwGuestCpu { uint64_t x[31]; uint64_t sp; uint64_t pc; uint32_t n, z, cf, vf; uint8_t q[32][16]; } NxwGuestCpu;
typedef void (*NxwGuestFn)(NxwGuestCpu *);
