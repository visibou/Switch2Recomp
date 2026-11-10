// Write traps on guest pages Eden caches (write_traps.cpp).
#pragma once
#include <cstdint>

void WriteTrapsInit(void* base, std::uint64_t size);
// Eden started (true) or stopped (false) caching [addr, addr+size) (guest addresses)
void WriteTrapsSetCached(std::uint64_t addr, std::uint64_t size, bool cached);
// Hands every page range written since the last call to invalidate(), then traps the ones still cached again
void WriteTrapsCollect(void (*invalidate)(std::uint64_t addr, std::uint64_t size, void* user), void* user);
// totals since start: write faults taken, pages made read-only, collections that overflowed the dirty list
void WriteTrapsStats(std::uint64_t* faults, std::uint64_t* protected_pages, std::uint64_t* overflows);
