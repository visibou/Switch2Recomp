# Changes to the Eglantine sources in external/eglantine-130

Applied by hand to the copy in `external\eglantine-130`. Re-apply them after copying a new
Eglantine tree.

1. `src/core/device_memory_manager.h`: `physical_max_bits` 35 -> 37. Eden's "physical" address is the runtime's
   guest address (one 128 GiB reservation). The runtime backs GPU sparse regions at 48 GB and up, and 35 bits
   (32 GiB) put them past the end of `compressed_device_addr`, which aborted. The table is a SparseLargeVector,
   committed only where it is used.
