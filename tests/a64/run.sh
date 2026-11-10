#!/bin/sh
# a64.py's LSE atomics (LDADD..SWP, CAS, CASP) and LDAPR against a real AArch64: each instruction runs on qemu-aarch64 -cpu max and through
# the C a64.py makes for it, with random registers and memory. Needs qemu-user and gcc-aarch64-linux-gnu.
set -e
cd "$(dirname "$0")"; ROOT=../..; O=$(mktemp -d)
for seed in 1 2 3 4 5; do
  python3 -I gen.py $ROOT $O $seed >/dev/null
  grep -v '^static void r\|^typedef void\|^F R' $O/ref.c | sed '/^#include/d;/^typedef uint64_t/d' > $O/ref_table.c
  sed "s/u64 seed=0x1234567;/u64 seed=0x1234567+$seed;/" main_ref.c > $O/main_ref.c
  sed "s/u64 seed=0x1234567;/u64 seed=0x1234567+$seed;/" main_gen.c > $O/main_gen.c
  aarch64-linux-gnu-gcc -O1 -march=armv8.4-a -static $O/main_ref.c -o $O/ref
  gcc -O2 -w -mcx16 -I $ROOT/runtime $O/main_gen.c $ROOT/runtime/cpu_helpers.c -o $O/gen -lm
  qemu-aarch64 -cpu max $O/ref > $O/ref.txt; $O/gen > $O/gen.txt
  if cmp -s $O/ref.txt $O/gen.txt; then echo "PASS seed $seed ($(wc -l < $O/ref.txt) instructions)"; else echo "FAIL seed $seed"; diff $O/ref.txt $O/gen.txt | head; fi
done
rm -rf $O
O=$(mktemp -d)
python3 -I fp16.py $ROOT $O
aarch64-linux-gnu-gcc -O1 -march=armv8.4-a -static $O/ref.c -o $O/ref
gcc -O2 -w -I $ROOT/runtime $O/gen.c -o $O/gen -lm
qemu-aarch64 -cpu max $O/ref > $O/ref.txt; $O/gen > $O/gen.txt
if cmp -s $O/ref.txt $O/gen.txt; then echo "PASS FP16 FABS/FNEG ($(wc -l < $O/ref.txt) runs)"; else echo "FAIL FP16"; diff $O/ref.txt $O/gen.txt | head; fi
rm -rf $O
