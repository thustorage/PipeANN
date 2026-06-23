// Check if CPU supports AVX512, AVX2, and AVX512VPOPCNTDQ instructions
// Return value (bitmask):
//   bit 0: AVX2
//   bit 1: AVX512F
//   bit 2: AVX512VPOPCNTDQ (requires AVX512F)

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

int main() {
  int info[4];
#if defined(_MSC_VER)
  __cpuidex(info, 7, 0);
#else
  __cpuid_count(7, 0, info[0], info[1], info[2], info[3]);
#endif
  int support_avx512f    = (info[1] & (1 << 16)) != 0;  // EBX bit 16: AVX512F
  int support_avx2       = (info[1] & (1 <<  5)) != 0;  // EBX bit  5: AVX2
  int support_vpopcntdq  = (info[2] & (1 << 14)) != 0;  // ECX bit 14: AVX512VPOPCNTDQ

  return (support_avx512f && support_vpopcntdq) << 2
       | support_avx512f << 1
       | support_avx2;
}