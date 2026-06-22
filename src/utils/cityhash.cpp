// Copyright (c) 2011 Google, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// CityHash, by Geoff Pike and Jyrki Alakuijala
// https://github.com/google/cityhash
//
// Minimal vendored subset: CityHash32 only.

#include "utils/cityhash.h"

#include <algorithm>
#include <cstring>

#if defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define pipeann_bswap_32(x) OSSwapInt32(x)
#elif defined(_MSC_VER)
#include <stdlib.h>
#define pipeann_bswap_32(x) _byteswap_ulong(x)
#else
#include <byteswap.h>
#define pipeann_bswap_32(x) bswap_32(x)
#endif

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define pipeann_u32_le(x) (pipeann_bswap_32(x))
#else
#define pipeann_u32_le(x) (x)
#endif

namespace pipeann {

namespace {

inline uint32_t UnalignedLoad32(const char *p) {
  uint32_t result;
  std::memcpy(&result, p, sizeof(result));
  return result;
}

inline uint32_t Fetch32(const char *p) {
  return pipeann_u32_le(UnalignedLoad32(p));
}

// Magic numbers for 32-bit hashing. Copied from Murmur3.
constexpr uint32_t kC1 = 0xcc9e2d51;
constexpr uint32_t kC2 = 0x1b873593;

inline uint32_t Fmix(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

inline uint32_t Rotate32(uint32_t val, int shift) {
  return shift == 0 ? val : ((val >> shift) | (val << (32 - shift)));
}

#define PIPEANN_PERMUTE3(a, b, c) \
  do {                            \
    std::swap(a, b);              \
    std::swap(a, c);              \
  } while (0)

inline uint32_t Mur(uint32_t a, uint32_t h) {
  a *= kC1;
  a = Rotate32(a, 17);
  a *= kC2;
  h ^= a;
  h = Rotate32(h, 19);
  return h * 5 + 0xe6546b64;
}

uint32_t Hash32Len0to4(const char *s, size_t len) {
  uint32_t b = 0;
  uint32_t c = 9;
  for (size_t i = 0; i < len; i++) {
    signed char v = static_cast<signed char>(s[i]);
    b = b * kC1 + static_cast<uint32_t>(v);
    c ^= b;
  }
  return Fmix(Mur(b, Mur(static_cast<uint32_t>(len), c)));
}

uint32_t Hash32Len5to12(const char *s, size_t len) {
  uint32_t a = static_cast<uint32_t>(len), b = a * 5, c = 9, d = b;
  a += Fetch32(s);
  b += Fetch32(s + len - 4);
  c += Fetch32(s + ((len >> 1) & 4));
  return Fmix(Mur(c, Mur(b, Mur(a, d))));
}

uint32_t Hash32Len13to24(const char *s, size_t len) {
  uint32_t a = Fetch32(s - 4 + (len >> 1));
  uint32_t b = Fetch32(s + 4);
  uint32_t c = Fetch32(s + len - 8);
  uint32_t d = Fetch32(s + (len >> 1));
  uint32_t e = Fetch32(s);
  uint32_t f = Fetch32(s + len - 4);
  uint32_t h = static_cast<uint32_t>(len);
  return Fmix(Mur(f, Mur(e, Mur(d, Mur(c, Mur(b, Mur(a, h)))))));
}

}  // namespace

uint32_t CityHash32(const char *s, size_t len) {
  if (len <= 24) {
    return len <= 12 ? (len <= 4 ? Hash32Len0to4(s, len) : Hash32Len5to12(s, len)) : Hash32Len13to24(s, len);
  }

  uint32_t h = static_cast<uint32_t>(len), g = kC1 * h, f = g;
  uint32_t a0 = Rotate32(Fetch32(s + len - 4) * kC1, 17) * kC2;
  uint32_t a1 = Rotate32(Fetch32(s + len - 8) * kC1, 17) * kC2;
  uint32_t a2 = Rotate32(Fetch32(s + len - 16) * kC1, 17) * kC2;
  uint32_t a3 = Rotate32(Fetch32(s + len - 12) * kC1, 17) * kC2;
  uint32_t a4 = Rotate32(Fetch32(s + len - 20) * kC1, 17) * kC2;
  h ^= a0;
  h = Rotate32(h, 19);
  h = h * 5 + 0xe6546b64;
  h ^= a2;
  h = Rotate32(h, 19);
  h = h * 5 + 0xe6546b64;
  g ^= a1;
  g = Rotate32(g, 19);
  g = g * 5 + 0xe6546b64;
  g ^= a3;
  g = Rotate32(g, 19);
  g = g * 5 + 0xe6546b64;
  f += a4;
  f = Rotate32(f, 19);
  f = f * 5 + 0xe6546b64;
  size_t iters = (len - 1) / 20;
  do {
    uint32_t aa0 = Rotate32(Fetch32(s) * kC1, 17) * kC2;
    uint32_t aa1 = Fetch32(s + 4);
    uint32_t aa2 = Rotate32(Fetch32(s + 8) * kC1, 17) * kC2;
    uint32_t aa3 = Rotate32(Fetch32(s + 12) * kC1, 17) * kC2;
    uint32_t aa4 = Fetch32(s + 16);
    h ^= aa0;
    h = Rotate32(h, 18);
    h = h * 5 + 0xe6546b64;
    f += aa1;
    f = Rotate32(f, 19);
    f = f * kC1;
    g += aa2;
    g = Rotate32(g, 18);
    g = g * 5 + 0xe6546b64;
    h ^= aa3 + aa1;
    h = Rotate32(h, 19);
    h = h * 5 + 0xe6546b64;
    g ^= aa4;
    g = pipeann_bswap_32(g) * 5;
    h += aa4 * 5;
    h = pipeann_bswap_32(h);
    f += aa0;
    PIPEANN_PERMUTE3(f, h, g);
    s += 20;
  } while (--iters != 0);
  g = Rotate32(g, 11) * kC1;
  g = Rotate32(g, 17) * kC1;
  f = Rotate32(f, 11) * kC1;
  f = Rotate32(f, 17) * kC1;
  h = Rotate32(h + g, 19);
  h = h * 5 + 0xe6546b64;
  h = Rotate32(h, 17) * kC1;
  h = Rotate32(h + f, 19);
  h = h * 5 + 0xe6546b64;
  h = Rotate32(h, 17) * kC1;
  return h;
}

}  // namespace pipeann
