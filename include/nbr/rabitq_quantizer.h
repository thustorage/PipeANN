#pragma once
#include "utils.h"
#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace pipeann {
  inline constexpr const char *RABITQ_METADATA_SUFFIXES[] = {"_rabitq_metadata.bin", "_rabitq_rotator.bin",
                                                             "_rabitq_pivots.bin"};

  // Full-code RaBitQ. Each record is [Header][packed unsigned coordinates].
  // One common block-orthogonal transform is shared by queries and centroids.
  template<typename T, unsigned Bits>
  class RaBitQQuantizer {
    static_assert(Bits == 1 || Bits == 2 || Bits == 4 || Bits == 8);

   public:
    struct Header {
      float scale;  // norm-preserving scale for centered lattice coordinates
      float norm2;  // residual norm for single-center L2; full reconstructed norm otherwise
      uint32_t sum;
      uint16_t pivot;
      uint16_t reserved;
    };
    struct QueryHeader {
      float scale, offset, norm2;
      uint32_t sum;
    };
    static_assert(sizeof(Header) == 16 && sizeof(QueryHeader) == 16);
    size_t dim = 0, padded_dim = 0, clusters = 0;
    Metric metric;
    std::vector<int8_t> signs;
    std::vector<float> pivots;  // rotated, padded
    double quant_scale = 1;     // calibrated RaBitQ scale on normalized residuals
    explicit RaBitQQuantizer(Metric m) : metric(m) {
    }
    static constexpr size_t MAX_CENTROIDS = size_t(UINT16_MAX) + 1;
    bool residual_l2() const {
      return metric == L2 && clusters == 1;
    }
    static size_t pad_dim(size_t dim) {
      return ROUND_UP(dim, 128);
    }
    static size_t vector_size(size_t dim) {
      return sizeof(Header) + pad_dim(dim) * Bits / 8;
    }
    size_t vector_size() const {
      return vector_size(dim);
    }
    void save(const std::string &prefix) const {
      uint64_t metadata[] = {dim, padded_dim, vector_size()};
      save_bin(prefix + "_rabitq_metadata.bin", metadata, 1, 3);
      save_bin(prefix + "_rabitq_pivots.bin", pivots.data(), clusters, padded_dim);
      std::ofstream f(prefix + "_rabitq_rotator.bin", std::ios::binary);
      f.write(reinterpret_cast<const char *>(&quant_scale), sizeof(quant_scale));
      f.write(reinterpret_cast<const char *>(signs.data()), signs.size());
      f.close();
      if (!f)
        throw std::runtime_error("Cannot save RaBitQ parameters");
    }
    void load(const std::string &prefix) {
      std::vector<uint64_t> metadata;
      size_t rows, cols;
      load_bin(prefix + "_rabitq_metadata.bin", metadata, rows, cols);
      if (rows != 1 || cols != 3 || !metadata[0] || metadata[0] > 1048576 || metadata[1] != pad_dim(metadata[0]))
        throw std::runtime_error("Invalid RaBitQ metadata");
      if (metadata[2] != vector_size(metadata[0]))
        throw std::runtime_error("Invalid RaBitQ record size");
      std::vector<float> loaded_pivots;
      load_bin(prefix + "_rabitq_pivots.bin", loaded_pivots, rows, cols);
      if (!rows || rows > MAX_CENTROIDS || cols != metadata[1])
        throw std::runtime_error("Invalid RaBitQ pivot shape");
      initialize(metadata[0], rows);
      pivots = std::move(loaded_pivots);
      std::ifstream f(prefix + "_rabitq_rotator.bin", std::ios::binary);
      f.read(reinterpret_cast<char *>(&quant_scale), sizeof(quant_scale));
      f.read(reinterpret_cast<char *>(signs.data()), signs.size());
      if (!f || !std::isfinite(quant_scale) || quant_scale <= 0)
        throw std::runtime_error("Truncated/invalid RaBitQ parameters");
      for (auto sign : signs)
        if (sign != -1 && sign != 1)
          throw std::runtime_error("Invalid FWHT sign");
    }
    size_t query_ctx_size() const {
      return sizeof(QueryHeader) + padded_dim + padded_dim + clusters * sizeof(float);
    }
    static unsigned value(const uint8_t *p, size_t i) {
      return (p[i * Bits / 8] >> (i * Bits % 8)) & ((1u << Bits) - 1);
    }
    static Header header(const uint8_t *p) {
      Header h;
      // Fixed-size byte load; optimized into registers without alignment or aliasing assumptions.
      memcpy(&h, p, sizeof(h));
      return h;
    }
    static void fwht(float *x, size_t n) {
      for (size_t width = 1; width < n; width *= 2)
        for (size_t base = 0; base < n; base += 2 * width)
          for (size_t j = 0; j < width; ++j) {
            float a = x[base + j], b = x[base + j + width];
            x[base + j] = a + b;
            x[base + j + width] = a - b;
          }
      float scale = 1.0f / std::sqrt(static_cast<float>(n));
      for (size_t i = 0; i < n; ++i)
        x[i] *= scale;
    }
    void transform(float *x, bool inverse = false) const {
      if (!inverse)
        for (size_t i = 0; i < padded_dim; ++i)
          x[i] *= signs[i];
      for (size_t offset = 0; offset < padded_dim;) {
        size_t n = 1;
        while (n <= (padded_dim - offset) / 2)
          n *= 2;
        fwht(x + offset, n);
        offset += n;
      }
      if (inverse)
        for (size_t i = 0; i < padded_dim; ++i)
          x[i] *= signs[i];
    }
    void initialize(size_t d, size_t k, uint32_t seed = 42) {
      dim = d;
      padded_dim = pad_dim(d);
      clusters = k;
      if (!d || !k || k > MAX_CENTROIDS)
        throw std::invalid_argument("Invalid RaBitQ dimensions/centroids");
      signs.resize(padded_dim);
      std::mt19937 rng(seed);
      for (auto &s : signs)
        s = (rng() & 1) ? 1 : -1;
      pivots.resize(k * padded_dim);
    }
    // RaBitQ angular scale search, calibrated once rather than repeated for
    // every vector (the fast RaBitQ quantizer). Midpoints are half-integers.
    static double optimal_scale(const float *abs_normalized, size_t n) {
      if constexpr (Bits == 1)
        return 1;
      constexpr unsigned maxcode = (1u << (Bits - 1)) - 1;
      static constexpr double tight[] = {0, .15, .20, .52, .59, .71, .75, .77};
      double maxv = *std::max_element(abs_normalized, abs_normalized + n);
      if (maxv == 0)
        return 1;
      double end = (maxcode + 10.0) / maxv;
      double start = end * tight[Bits - 1];
      std::vector<unsigned> codes(n);
      double numerator = 0, denominator = 0;
      using Event = std::pair<double, size_t>;
      std::priority_queue<Event, std::vector<Event>, std::greater<Event>> events;
      for (size_t i = 0; i < n; ++i) {
        codes[i] = std::min<unsigned>(maxcode, static_cast<unsigned>(start * abs_normalized[i]));
        double c = codes[i] + .5;
        numerator += c * abs_normalized[i];
        denominator += c * c;
        if (codes[i] < maxcode && abs_normalized[i] > 0)
          events.emplace((codes[i] + 1.0) / abs_normalized[i], i);
      }
      double best = numerator * numerator / denominator, scale = start;
      while (!events.empty() && events.top().first <= end) {
        auto [t, i] = events.top();
        events.pop();
        ++codes[i];
        denominator += 2 * codes[i];
        numerator += abs_normalized[i];
        double score = numerator * numerator / denominator;
        if (score > best) {
          best = score;
          scale = t * (1 + 1e-7);
        }
        if (codes[i] < maxcode)
          events.emplace((codes[i] + 1.0) / abs_normalized[i], i);
      }
      return scale;
    }
    void calibrate(const float *samples, const uint32_t *assignments, size_t count) {
      if constexpr (Bits == 1) {
        quant_scale = 1;
        return;
      }
      double total = 0;
      size_t used = 0;
      std::vector<float> r(padded_dim);
      for (size_t j = 0; j < std::min<size_t>(count, 128); ++j) {
        std::fill(r.begin(), r.end(), 0);
        std::copy_n(samples + j * dim, dim, r.data());
        transform(r.data());
        double norm = 0;
        for (size_t i = 0; i < padded_dim; ++i) {
          r[i] -= pivots[assignments[j] * padded_dim + i];
          norm += r[i] * r[i];
        }
        if (norm == 0)
          continue;
        for (auto &x : r)
          x = std::abs(x) / std::sqrt(norm);
        total += optimal_scale(r.data(), padded_dim);
        ++used;
      }
      quant_scale = used ? total / used : 1;
    }
    void encode(const T *input, uint8_t *out, int pivot = -1) const {
      std::vector<float> r(padded_dim, 0);
      for (size_t i = 0; i < dim; ++i)
        r[i] = static_cast<float>(input[i]);
      transform(r.data());
      encode_rotated(r.data(), out, pivot);
    }

    void decode_rotated(const uint8_t *code, float *out) const {
      Header h = header(code);
      const float *center = pivots.data() + h.pivot * padded_dim;
      constexpr double mid = ((1u << Bits) - 1) * .5f;
      for (size_t i = 0; i < padded_dim; ++i)
        out[i] = center[i] + h.scale * (value(code + sizeof(Header), i) - mid);
    }
    void decode(const uint8_t *code, T *out) const {
      std::vector<float> x(padded_dim);
      decode_rotated(code, x.data());
      transform(x.data(), true);
      for (size_t i = 0; i < dim; ++i) {
        if constexpr (std::is_integral_v<T>)
          x[i] = std::clamp(x[i], (float) std::numeric_limits<T>::lowest(), (float) std::numeric_limits<T>::max());
        out[i] = static_cast<T>(x[i]);
      }
    }
    void initialize_query(const T *input, void *context) const {
      auto h = static_cast<QueryHeader *>(context);
      auto q = reinterpret_cast<uint8_t *>(h + 1);
      auto planes = q + padded_dim;
      auto center_ip = reinterpret_cast<float *>(planes + padded_dim);
      std::vector<float> x(padded_dim, 0);
      for (size_t i = 0; i < dim; ++i)
        x[i] = static_cast<float>(input[i]);
      transform(x.data());
      if (residual_l2())
        for (size_t i = 0; i < padded_dim; ++i)
          x[i] -= pivots[i];
      auto limits = std::minmax_element(x.begin(), x.end());
      h->offset = *limits.first;
      h->scale = (*limits.second - *limits.first) / 255;
      h->sum = 0;
      h->norm2 = 0;
      double query_norm = 0;
      memset(planes, 0, padded_dim);
      for (size_t i = 0; i < padded_dim; ++i) {
        q[i] = h->scale > 0 ? std::clamp<int>(std::lround((x[i] - h->offset) / h->scale), 0, 255) : 0;
        x[i] = h->offset + h->scale * q[i];
        h->sum += q[i];
        query_norm += double(x[i]) * x[i];
        if constexpr (Bits == 1)
          for (unsigned bit = 0; bit < 8; ++bit)
            planes[bit * (padded_dim / 8) + i / 8] |= ((q[i] >> bit) & 1) << (i % 8);
      }
      h->norm2 = query_norm;
      if (residual_l2()) {
        center_ip[0] = 0;
        return;
      }
      for (size_t c = 0; c < clusters; ++c) {
        double ip = 0;
#pragma omp simd reduction(+ : ip)
        for (size_t i = 0; i < padded_dim; ++i)
          ip += double(x[i]) * pivots[c * padded_dim + i];
        center_ip[c] = ip;
      }
    }
    // Prepared query versus stored code, or two stored codes.
    float compute_distance(const void *a, const uint8_t *b) const;
    float compute_distance(const uint8_t *a, const uint8_t *b) const;
    void compute_distances(const void *a, const uint8_t *b, const uint32_t *ids, size_t n, float *out,
                           size_t data_size) const;
    void compute_distances(const uint8_t *a, const uint8_t *b, const uint32_t *ids, size_t n, float *out,
                           size_t data_size) const;

   private:
    void encode_rotated(float *r, uint8_t *out, int pivot) const {
      if (clusters == 1)
        pivot = 0;
      if (pivot < 0) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t c = 0; c < clusters; ++c) {
          double d = 0;
          for (size_t i = 0; i < padded_dim; ++i) {
            double x = r[i] - pivots[c * padded_dim + i];
            d += x * x;
          }
          if (d < best) {
            best = d;
            pivot = c;
          }
        }
      }
      const float *center = pivots.data() + pivot * padded_dim;
      double norm = 0;
      for (size_t i = 0; i < padded_dim; ++i) {
        r[i] -= center[i];
        norm += r[i] * r[i];
      }
      double multiplier = norm > 0 ? quant_scale / std::sqrt(norm) : 0;
      constexpr unsigned half = 1u << (Bits - 1);
      constexpr double mid = half - .5f;
      uint8_t *packed = out + sizeof(Header);
      memset(packed, 0, padded_dim * Bits / 8);
      Header h{};
      h.pivot = pivot;
      double lattice_norm = 0;
      for (size_t i = 0; i < padded_dim; ++i) {
        unsigned magnitude = std::min<unsigned>(half - 1, static_cast<unsigned>(std::abs(r[i]) * multiplier));
        unsigned c = r[i] >= 0 ? half + magnitude : half - 1 - magnitude;
        packed[i * Bits / 8] |= c << (i * Bits % 8);
        h.sum += c;
        lattice_norm += (c - mid) * (c - mid);
      }
      h.scale = std::sqrt(norm / lattice_norm);
      if (residual_l2()) {
        h.norm2 = double(h.scale) * h.scale * lattice_norm;
      } else {
        double norm_full = 0;
        for (size_t i = 0; i < padded_dim; ++i) {
          double x = center[i] + h.scale * (value(packed, i) - mid);
          norm_full += x * x;
        }
        h.norm2 = norm_full;
      }
      memcpy(out, &h, sizeof(h));
    }
#if defined(__AVX512BW__)
    static __m512i load64(const uint8_t *p);
#elif defined(__AVX2__)
    static __m256i load32(const uint8_t *p);
#endif
    static uint64_t dot(const uint8_t *a, const uint8_t *b, size_t padded_dim, uint32_t known_sum);
  };

  // Load one SIMD block of unsigned coordinates, unpacking 2/4-bit codes as needed.
#if defined(__AVX512BW__)
  template<typename T, unsigned Bits>
  [[gnu::always_inline]] inline __m512i RaBitQQuantizer<T, Bits>::load64(const uint8_t *p) {
    static_assert(Bits == 2 || Bits == 4 || Bits == 8);
    if constexpr (Bits == 8) {
      return _mm512_loadu_si512(p);
    } else if constexpr (Bits == 4) {
      auto w = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *) p));
      auto mask = _mm512_set1_epi16(15);
      return _mm512_or_si512(_mm512_and_si512(w, mask),
                             _mm512_slli_epi16(_mm512_and_si512(_mm512_srli_epi16(w, 4), mask), 8));
    } else {
      auto w = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) p));
      auto mask = _mm512_set1_epi32(3);
      return _mm512_or_si512(_mm512_or_si512(_mm512_and_si512(w, mask),
                                             _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(w, 2), mask), 8)),
                             _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(w, 4), mask), 16),
                                             _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(w, 6), mask), 24)));
    }
  }
#elif defined(__AVX2__)
  template<typename T, unsigned Bits>
  [[gnu::always_inline]] inline __m256i RaBitQQuantizer<T, Bits>::load32(const uint8_t *p) {
    static_assert(Bits == 2 || Bits == 4 || Bits == 8);
    if constexpr (Bits == 8) {
      return _mm256_loadu_si256((const __m256i *) p);
    } else if constexpr (Bits == 4) {
      const auto w = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *) p));
      const auto mask = _mm256_set1_epi16(15);
      return _mm256_or_si256(_mm256_and_si256(w, mask),
                            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(w, 4), mask), 8));
    } else {
      const auto w = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) p));
      const auto mask = _mm256_set1_epi32(3);
      return _mm256_or_si256(
          _mm256_or_si256(_mm256_and_si256(w, mask),
                          _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(w, 2), mask), 8)),
          _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(w, 4), mask), 16),
                          _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(w, 6), mask), 24)));
    }
  }
#endif

  // Padded codes only; known_sum is the cached sum of all query bytes.
  // For 1-bit codes, q is followed by its eight bitplanes, prepared by initialize_query().
  template<typename T, unsigned Bits>
  [[gnu::always_inline]] inline uint64_t RaBitQQuantizer<T, Bits>::dot(const uint8_t *q, const uint8_t *code,
                                                                    size_t padded_dim, uint32_t known_sum) {
    assert(padded_dim % 64 == 0);
    if constexpr (Bits == 1) {
      const uint8_t *planes = q + padded_dim;
      uint64_t total = 0;
      for (size_t base = 0; base < padded_dim; base += 64) {
        uint64_t word;
        memcpy(&word, code + base / 8, sizeof(word));
        for (unsigned bit = 0; bit < 8; ++bit) {
          uint64_t qw;
          memcpy(&qw, planes + bit * padded_dim / 8 + base / 8, sizeof(qw));
          total += uint64_t(__builtin_popcountll(qw & word)) << bit;
        }
      }
      return total;
    } else {
      int64_t total = 0;
      size_t i = 0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
      // Bound partial sums for arbitrarily large dimensions; reduce once per
      // 4096 coordinates rather than once per cache line.
      while (i < padded_dim) {
        const size_t end = std::min(padded_dim, i + 4096);
        __m512i accum = _mm512_setzero_si512();
        for (; i < end; i += 64) {
          __m512i x = load64(code + i * Bits / 8);
          if constexpr (Bits == 8)
            x = _mm512_xor_si512(x, _mm512_set1_epi8(-128));
          __m512i qq = _mm512_loadu_si512(q + i);
          accum = _mm512_dpbusd_epi32(accum, qq, x);
        }
        total += _mm512_reduce_add_epi32(accum);
      }
      // VNNI uses a signed second operand. Undo its shift after reduction.
      if constexpr (Bits == 8)
        total += 128LL * known_sum;
#elif defined(__AVX512BW__)
      for (; i < padded_dim; i += 64) {
        const auto packed = load64(code + i * Bits / 8);
        const auto x0 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(packed));
        const auto x1 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(packed, 1));
        const auto q0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *) (q + i)));
        const auto q1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *) (q + i + 32)));
        total += _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_madd_epi16(q0, x0), _mm512_madd_epi16(q1, x1)));
      }
#elif defined(__AVX2__)
      while (i < padded_dim) {
        const size_t end = std::min(padded_dim, i + 4096);
        auto accum = _mm256_setzero_si256();
        for (; i < end; i += 32) {
          const auto packed = load32(code + i * Bits / 8);
          const auto qq = _mm256_loadu_si256((const __m256i *) (q + i));
          if constexpr (Bits == 8) {
            const auto x0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(packed));
            const auto x1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(packed, 1));
            const auto q0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(qq));
            const auto q1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(qq, 1));
            accum = _mm256_add_epi32(accum, _mm256_add_epi32(_mm256_madd_epi16(q0, x0), _mm256_madd_epi16(q1, x1)));
          } else {
            // At most 2 * 255 * 15 per pair: maddubs cannot saturate for 2/4-bit codes.
            const auto pairs = _mm256_maddubs_epi16(qq, packed);
            accum = _mm256_add_epi32(accum, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
          }
        }
        auto sum = _mm_add_epi32(_mm256_castsi256_si128(accum), _mm256_extracti128_si256(accum, 1));
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4e));
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb1));
        total += _mm_cvtsi128_si32(sum);
      }
#else
      for (; i < padded_dim; ++i)
        total += static_cast<uint32_t>(q[i]) * value(code, i);
#endif
      return static_cast<uint64_t>(total);
    }
  }

  template<typename T, unsigned Bits>
  [[gnu::always_inline]] inline void RaBitQQuantizer<T, Bits>::compute_distances(
      const void *a, const uint8_t *b, const uint32_t *ids, size_t n, float *out, size_t data_size) const {
    if (!n)
      return;
    const auto h = *static_cast<const QueryHeader *>(a);
    const auto q = static_cast<const uint8_t *>(a) + sizeof(QueryHeader);
    const size_t dimensions = padded_dim;
    const auto centers = reinterpret_cast<const float *>(q + 2 * dimensions);
    constexpr double mid = ((1u << Bits) - 1) * .5;
    constexpr size_t kPrefetchWindow = 8;
    for (size_t i = 0; i < std::min(kPrefetchWindow, n); ++i)
      prefetch_vector(reinterpret_cast<const char *>(b + ids[i] * data_size), data_size);
    for (size_t i = 0; i < n; ++i) {
      if (kPrefetchWindow < n - i)
        prefetch_vector(reinterpret_cast<const char *>(b + ids[i + kPrefetchWindow] * data_size), data_size);
      const uint8_t *record = b + ids[i] * data_size;
      const auto x = header(record);
      const uint8_t *code = record + sizeof(Header);
      const uint64_t product = dot(q, code, dimensions, h.sum);
      const double ip = centers[x.pivot] +
                        x.scale * (h.scale * (double(product) - mid * h.sum) +
                                   h.offset * (double(x.sum) - mid * dimensions));
      if (metric == INNER_PRODUCT) {
        out[i] = float(metric_scale<T>() - ip);
      } else if (metric == COSINE) {
        const double norm = std::sqrt(double(h.norm2) * x.norm2);
        out[i] = norm > 0 ? float(metric_scale<T>() * (1 - std::clamp(ip / norm, -1.0, 1.0))) : metric_scale<T>();
      } else {
        out[i] = float(std::max(0.0, double(h.norm2) + x.norm2 - 2 * ip));
      }
    }
  }

  template<typename T, unsigned Bits>
  [[gnu::always_inline]] inline void RaBitQQuantizer<T, Bits>::compute_distances(
      const uint8_t *a, const uint8_t *b, const uint32_t *ids, size_t n, float *out, size_t data_size) const {
    if (!n)
      return;
    if (Bits != 8 || clusters != 1)
      throw std::invalid_argument("Symmetric RaBitQ distance requires 8-bit single-center codes");
    const auto ah = header(a);
    const uint8_t *ac = a + sizeof(Header);
    const size_t dimensions = padded_dim;
    const double a_norm2 = ah.norm2;
    constexpr double mid = 127.5;
    constexpr size_t kPrefetchWindow = 8;
    for (size_t i = 0; i < std::min(n, kPrefetchWindow); ++i)
      prefetch_vector(reinterpret_cast<const char *>(b + ids[i] * data_size), data_size);
    if (metric == L2) {
      const double a_sum = ah.sum;
      const double twice_a_scale = 2 * double(ah.scale);
      for (size_t i = 0; i < n; ++i) {
        if (kPrefetchWindow < n - i)
          prefetch_vector(reinterpret_cast<const char *>(b + ids[i + kPrefetchWindow] * data_size), data_size);
        const uint8_t *record = b + ids[i] * data_size;
        if (a == record) {
          out[i] = 0;
          continue;
        }
        const auto bh = header(record);
        const double ip = double(dot(ac, record + sizeof(Header), dimensions, ah.sum)) - mid * (a_sum + bh.sum) +
                          dimensions * mid * mid;
        out[i] = float(std::max(0.0, a_norm2 + bh.norm2 - twice_a_scale * bh.scale * ip));
      }
      return;
    }
    const bool inner_product = metric == INNER_PRODUCT;
    for (size_t i = 0; i < n; ++i) {
      if (kPrefetchWindow < n - i)
        prefetch_vector(reinterpret_cast<const char *>(b + ids[i + kPrefetchWindow] * data_size), data_size);
      const uint8_t *record = b + ids[i] * data_size;
      if (a == record) {
        out[i] = inner_product ? metric_scale<T>() - ah.norm2 : 0;
        continue;
      }
      const auto bh = header(record);
      const uint8_t *bc = record + sizeof(Header);
      double ip = 0;
#pragma omp simd reduction(+ : ip)
      for (size_t j = 0; j < dimensions; ++j) {
        const double x = pivots[j] + ah.scale * (ac[j] - mid);
        const double y = pivots[j] + bh.scale * (bc[j] - mid);
        ip += x * y;
      }
      if (inner_product) {
        out[i] = float(metric_scale<T>() - ip);
      } else {
        const double norm = std::sqrt(a_norm2 * bh.norm2);
        out[i] = norm > 0 ? float(metric_scale<T>() * (1 - std::clamp(ip / norm, -1.0, 1.0))) : metric_scale<T>();
      }
    }
  }

  template<typename T, unsigned Bits>
  [[gnu::always_inline]] inline float RaBitQQuantizer<T, Bits>::compute_distance(
      const void *a, const uint8_t *b) const {
    const uint32_t id = 0;
    float result;
    compute_distances(a, b, &id, 1, &result, vector_size());
    return result;
  }

  template<typename T, unsigned Bits>
  [[gnu::always_inline]] inline float RaBitQQuantizer<T, Bits>::compute_distance(
      const uint8_t *a, const uint8_t *b) const {
    const uint32_t id = 0;
    float result;
    compute_distances(a, b, &id, 1, &result, vector_size());
    return result;
  }

}  // namespace pipeann
