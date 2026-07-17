#pragma once

#include "utils.h"
#include "utils/tsl/robin_set.h"
#include "utils/visited_set.h"

constexpr size_t MAX_N_SECTOR_READS = 128;
constexpr size_t MAX_N_EDGES = 2048;
constexpr size_t INDEX_SIZE_FACTOR = 2;  // space amplification during insert.

// Both unaligned and aligned.
// example: a record locates in [300, 500], then
// offset = 0, len = 4096 (aligned read for disk)
// u_offset = 300, u_len = 200 (unaligned read)
// Unaligned read: read u_len from u_offset, read to buf + 0.
struct IORequest {
  uint64_t offset;    // where to read from (page)
  uint64_t len;       // how much to read
  void *buf;          // where to read into
  uint32_t n_pending; // for SPDK internal use: one IORequest may be split.
  bool finished;      // for async IO
  uint64_t u_offset;  // where to read from (unaligned)
  uint64_t u_len;     // how much to read (unaligned)
  void *base;         // starting address of this sector scratch

  IORequest() : offset(0), len(0), buf(nullptr) {
  }

  IORequest(uint64_t offset, uint64_t len, void *buf, uint64_t u_offset, uint64_t u_len, void *base = nullptr)
      : offset(offset), len(len), buf(buf), u_offset(u_offset), u_len(u_len), base(base) {
    assert((uint64_t) buf % SECTOR_LEN == 0);
    assert(offset % SECTOR_LEN == 0);
    assert(len % SECTOR_LEN == 0);
  }
};

namespace pipeann {
  template<typename T, typename IdT = uint32_t>
  struct SSDIndexMetadata {
    // The order matches that on SSD.
    uint32_t nr = 0, nc = 0;
    uint64_t npoints = 0;  // size.
    uint64_t data_dim = 0;
    uint64_t entry_point = 0;
    uint64_t max_node_len = 0;  // dense node length.
    uint64_t nnodes_per_sector = 0;
    uint64_t npts_cur_shard = 0;
    uint64_t attr_size = 0;
    uint64_t max_npts = 0;  // capacity.
    uint64_t range = 0;     // maximum out-degree. For densified graph, range << range_dense (typically).
    uint64_t R_ood = 0;     // NGFix refine out-degree. Last R_ood positions of nnbrs are refine edges.

    /* temporary fields (currently not stored on disk). */
    uint64_t normal_node_len = 0;  // normal node length (do not read dense_nbrs).
    uint64_t range_dense = 0;      // R_dense for densified graph.
    IdT entry_point_id = 0;
    enum DataType : uint64_t {
      UNDEFINED = 0,
      FLOAT = 1,
      UINT8 = 2,
      INT8 = 3
    } data_type = UNDEFINED;  // currently unused.

    SSDIndexMetadata() = default;

    SSDIndexMetadata(uint64_t npoints, uint64_t data_dim, uint64_t entry_point, uint64_t max_node_len,
                     uint64_t nnodes_per_sector, uint64_t range, uint64_t attr_size, uint64_t R_ood)
        : npoints(npoints), data_dim(data_dim), entry_point(entry_point), max_node_len(max_node_len),
          nnodes_per_sector(nnodes_per_sector), npts_cur_shard(npoints), attr_size(attr_size), range(range),
          R_ood(R_ood), data_type(UNDEFINED) {
      this->init_temporary_fields();
    }

    void init_temporary_fields() {
      this->max_npts = npoints;
      this->range_dense = (max_node_len - data_dim * sizeof(T) - attr_size) / sizeof(unsigned) - 1;

      // Backward compatible: range is not stored on disk.
      if (this->range == 0 || this->range >= this->range_dense) {
        this->range = this->range_dense;
      }

      if (this->R_ood > this->range) {
        this->R_ood = 0;
      }

      this->normal_node_len = max_node_len - (range_dense - range) * sizeof(unsigned);
      this->entry_point_id = static_cast<IdT>(entry_point);
      assert(entry_point_id == entry_point);
    }

    // Bytes to read/write for one "group" of contiguous nodes on disk:
    //   - nnodes_per_sector > 0: group == 1 sector, holding nnodes_per_sector nodes.
    //   - nnodes_per_sector == 0: group == ROUND_UP(max_node_len, SECTOR_LEN), holding 1 node.
    // io_size excludes dense nbrs (use for compact reads); io_size_dense covers the full node.
    inline uint64_t io_size() const {
      return SECTOR_LEN * (nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(normal_node_len, SECTOR_LEN));
    }
    inline uint64_t io_size_dense() const {
      return SECTOR_LEN * (nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(max_node_len, SECTOR_LEN));
    }
    inline uint64_t nodes_per_io() const {
      return nnodes_per_sector > 0 ? nnodes_per_sector : 1;
    }

    // Disk sector number for a given loc (data region starts at sector 1; sector 0 is metadata).
    inline uint64_t loc_sector_no(uint64_t loc) const {
      return 1 + (nnodes_per_sector > 0 ? loc / nnodes_per_sector : loc * DIV_ROUND_UP(max_node_len, SECTOR_LEN));
    }
    inline uint64_t sector_to_loc(uint64_t sector_no, uint32_t sector_off) const {
      return nnodes_per_sector == 0 ? (sector_no - 1) / DIV_ROUND_UP(max_node_len, SECTOR_LEN)
                                    : (sector_no - 1) * nnodes_per_sector + sector_off;
    }

    // Unaligned (compact) offset for a loc — used as IORequest::u_offset / u_len payload.
    inline uint64_t u_loc_offset(uint64_t loc) const {
      return loc * max_node_len;
    }
    inline uint64_t u_loc_offset_nbr(uint64_t loc) const {
      return loc * max_node_len + data_dim * sizeof(T);
    }

    void print() const {
      LOG(INFO) << "Max npts: " << max_npts << " Npoints: " << npoints << " Entry point: " << entry_point
                << " Data dim: " << data_dim << " Range: " << range << " Range dense: " << range_dense
                << " Range ood: " << R_ood;
      LOG(INFO) << "Normal node len (without dense nbrs): " << normal_node_len
                << " Max node len (with dense nbrs): " << max_node_len << " Nnodes per sector: " << nnodes_per_sector
                << " Npts cur shard: " << npts_cur_shard << " Attr size: " << attr_size;
    }

    void load_from_disk_index(const std::string &filename, bool sharded = false) {
      if (file_exists(filename) == false) {
        LOG(ERROR) << "File " << filename << " does not exist.";
        exit(-1);
      }
      std::ifstream in(filename, std::ios::binary);
      load_from_disk_index(in, sharded);
      in.close();
    }

    void load_from_disk_index(std::ifstream &in, bool sharded = false) {
      in.read((char *) &nr, sizeof(uint32_t));
      in.read((char *) &nc, sizeof(uint32_t));
      LOG(INFO) << "Loading metadata from disk index, sharded: " << sharded << " nr: " << nr << " nc: " << nc;

      in.read((char *) &npoints, sizeof(uint64_t));
      in.read((char *) &data_dim, sizeof(uint64_t));

      in.read((char *) &entry_point, sizeof(uint64_t));
      in.read((char *) &max_node_len, sizeof(uint64_t));
      in.read((char *) &nnodes_per_sector, sizeof(uint64_t));
      in.read((char *) &npts_cur_shard, sizeof(uint64_t));
      in.read((char *) &attr_size, sizeof(uint64_t));
      in.read((char *) &range, sizeof(uint64_t));

      if (nr >= 9) {
        in.read((char *) &R_ood, sizeof(uint64_t));
      } else {
        this->R_ood = 0;
      }

      if (!sharded) {
        this->npts_cur_shard = this->npoints;
      }

      if (nr < 7) {  // backward compatible (no range & attr_size).
        this->attr_size = 0;
        this->range = 0;
      }

      this->init_temporary_fields();
    }

    void save_to_disk_index(const std::string &filename) {
      std::ofstream out(filename, std::ios::in | std::ios::out | std::ios::binary);
      save_to_disk_index(out);
      out.close();
    }

    void save_to_disk_index(std::ofstream &out) {
      // clear the first page.
      std::vector<char> buf(SECTOR_LEN, 0);
      out.write(buf.data(), buf.size());
      out.seekp(0, std::ios::beg);

      nr = 9;  // hard-coded for the number of uint64_t below.
      nc = 1;
      out.write((char *) &nr, sizeof(uint32_t));
      out.write((char *) &nc, sizeof(uint32_t));

      out.write((char *) &npoints, sizeof(uint64_t));
      out.write((char *) &data_dim, sizeof(uint64_t));

      out.write((char *) &entry_point, sizeof(uint64_t));
      out.write((char *) &max_node_len, sizeof(uint64_t));
      out.write((char *) &nnodes_per_sector, sizeof(uint64_t));
      out.write((char *) &npts_cur_shard, sizeof(uint64_t));
      out.write((char *) &attr_size, sizeof(uint64_t));
      out.write((char *) &range, sizeof(uint64_t));
      out.write((char *) &R_ood, sizeof(uint64_t));
    }
  };

  // The index is stored as fixed-size DiskNodes (records) on disk.
  // Each DiskNode has: [vector | <nnbrs, n_dense_nbrs> | nnbrs IDs | attrs (maybe 0 length) | n_dense_nbrs IDs ].
  // This struct serves as a reference to a DiskNode<T> in the in-memory page-aligned buffer.
  template<typename T>
  struct DiskNode {
    T *coords;
    // Old: uint32_t nnbrs. Insufficient connectivity for filtered search.
    // New: dense graph with <nnbrs> 1-hop nbrs, and <n_dense_nbrs - nnbrs> 2-hop nbrs.
    // - Cut the low 16 bits for the original nnbrs (compatibility, assume little-endian).
    // - The high 16 bits store the total number of dense nbrs.
    uint16_t &nnbrs;
    uint16_t &n_dense_nbrs;
    uint32_t *nbrs;
    void *attrs;
    uint32_t *dense_nbrs;

    DiskNode(char *page_buf, uint32_t loc, const SSDIndexMetadata<T> &meta)
        : coords((T *) (page_buf +
                        (meta.nnodes_per_sector == 0 ? 0 : (loc % meta.nnodes_per_sector) * meta.max_node_len))),
          nnbrs(*(uint16_t *) ((char *) coords + meta.data_dim * sizeof(T))),
          n_dense_nbrs(*(uint16_t *) ((char *) coords + meta.data_dim * sizeof(T) + sizeof(uint16_t))),
          nbrs((uint32_t *) ((char *) coords + meta.data_dim * sizeof(T) + sizeof(uint32_t))),
          attrs((void *) ((char *) coords + meta.data_dim * sizeof(T) + (1 + meta.range) * sizeof(uint32_t))),
          dense_nbrs((uint32_t *) ((char *) coords + meta.data_dim * sizeof(T) + (1 + meta.range) * sizeof(uint32_t) +
                                   meta.attr_size)) {
    }
  };

  struct QueryBuffer {
    uint8_t *coord_scratch_ = nullptr;  // MUST BE AT LEAST [aligned_dim * sizeof(T)], for current vector in comparison.

    char *sector_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN], for sectors.
    uint64_t sector_idx = 0;         // index of next [SECTOR_LEN] scratch to use

    uint32_t *nbr_id_scratch = nullptr;     // MUST BE AT LEAST [MAX_N_EDGES], for nbr IDs (used in filter search).
    float *nbr_ctx_scratch = nullptr;       // MUST BE AT LEAST [256 * NCHUNKS], for pq table distance.
    float *aligned_dist_scratch = nullptr;  // MUST BE AT LEAST pipeann MAX_DEGREE, for exact dist.
    uint8_t *nbr_vec_scratch = nullptr;     // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE], for neighbor PQ vectors.
    uint8_t *aligned_query_ = nullptr;      // MUST BE AT LEAST [aligned_dim * sizeof(T)], for aligned query.
    char *update_buf = nullptr;             // Dynamic allocate in insert_in_place.

    // Prefetchable visited set (see visited_set.h): insert-only, the probe
    // address is a pure function of the id so the search loop can prefetch a
    // neighbor's slot a few iterations ahead of insert() and hide the miss.
    FlatVisitedSet *visited = nullptr;
    tsl::robin_set<unsigned> *page_visited = nullptr;
    IORequest reqs[MAX_N_SECTOR_READS];

    template<typename T>
    T *coord_scratch() {
      return (T *) coord_scratch_;
    }

    template<typename T>
    T *aligned_query() {
      return (T *) aligned_query_;
    }

    void reset() {
      sector_idx = 0;
      visited->clear();  // does not deallocate memory.
      page_visited->clear();
    }
  };
};  // namespace pipeann
