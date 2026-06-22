#pragma once

// Persistent id<->tag<->document key-value store backed by RocksDB.
//
// Header-only so that RocksDB is only required by targets that actually
// instantiate DocStore (the gRPC server and the Python binding).
//
// On-disk key layout (single column family, one prefix byte):
//   't' + tag(8B big-endian)  -> id + '\0' + document   (tag-ordered)
//   'i' + id(variable bytes)  -> tag(8B big-endian)      (reverse lookup)

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <rocksdb/cache.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/table.h>
#include <rocksdb/write_batch.h>

#include "utils/log.h"

namespace pipeann {

namespace doc_store_detail {

constexpr char kTagPrefix = 't';
constexpr char kIdPrefix = 'i';

inline void put_be64(char *dst, uint64_t v) {
  for (int i = 0; i < 8; i++) dst[i] = static_cast<char>((v >> (8 * (7 - i))) & 0xFF);
}

inline uint64_t get_be64(const char *src, size_t len) {
  uint64_t v = 0;
  for (size_t i = 0; i < 8 && i < len; i++) v = (v << 8) | static_cast<unsigned char>(src[i]);
  return v;
}

inline std::string tag_key(uint32_t tag) {
  std::string k(9, '\0');
  k[0] = kTagPrefix;
  put_be64(&k[1], tag);
  return k;
}

inline std::string id_key(const std::string &id) {
  std::string k;
  k.reserve(1 + id.size());
  k.push_back(kIdPrefix);
  k.append(id);
  return k;
}

inline std::string encode_tag_value(const std::string &id, const std::string &doc) {
  std::string v;
  v.reserve(id.size() + 1 + doc.size());
  v.append(id);
  v.push_back('\0');
  v.append(doc);
  return v;
}

inline void decode_tag_value(const std::string &v, std::string &id, std::string &doc) {
  size_t p = v.find('\0');
  if (p == std::string::npos) {
    id = v;
    doc.clear();
  } else {
    id.assign(v, 0, p);
    doc.assign(v, p + 1, std::string::npos);
  }
}

}  // namespace doc_store_detail

class DocStore {
 public:
  DocStore() = default;
  ~DocStore() { close(); }
  DocStore(const DocStore &) = delete;
  DocStore &operator=(const DocStore &) = delete;

  bool open(const std::string &dir) {
    close();
    rocksdb::Options options;
    options.create_if_missing = true;
    rocksdb::BlockBasedTableOptions tbo;
    auto cache = rocksdb::NewLRUCache(256ull << 20);
    tbo.block_cache = cache;
    options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(tbo));

    rocksdb::DB *db = nullptr;
    rocksdb::Status s = rocksdb::DB::Open(options, dir, &db);
    if (!s.ok()) {
      LOG(ERROR) << "DocStore: failed to open " << dir << ": " << s.ToString();
      return false;
    }
    db_ = db;
    block_cache_ = std::move(cache);
    return true;
  }

  void close() {
    if (db_) {
      delete db_;
      db_ = nullptr;
    }
    block_cache_.reset();
  }

  bool is_open() const { return db_ != nullptr; }

  void put_batch(const std::vector<std::tuple<uint32_t, std::string, std::string>> &rows) {
    using namespace doc_store_detail;
    if (!db_ || rows.empty()) return;
    rocksdb::WriteBatch batch;
    char tagbuf[8];
    for (const auto &[tag, id, doc] : rows) {
      batch.Put(tag_key(tag), encode_tag_value(id, doc));
      put_be64(tagbuf, tag);
      batch.Put(id_key(id), rocksdb::Slice(tagbuf, 8));
    }
    rocksdb::WriteOptions wo;
    rocksdb::Status s = db_->Write(wo, &batch);
    if (!s.ok()) LOG(ERROR) << "DocStore: put_batch failed: " << s.ToString();
  }

  // out_found (when non-null) reports, per tag, whether the doc-store still has a
  // live entry for it. A tag absent here was deleted (lazy index deletes leave
  // tombstones the attr index still returns), so callers use this to drop them.
  void get_by_tags(const std::vector<uint32_t> &tags, std::vector<std::string> &out_ids,
                   std::vector<std::string> &out_docs, std::vector<char> *out_found = nullptr) const {
    using namespace doc_store_detail;
    out_ids.assign(tags.size(), std::string());
    out_docs.assign(tags.size(), std::string());
    if (out_found) out_found->assign(tags.size(), 0);
    if (!db_ || tags.empty()) return;

    std::vector<std::string> keys(tags.size());
    std::vector<rocksdb::Slice> slices(tags.size());
    for (size_t i = 0; i < tags.size(); i++) {
      keys[i] = tag_key(tags[i]);
      slices[i] = keys[i];
    }
    std::vector<std::string> values;
    std::vector<rocksdb::Status> statuses = db_->MultiGet(rocksdb::ReadOptions(), slices, &values);
    for (size_t i = 0; i < statuses.size(); i++) {
      if (statuses[i].ok()) {
        decode_tag_value(values[i], out_ids[i], out_docs[i]);
        if (out_found) (*out_found)[i] = 1;
      }
    }
  }

  // Number of live entries (one per inserted row, decremented on delete). Used
  // for count(*): npoints() on the index includes not-yet-merged tombstones.
  int64_t count() const {
    using namespace doc_store_detail;
    if (!db_) return 0;
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(rocksdb::ReadOptions()));
    int64_t n = 0;
    std::string lower(1, kTagPrefix);
    for (it->Seek(lower); it->Valid(); it->Next()) {
      const rocksdb::Slice &k = it->key();
      if (k.size() == 0 || k.data()[0] != kTagPrefix) break;
      if (k.size() != 9) continue;
      n++;
    }
    return n;
  }

  void get_tags_by_ids(const std::vector<std::string> &ids, std::vector<int64_t> &out_tags) const {
    using namespace doc_store_detail;
    out_tags.assign(ids.size(), -1);
    if (!db_ || ids.empty()) return;

    std::vector<std::string> keys(ids.size());
    std::vector<rocksdb::Slice> slices(ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
      keys[i] = id_key(ids[i]);
      slices[i] = keys[i];
    }
    std::vector<std::string> values;
    std::vector<rocksdb::Status> statuses = db_->MultiGet(rocksdb::ReadOptions(), slices, &values);
    for (size_t i = 0; i < statuses.size(); i++) {
      if (statuses[i].ok()) out_tags[i] = static_cast<int64_t>(get_be64(values[i].data(), values[i].size()));
    }
  }

  void get_docs_by_ids(const std::vector<std::string> &ids, std::vector<std::string> &out_ids,
                       std::vector<std::string> &out_docs) const {
    out_ids.clear();
    out_docs.clear();
    if (!db_ || ids.empty()) return;

    std::vector<int64_t> tags;
    get_tags_by_ids(ids, tags);
    std::vector<uint32_t> existing_tags;
    std::vector<size_t> existing_idx;
    for (size_t i = 0; i < ids.size(); i++) {
      if (tags[i] >= 0) {
        existing_tags.push_back(static_cast<uint32_t>(tags[i]));
        existing_idx.push_back(i);
      }
    }
    std::vector<std::string> gids, gdocs;
    get_by_tags(existing_tags, gids, gdocs);
    for (size_t j = 0; j < existing_idx.size(); j++) {
      out_ids.push_back(ids[existing_idx[j]]);
      out_docs.push_back(gdocs[j]);
    }
  }

  void delete_by_ids(const std::vector<std::string> &ids) {
    using namespace doc_store_detail;
    if (!db_ || ids.empty()) return;
    std::vector<int64_t> tags;
    get_tags_by_ids(ids, tags);
    rocksdb::WriteBatch batch;
    for (size_t i = 0; i < ids.size(); i++) {
      if (tags[i] < 0) continue;
      batch.Delete(tag_key(static_cast<uint32_t>(tags[i])));
      batch.Delete(id_key(ids[i]));
    }
    rocksdb::Status s = db_->Write(rocksdb::WriteOptions(), &batch);
    if (!s.ok()) LOG(ERROR) << "DocStore: delete_by_ids failed: " << s.ToString();
  }

  int64_t max_tag() const {
    using namespace doc_store_detail;
    if (!db_) return -1;
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(rocksdb::ReadOptions()));
    std::string upper(9, '\0');
    upper[0] = kTagPrefix;
    for (int i = 1; i < 9; i++) upper[i] = static_cast<char>(0xFF);
    it->SeekForPrev(upper);
    if (it->Valid() && it->key().size() == 9 && it->key().data()[0] == kTagPrefix) {
      return static_cast<int64_t>(get_be64(it->key().data() + 1, 8));
    }
    return -1;
  }

  void scan(std::vector<uint32_t> &out_tags, std::vector<std::string> &out_ids,
            std::vector<std::string> &out_docs) const {
    using namespace doc_store_detail;
    out_tags.clear();
    out_ids.clear();
    out_docs.clear();
    if (!db_) return;
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(rocksdb::ReadOptions()));
    std::string lower(1, kTagPrefix);
    for (it->Seek(lower); it->Valid(); it->Next()) {
      const rocksdb::Slice &k = it->key();
      if (k.size() == 0 || k.data()[0] != kTagPrefix) break;
      if (k.size() != 9) continue;
      out_tags.push_back(static_cast<uint32_t>(get_be64(k.data() + 1, 8)));
      std::string id, doc;
      decode_tag_value(it->value().ToString(), id, doc);
      out_ids.push_back(std::move(id));
      out_docs.push_back(std::move(doc));
    }
  }

  void clear() {
    using namespace doc_store_detail;
    if (!db_) return;
    rocksdb::WriteBatch batch;
    batch.DeleteRange(std::string(1, kTagPrefix), std::string(1, kTagPrefix + 1));
    batch.DeleteRange(std::string(1, kIdPrefix), std::string(1, kIdPrefix + 1));
    rocksdb::Status s = db_->Write(rocksdb::WriteOptions(), &batch);
    if (!s.ok()) LOG(ERROR) << "DocStore: clear failed: " << s.ToString();
  }

  void flush() {
    if (!db_) return;
    db_->FlushWAL(true);
    rocksdb::FlushOptions fo;
    db_->Flush(fo);
  }

 private:
  rocksdb::DB *db_ = nullptr;
  std::shared_ptr<void> block_cache_;
};

}  // namespace pipeann
