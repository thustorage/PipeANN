#pragma once

// Shared field <-> Attribute codec for the Milvus-compatible layer.
//
// This is the single source of truth for converting between user-facing typed
// scalar values (int / float / bool / string / label-array) and the on-disk
// `Attributes` encoding consumed by the filter/attr-index machinery. Both the
// gRPC server (src/server/milvus_server.cpp) and the Python binding
// (src/python) call into these functions so the two stacks stay bit-for-bit
// identical.
//
// Logical types (a refinement of the attr "type" used by the attr index):
//   "label"  -> Array<int>, stored verbatim as a list of uint32 label ids.
//   "range"  with logical int   -> single uint32 == the integer value.
//            with logical float  -> single uint32 == order-preserving f32 code.
//            with logical bool   -> single uint32 in {0,1}.
//   "string" -> packed UTF-8 bytes (see pack_string_attr).
//
// NOTE on range storage: a *node's* stored range attribute is a single value
// (RangeSelector::is_member reads only target_attr[0]); the two-element
// [lo, hi) form is only used for *query* attributes. encode_field therefore
// emits a single-element Attribute for range fields, and decode_field reads
// element [0].

#include <cstdint>
#include <cstring>
#include <string>

#include "filter/attribute.h"
#include "utils/picojson.h"

namespace pipeann {
namespace server {

// ---------------------------------------------------------------------------
// Order-preserving IEEE-754 float32 <-> uint32 (bijective).
// Mirrors pipeann/collection.py:_encode_float32 and milvus.py:_encode_float.
//   positive floats: flip the sign bit.
//   negative floats: flip all bits.
// Guarantees a < b  <=>  encode(a) < encode(b) over the uint32 ordering, so the
// range attr index can compare encoded floats directly.
// ---------------------------------------------------------------------------
inline uint32_t encode_float_ordered(float value) {
  uint32_t u;
  std::memcpy(&u, &value, sizeof(u));
  return (u & 0x80000000u) ? (~u & 0xFFFFFFFFu) : (u ^ 0x80000000u);
}

inline float decode_float_ordered(uint32_t code) {
  uint32_t u = (code & 0x80000000u) ? (code ^ 0x80000000u) : (~code & 0xFFFFFFFFu);
  float value;
  std::memcpy(&value, &u, sizeof(value));
  return value;
}

// ---------------------------------------------------------------------------
// Logical type tags. attr_type is the coarse attr-index family ("label" /
// "range" / "string"); logical_type refines "range" into int / float / bool so
// values round-trip to their original Python/Milvus representation.
// ---------------------------------------------------------------------------
enum class LogicalType { Int, Float, Bool, String, Label };

inline LogicalType logical_type_from_string(const std::string &s) {
  if (s == "float") return LogicalType::Float;
  if (s == "bool") return LogicalType::Bool;
  if (s == "string") return LogicalType::String;
  if (s == "label") return LogicalType::Label;
  return LogicalType::Int;
}

inline const char *logical_type_to_string(LogicalType t) {
  switch (t) {
    case LogicalType::Float: return "float";
    case LogicalType::Bool: return "bool";
    case LogicalType::String: return "string";
    case LogicalType::Label: return "label";
    case LogicalType::Int:
    default: return "int";
  }
}

// ---------------------------------------------------------------------------
// Single-value encoders. Each returns the node-stored Attribute for one field
// of one row. The coarse attr family is implied by the logical type:
//   Int/Float/Bool -> "range" (single-element value)
//   String         -> "string" (packed bytes)
//   Label          -> "label" (caller builds the uint32 list directly)
// ---------------------------------------------------------------------------
inline Attribute encode_int_range(int64_t v) { return Attribute{static_cast<uint32_t>(v)}; }

inline Attribute encode_float_range(double v) { return Attribute{encode_float_ordered(static_cast<float>(v))}; }

inline Attribute encode_bool_range(bool b) { return Attribute{b ? 1u : 0u}; }

// ---------------------------------------------------------------------------
// Decode a node-stored Attribute back to its original typed value as a
// picojson::value, for returning via output_fields. Integers round-trip through
// int32 (the representable range for filterable scalars; large int64 keys are a
// known limitation shared with the Python stack). Numbers are emitted as double
// because picojson is built without PICOJSON_USE_INT64 here.
// ---------------------------------------------------------------------------
inline picojson::value decode_field(const Attribute &attr, LogicalType type) {
  switch (type) {
    case LogicalType::Float:
      return attr.empty() ? picojson::value() : picojson::value(static_cast<double>(decode_float_ordered(attr[0])));
    case LogicalType::Bool:
      return attr.empty() ? picojson::value() : picojson::value(attr[0] != 0);
    case LogicalType::String:
      return picojson::value(string_attr_to_string(attr));
    case LogicalType::Label: {
      picojson::array arr;
      arr.reserve(attr.size());
      for (uint32_t v : attr) arr.push_back(picojson::value(static_cast<double>(v)));
      return picojson::value(arr);
    }
    case LogicalType::Int:
    default:
      return attr.empty() ? picojson::value()
                          : picojson::value(static_cast<double>(static_cast<int32_t>(attr[0])));
  }
}

}  // namespace server
}  // namespace pipeann

