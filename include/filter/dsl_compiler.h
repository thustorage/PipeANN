#ifndef DSL_COMPILER_H_
#define DSL_COMPILER_H_

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "filter/attribute.h"
#include "filter/selector.h"
#include "utils/picojson.h"

namespace pipeann::dsl {

  struct FieldInfo {
    uint32_t key = 0;
    std::string type;                 // "label" | "range" | "string"
    AttrIndex *attr_index = nullptr;  // borrowed; lifetime must exceed Selector
    uint32_t n_vectors = 0;
  };

  using Schema = std::unordered_map<std::string, FieldInfo>;

  // Result of compiling a SQL-like filter expression.
  // `attrs_template` has literal slots filled; placeholder slots are left empty
  // and must be filled at bind time using `slot_map[varName] -> slot_idx`.
  struct CompiledFilter {
    Selector *selector = nullptr;
    Attributes attrs_template;
    std::unordered_map<std::string, uint32_t> slot_map;
    std::unordered_map<std::string, std::string> var_field_type;
  };

  namespace detail {

    inline bool is_placeholder(const std::string &s) {
      return s.size() >= 3 && s[0] == '$' && s[1] == '$';
    }

    inline std::string placeholder_name(const std::string &s) {
      return s.substr(2);
    }

    inline uint32_t schema_n_vectors(const Schema &schema) {
      return schema.begin()->second.n_vectors;
    }

    inline uint32_t to_u32(const picojson::value &v) {
      return static_cast<uint32_t>(v.get<double>());
    }

    inline void append_record(Attribute &dest, const Attribute &bare) {
      dest.push_back(static_cast<uint32_t>(bare.size()));
      dest.insert(dest.end(), bare.begin(), bare.end());
    }

    // Allocate a fresh slot in the query Attributes. The same field appearing
    // in multiple clauses (e.g. {"$or": [{"f": "a"}, {"f": "b"}]}) needs distinct
    // slots so each Selector reads its own query value.
    inline uint32_t alloc_slot(Attributes &attrs, Attribute value) {
      uint32_t slot = static_cast<uint32_t>(attrs.attrs_.size());
      attrs.set(slot, std::move(value));
      return slot;
    }

    inline uint32_t alloc_placeholder_slot(CompiledFilter &cf, const std::string &var, const std::string &field_type) {
      uint32_t slot = alloc_slot(cf.attrs_template, Attribute{});
      cf.slot_map.emplace(var, slot);
      cf.var_field_type.emplace(var, field_type);
      return slot;
    }

    inline Selector *compile_node(const picojson::value &node, const Schema &schema, CompiledFilter &cf);

    inline Selector *compile_field_clause(const FieldInfo &info, const picojson::value &val, CompiledFilter &cf) {
      if (!val.is<picojson::object>()) {
        if (val.is<std::string>() && is_placeholder(val.get<std::string>())) {
          uint32_t slot = alloc_placeholder_slot(cf, placeholder_name(val.get<std::string>()), info.type);
          if (info.type == "label")
            return new LabelOrSelector(slot, info.key, info.attr_index);
          if (info.type == "range")
            return new RangeSelector(slot, info.key, info.attr_index);
          return new StringEqSelector(slot, info.key, info.attr_index);
        }
        if (info.type == "label") {
          uint32_t slot = alloc_slot(cf.attrs_template, Attribute{to_u32(val)});
          return new LabelOrSelector(slot, info.key, info.attr_index);
        }
        if (info.type == "range") {
          uint32_t v = to_u32(val);
          uint32_t hi = (v == std::numeric_limits<uint32_t>::max()) ? v : v + 1;
          uint32_t slot = alloc_slot(cf.attrs_template, Attribute{v, hi});
          return new RangeSelector(slot, info.key, info.attr_index);
        }
        Attribute rec;
        append_record(rec, pack_string_attr(val.get<std::string>()));
        uint32_t slot = alloc_slot(cf.attrs_template, std::move(rec));
        return new StringEqSelector(slot, info.key, info.attr_index);
      }

      const auto &ops = val.get<picojson::object>();

      if (ops.count("$eq")) {
        return compile_field_clause(info, ops.at("$eq"), cf);
      }

      if (ops.count("$in")) {
        const auto &items = ops.at("$in").get<picojson::array>();
        if (info.type == "label") {
          Attribute vals;
          for (const auto &item : items)
            vals.push_back(to_u32(item));
          uint32_t slot = alloc_slot(cf.attrs_template, std::move(vals));
          return new LabelOrSelector(slot, info.key, info.attr_index);
        }
        Attribute rec;
        for (const auto &item : items)
          append_record(rec, pack_string_attr(item.get<std::string>()));
        uint32_t slot = alloc_slot(cf.attrs_template, std::move(rec));
        return new StringEqSelector(slot, info.key, info.attr_index);
      }

      if (ops.count("$all")) {
        const auto &val_node = ops.at("$all");
        if (val_node.is<std::string>() && is_placeholder(val_node.get<std::string>())) {
          uint32_t slot = alloc_placeholder_slot(cf, placeholder_name(val_node.get<std::string>()), info.type);
          return new LabelAndSelector(slot, info.key, info.attr_index);
        }
        Attribute vals;
        for (const auto &item : val_node.get<picojson::array>())
          vals.push_back(to_u32(item));
        uint32_t slot = alloc_slot(cf.attrs_template, std::move(vals));
        return new LabelAndSelector(slot, info.key, info.attr_index);
      }

      if (ops.count("$prefix")) {
        const auto &v = ops.at("$prefix");
        Attribute rec;
        append_record(rec, pack_string_attr(v.get<std::string>()));
        uint32_t slot = alloc_slot(cf.attrs_template, std::move(rec));
        return new StringPrefixSelector(slot, info.key, info.attr_index);
      }
      if (ops.count("$suffix")) {
        const auto &v = ops.at("$suffix");
        Attribute rec;
        append_record(rec, pack_string_attr(v.get<std::string>()));
        uint32_t slot = alloc_slot(cf.attrs_template, std::move(rec));
        return new StringSuffixSelector(slot, info.key, info.attr_index);
      }
      if (ops.count("$like")) {
        const auto &v = ops.at("$like");
        Attribute rec;
        append_record(rec, pack_string_attr(v.get<std::string>()));
        uint32_t slot = alloc_slot(cf.attrs_template, std::move(rec));
        return new StringLikeSelector(slot, info.key, info.attr_index);
      }

      // Range comparators ($gt/$ge/$lt/$le) combined into one [lo, hi) interval.
      uint64_t lo = 0;
      uint64_t hi = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
      for (const auto &[op, v] : ops) {
        uint32_t val_u32 = to_u32(v);
        if (op == "$gt")
          lo = std::max<uint64_t>(lo, static_cast<uint64_t>(val_u32) + 1);
        else if (op == "$ge")
          lo = std::max<uint64_t>(lo, val_u32);
        else if (op == "$lt")
          hi = std::min<uint64_t>(hi, val_u32);
        else if (op == "$le")
          hi = std::min<uint64_t>(hi, static_cast<uint64_t>(val_u32) + 1);
      }
      lo = std::min<uint64_t>(lo, std::numeric_limits<uint32_t>::max());
      hi = std::min<uint64_t>(hi, std::numeric_limits<uint32_t>::max());
      uint32_t slot = alloc_slot(cf.attrs_template, Attribute{static_cast<uint32_t>(lo), static_cast<uint32_t>(hi)});
      return new RangeSelector(slot, info.key, info.attr_index);
    }

    inline Selector *compile_node(const picojson::value &node, const Schema &schema, CompiledFilter &cf) {
      const auto &obj = node.get<picojson::object>();
      std::vector<Selector *> children;

      auto compile_logical_array = [&](const picojson::value &val) {
        std::vector<Selector *> subs;
        for (const auto &item : val.get<picojson::array>()) {
          subs.push_back(compile_node(item, schema, cf));
        }
        return subs;
      };

      for (const auto &[name, val] : obj) {
        if (name == "$and") {
          children.push_back(new AndSelector(compile_logical_array(val)));
        } else if (name == "$or") {
          children.push_back(new OrSelector(compile_logical_array(val)));
        } else if (name == "$not") {
          children.push_back(new NotSelector(compile_node(val, schema, cf), schema_n_vectors(schema)));
        } else {
          const FieldInfo &info = schema.at(name);
          Selector *child = nullptr;
          if (val.is<picojson::object>()) {
            const auto &ops = val.get<picojson::object>();
            if (ops.count("$ne")) {
              child = new NotSelector(compile_field_clause(info, ops.at("$ne"), cf), schema_n_vectors(schema));
            } else if (ops.count("$nin")) {
              picojson::object inner;
              inner["$in"] = ops.at("$nin");
              picojson::value inner_val(inner);
              child = new NotSelector(compile_field_clause(info, inner_val, cf), schema_n_vectors(schema));
            } else {
              child = compile_field_clause(info, val, cf);
            }
          } else {
            child = compile_field_clause(info, val, cf);
          }
          children.push_back(child);
        }
      }

      if (children.size() == 1) {
        Selector *only = children.front();
        children.clear();
        return only;
      }
      return new AndSelector(std::move(children));
    }

    inline std::string json_escape(const std::string &s) {
      std::ostringstream out;
      for (unsigned char c : s) {
        switch (c) {
          case '"': out << "\\\""; break;
          case '\\': out << "\\\\"; break;
          case '\b': out << "\\b"; break;
          case '\f': out << "\\f"; break;
          case '\n': out << "\\n"; break;
          case '\r': out << "\\r"; break;
          case '\t': out << "\\t"; break;
          default:
            if (c < 0x20) {
              out << "\\u" << std::hex << std::uppercase << (int) c;
            } else {
              out << static_cast<char>(c);
            }
        }
      }
      return out.str();
    }

    struct SqlValue {
      bool is_string = false;
      bool is_placeholder = false;  // a "$$var" binding placeholder
      std::string s;
      std::string json() const {
        // Strings and placeholders both serialize as JSON strings; a placeholder
        // keeps its literal "$$var" text so compile_field_clause's
        // is_placeholder() check fires downstream and reserves a bind slot.
        return (is_string || is_placeholder) ? (std::string("\"") + json_escape(s) + "\"") : s;
      }
    };

    struct SqlParser {
      enum class TokKind { End, Ident, Number, String, LParen, RParen, LBrack, RBrack, Comma, Op };
      struct Tok {
        TokKind kind = TokKind::End;
        std::string text;
      };

      std::string input;
      size_t pos = 0;
      Tok tok;

      explicit SqlParser(std::string s) : input(std::move(s)) {
        next();
      }

      [[noreturn]] void error(const std::string &msg) const {
        throw std::runtime_error("SQL filter parse error at byte " + std::to_string(pos) + ": " + msg);
      }

      static std::string lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        return s;
      }

      void skip_ws() {
        while (pos < input.size() && std::isspace((unsigned char) input[pos])) pos++;
      }

      void next() {
        skip_ws();
        tok = Tok{};
        if (pos >= input.size()) return;
        char c = input[pos];
        if (std::isalpha((unsigned char) c) || c == '_' || c == '$') {
          size_t st = pos++;
          while (pos < input.size()) {
            char d = input[pos];
            if (!(std::isalnum((unsigned char) d) || d == '_' || d == '.' || d == '$')) break;
            pos++;
          }
          tok.kind = TokKind::Ident;
          tok.text = input.substr(st, pos - st);
          return;
        }
        if (std::isdigit((unsigned char) c)) {
          size_t st = pos++;
          while (pos < input.size() && std::isdigit((unsigned char) input[pos])) pos++;
          tok.kind = TokKind::Number;
          tok.text = input.substr(st, pos - st);
          return;
        }
        if (c == '\'' || c == '"') {
          char quote = c;
          pos++;
          std::string out;
          while (pos < input.size()) {
            char d = input[pos++];
            if (d == quote) {
              tok.kind = TokKind::String;
              tok.text = out;
              return;
            }
            if (d == '\\' && pos < input.size()) {
              char e = input[pos++];
              if (e == 'n') out.push_back('\n');
              else if (e == 'r') out.push_back('\r');
              else if (e == 't') out.push_back('\t');
              else out.push_back(e);
            } else {
              out.push_back(d);
            }
          }
          error("unterminated string literal");
        }
        pos++;
        if (c == '(') tok.kind = TokKind::LParen;
        else if (c == ')') tok.kind = TokKind::RParen;
        else if (c == '[') tok.kind = TokKind::LBrack;
        else if (c == ']') tok.kind = TokKind::RBrack;
        else if (c == ',') tok.kind = TokKind::Comma;
        else if (c == '=' || c == '!' || c == '<' || c == '>') {
          tok.kind = TokKind::Op;
          tok.text.push_back(c);
          if (pos < input.size() && (input[pos] == '=' || (c == '<' && input[pos] == '>'))) tok.text.push_back(input[pos++]);
        } else {
          error(std::string("unexpected character '") + c + "'");
        }
      }

      bool ident_is(const char *kw) const {
        return tok.kind == TokKind::Ident && lower(tok.text) == kw;
      }

      void expect(TokKind k, const char *what) {
        if (tok.kind != k) error(std::string("expected ") + what);
        next();
      }

      SqlValue parse_value() {
        SqlValue v;
        if (tok.kind == TokKind::String) {
          v.is_string = true;
          v.s = tok.text;
          next();
          return v;
        }
        if (tok.kind == TokKind::Number) {
          v.s = tok.text;
          next();
          return v;
        }
        if (ident_is("true") || ident_is("false")) {
          v.s = ident_is("true") ? "1" : "0";
          next();
          return v;
        }
        // "$$var" binding placeholder: the tokenizer lexes it as an Ident (it
        // allows leading/embedded '$'). Carry it through as a placeholder value
        // so the bind-slot machinery in compile_field_clause picks it up.
        if (tok.kind == TokKind::Ident && tok.text.size() >= 3 && tok.text[0] == '$' && tok.text[1] == '$') {
          v.is_placeholder = true;
          v.s = tok.text;
          next();
          return v;
        }
        error("expected literal value");
      }

      std::string field_clause(const std::string &field, const std::string &body) {
        return "{\"" + json_escape(field) + "\":" + body + "}";
      }

      std::string unary_not(const std::string &node) {
        return "{\"$not\":" + node + "}";
      }

      std::string logical(const char *op, std::vector<std::string> nodes) {
        if (nodes.size() == 1) return nodes[0];
        std::string out = std::string("{\"") + op + "\":[";
        for (size_t i = 0; i < nodes.size(); i++) {
          if (i) out += ",";
          out += nodes[i];
        }
        out += "]}";
        return out;
      }

      std::string parse_expr() { return parse_or(); }

      std::string parse_or() {
        std::vector<std::string> nodes{parse_and()};
        while (ident_is("or")) {
          next();
          nodes.push_back(parse_and());
        }
        return logical("$or", std::move(nodes));
      }

      std::string parse_and() {
        std::vector<std::string> nodes{parse_unary()};
        while (ident_is("and")) {
          next();
          nodes.push_back(parse_unary());
        }
        return logical("$and", std::move(nodes));
      }

      std::string parse_unary() {
        if (ident_is("not")) {
          next();
          return unary_not(parse_unary());
        }
        if (tok.kind == TokKind::LParen) {
          next();
          std::string n = parse_expr();
          expect(TokKind::RParen, "')'");
          return n;
        }
        return parse_cmp();
      }

      std::string parse_array_values() {
        expect(TokKind::LBrack, "'['");
        std::string arr = "[";
        bool first = true;
        while (tok.kind != TokKind::RBrack) {
          if (!first) {
            expect(TokKind::Comma, "','");
            arr += ",";
          }
          arr += parse_value().json();
          first = false;
        }
        arr += "]";
        expect(TokKind::RBrack, "']'");
        return arr;
      }

      std::string parse_cmp() {
        if (tok.kind != TokKind::Ident) error("expected field name");
        std::string field = tok.text;
        next();

        // Handle function-call syntax: array_contains(field, val),
        // array_contains_all(field, [vals]), array_contains_any(field, [vals])
        std::string func_name = lower(field);
        if ((func_name == "array_contains" || func_name == "array_contains_all" ||
             func_name == "array_contains_any") &&
            tok.kind == TokKind::LParen) {
          next();  // consume '('
          if (tok.kind != TokKind::Ident) error("expected field name in " + func_name + "()");
          std::string actual_field = tok.text;
          next();
          expect(TokKind::Comma, "','");
          std::string n;
          if (func_name == "array_contains") {
            SqlValue v = parse_value();
            n = field_clause(actual_field, v.json());
          } else if (func_name == "array_contains_all") {
            // Accept either a literal array or a "$$var" placeholder bound to a
            // per-query .spmat (the benchmark/groundtruth path).
            if (tok.kind == TokKind::Ident) {
              n = field_clause(actual_field, "{\"$all\":" + parse_value().json() + "}");
            } else {
              n = field_clause(actual_field, "{\"$all\":" + parse_array_values() + "}");
            }
          } else {  // array_contains_any
            if (tok.kind == TokKind::Ident) {
              n = field_clause(actual_field, "{\"$in\":" + parse_value().json() + "}");
            } else {
              n = field_clause(actual_field, "{\"$in\":" + parse_array_values() + "}");
            }
          }
          expect(TokKind::RParen, "')'");
          return n;
        }

        bool negate = false;
        if (ident_is("not")) {
          negate = true;
          next();
        }

        if (ident_is("between")) {
          next();
          SqlValue lo = parse_value();
          if (!ident_is("and")) error("expected AND in BETWEEN expression");
          next();
          SqlValue hi = parse_value();
          std::string n = field_clause(field, "{\"$ge\":" + lo.json() + ",\"$le\":" + hi.json() + "}");
          return negate ? unary_not(n) : n;
        }

        if (ident_is("in")) {
          next();
          expect(TokKind::LBrack, "'['");
          std::vector<SqlValue> vals;
          if (tok.kind != TokKind::RBrack) {
            vals.push_back(parse_value());
            while (tok.kind == TokKind::Comma) {
              next();
              vals.push_back(parse_value());
            }
          }
          expect(TokKind::RBrack, "']'");
          bool all_numeric = true;
          for (const auto &v : vals) all_numeric = all_numeric && !v.is_string;
          if (all_numeric) {
            std::vector<std::string> eqs;
            eqs.reserve(vals.size());
            for (const auto &v : vals) eqs.push_back(field_clause(field, v.json()));
            std::string n = logical("$or", std::move(eqs));
            return negate ? unary_not(n) : n;
          }

          std::string arr = "[";
          for (size_t i = 0; i < vals.size(); i++) {
            if (i) arr += ",";
            arr += vals[i].json();
          }
          arr += "]";
          std::string n = field_clause(field, "{\"$in\":" + arr + "}");
          return negate ? unary_not(n) : n;
        }

        if (ident_is("like")) {
          next();
          SqlValue v = parse_value();
          if (!v.is_string) error("LIKE expects a string literal");
          std::string n;
          if (!v.s.empty() && v.s.back() == '%' && v.s.find('%') == v.s.size() - 1 && v.s.find('_') == std::string::npos) {
            v.s.pop_back();
            n = field_clause(field, "{\"$prefix\":" + v.json() + "}");
          } else if (!v.s.empty() && v.s.front() == '%' && v.s.find('%', 1) == std::string::npos && v.s.find('_') == std::string::npos) {
            v.s.erase(v.s.begin());
            n = field_clause(field, "{\"$suffix\":" + v.json() + "}");
          } else if (v.s.find('%') == std::string::npos && v.s.find('_') == std::string::npos) {
            n = field_clause(field, v.json());
          } else {
            n = field_clause(field, "{\"$like\":" + v.json() + "}");
          }
          return negate ? unary_not(n) : n;
        }

        if (tok.kind != TokKind::Op) error("expected comparison operator");
        std::string op = tok.text;
        next();
        SqlValue v = parse_value();

        std::string n;
        if (op == "=" || op == "==") n = field_clause(field, v.json());
        else if (op == "!=" || op == "<>") n = field_clause(field, "{\"$ne\":" + v.json() + "}");
        else if (op == ">") n = field_clause(field, "{\"$gt\":" + v.json() + "}");
        else if (op == ">=") n = field_clause(field, "{\"$ge\":" + v.json() + "}");
        else if (op == "<") n = field_clause(field, "{\"$lt\":" + v.json() + "}");
        else if (op == "<=") n = field_clause(field, "{\"$le\":" + v.json() + "}");
        else error("unsupported comparison operator");
        return negate ? unary_not(n) : n;
      }
    };

    inline std::string sql_to_mongo_json(const std::string &expr) {
      SqlParser p(expr);
      auto out = p.parse_expr();
      if (p.tok.kind != SqlParser::TokKind::End) p.error("unexpected trailing token");
      return out;
    }

    inline std::string trim_left(const std::string &s) {
      size_t i = 0;
      while (i < s.size() && std::isspace((unsigned char) s[i])) i++;
      return s.substr(i);
    }

  }  // namespace detail

  // Compile a Milvus-like SQL boolean filter expression into a CompiledFilter.
  // Caller takes ownership of the returned Selector tree.
  //
  // SQL is the only accepted filter language. Internally the SQL is transcoded
  // to the internal JSON node form that compile_node consumes.
  //
  // Values may use placeholders of the form "$$varName" (mirroring the MongoDB
  // aggregation $$variable convention). A placeholder reserves a slot in
  // `attrs_template` that the caller must fill via bind_row / bind_batch_per_var
  // before running search. When no placeholders are used, `attrs_template` is
  // fully populated and `slot_map` is empty.
  inline CompiledFilter compile(const std::string &sql, const Schema &schema) {
    std::string src = detail::sql_to_mongo_json(detail::trim_left(sql));
    picojson::value root;
    picojson::parse(root, src);
    CompiledFilter out;
    out.selector = detail::compile_node(root, schema, out);
    return out;
  }

  // Result of loading a unified schema config: AttrIndexes (owned by caller via
  // `base_stores`) plus a ready-to-compile Schema keyed by field name.
  struct LoadedSchema {
    std::map<uint32_t, AttrIndex *> base_stores;  // key -> AttrIndex* (caller frees)
    Schema schema;                                // name -> FieldInfo (borrows AttrIndex)
  };

  // Load AttrIndexes and build a dsl::Schema from a unified schema config:
  //   {
  //     "attr_indexes": [
  //       {"name": "category", "key": 0, "type": "label",  "file": "..."},
  //       {"name": "price",    "key": 1, "type": "range",  "file": "..."}
  //     ],
  //     ...
  //   }
  // Other top-level keys (e.g. "filter", "bindings") are ignored here.
  namespace detail {
    inline picojson::value parse_config_file(const std::string &config_path) {
      std::ifstream f(config_path);
      picojson::value config;
      picojson::parse(config, f);
      return config;
    }

    inline LoadedSchema load_schema_from_parsed(const picojson::value &config, uint64_t n_vectors) {
      LoadedSchema out;
      const auto &arr = config.get<picojson::object>().at("attr_indexes").get<picojson::array>();
      for (const auto &item : arr) {
        const auto &obj = item.get<picojson::object>();
        std::string name = obj.at("name").get<std::string>();
        uint32_t key = static_cast<uint32_t>(obj.at("key").get<double>());
        std::string type = obj.at("type").get<std::string>();
        std::string file = obj.at("file").get<std::string>();
        AttrIndex *ai = pipeann::load_attr_index_from_file(file, type, n_vectors);
        out.base_stores[key] = ai;
        FieldInfo info;
        info.key = key;
        info.type = type;
        info.attr_index = ai;
        info.n_vectors = static_cast<uint32_t>(n_vectors);
        out.schema.emplace(name, info);
      }
      return out;
    }
  }  // namespace detail

  // Bind a single row of placeholder values onto a CompiledFilter, producing an
  // Attributes ready to feed into search. `values` maps each $$var name to the
  // raw Attribute (already decoded to the field's runtime representation):
  //   - label: list of u32 label values
  //   - range: 2-element [lo, hi) interval
  //   - string: packed multi-record bytes (see pack_string_attr + append_record)
  inline Attributes bind_row(const CompiledFilter &cf, const std::unordered_map<std::string, Attribute> &values) {
    Attributes out = cf.attrs_template;
    for (const auto &[var, slot] : cf.slot_map) {
      out.set(slot, values.at(var));
    }
    return out;
  }

  // Batch-bind from per-var .spmat files. Each file decodes to one Attribute
  // per query row; all files share the same row count (taken from the first).
  inline std::vector<Attributes> bind_batch_per_var(const CompiledFilter &cf,
                                                    const std::unordered_map<std::string, std::string> &var_to_path) {
    if (cf.slot_map.empty())
      return {cf.attrs_template};
    std::vector<Attributes> out;
    for (const auto &[var, slot] : cf.slot_map) {
      auto rows = pipeann::decode_spmat_rows(var_to_path.at(var), cf.var_field_type.at(var));
      if (out.empty())
        out.assign(rows.size(), cf.attrs_template);
      for (size_t i = 0; i < rows.size(); i++)
        out[i].set(slot, std::move(rows[i]));
    }
    return out;
  }

  // End-to-end filter config loader: parses the unified config
  //   {"attr_indexes": [...], "filter": "<SQL expr>", "bindings": {var: spmat_path}}
  // and returns (selector, per-query attributes, base AttrIndex map). The
  // Selector borrows raw AttrIndex pointers from the third element; keep the
  // map alive (or accept the leak in short-lived processes) until the Selector
  // is done being used.
  inline std::tuple<Selector *, std::vector<Attributes>, std::map<uint32_t, AttrIndex *>> load_filter_from_json(
      const std::string &config_path, uint64_t n_vectors) {
    auto config = detail::parse_config_file(config_path);
    auto loaded_schema = detail::load_schema_from_parsed(config, n_vectors);

    const auto &root = config.get<picojson::object>();
    // `filter` is a SQL expression string (may contain "$$var" placeholders).
    auto cf = compile(root.at("filter").get<std::string>(), loaded_schema.schema);

    std::unordered_map<std::string, std::string> bindings;
    auto bind_it = root.find("bindings");
    if (bind_it != root.end()) {
      for (const auto &[var, path_val] : bind_it->second.get<picojson::object>()) {
        bindings[var] = path_val.get<std::string>();
      }
    }
    auto query_attrs = bind_batch_per_var(cf, bindings);
    return {cf.selector, std::move(query_attrs), std::move(loaded_schema.base_stores)};
  }

}  // namespace pipeann::dsl

#endif  // DSL_COMPILER_H_
