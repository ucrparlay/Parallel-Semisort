
#ifndef COLLECT_REDUCE_H_
#define COLLECT_REDUCE_H_

#include "internal/get_time.h"
#include "internal/sample_sort.h"
#include "internal/uninitialized_sequence.h"
#include "primitives.h"
#include "semisort.h"
#include "sequence.h"
#include "slice.h"

namespace parlay {

constexpr size_t PARLAY_COLLECT_REDUCE_BASE_CASE_SIZE = 1 << 14;
constexpr size_t BLOCK_SIZE = 1024;

template<typename InIterator, typename OutIterator, typename Helper>
size_t collect_reduce_serial(const slice<InIterator, InIterator> In,
                             [[maybe_unused]] slice<OutIterator, OutIterator> Out, const Helper &helper,
                             size_t shift_bits) {
  using val_type = typename Helper::val_type;
  size_t n = In.size();
  if (n == 0) {
    return 0;
  }
  size_t bits = log2_up(static_cast<size_t>(n * 1.1));
  size_t size = size_t{1} << bits;
  size_t mask = size - 1;
  auto table = sequence<std::pair<size_t, val_type>>(size, {ULLONG_MAX, helper.monoid.identity});
  for (size_t j = 0; j < n; j++) {
    size_t v = (helper.hash(In[j].first) >> shift_bits) & mask;
    while (table[v].first != ULLONG_MAX && !helper.equal(In[j].first, In[table[v].first].first)) {
      v = (v + 1) & mask;
    }
    if (table[v].first == ULLONG_MAX) {
      table[v].first = j;
    }
    table[v].second = helper.monoid(table[v].second, helper.get_value(In[j]));
  }
  // number of distinct keys
  size_t tot = 0;
  for (size_t j = 0; j < size; j++) {
    size_t idx = table[j].first;
    if (idx != ULLONG_MAX) {
      assign_dispatch(Out[tot].first, In[idx].first, uninitialized_copy_tag());
      Out[tot].second = table[j].second;
      tot++;
    }
  }
  return tot;
}

template<typename s_size_t, typename InIterator, typename OutIterator, typename Helper>
size_t collect_reduce_(const slice<InIterator, InIterator> In, slice<OutIterator, OutIterator> Out,
                       const Helper &helper, size_t shift_bits = 0, double parallelism = 1.0) {
  size_t n = In.size();
  using in_type = typename slice<InIterator, InIterator>::value_type;
  using key_type = typename Helper::key_type;
  using val_type = typename Helper::val_type;
  using result_type = typename Helper::result_type;
  constexpr size_t hash_bits = sizeof(typename std::invoke_result<decltype(helper.hash), key_type>::type) * 8;
  if (n < PARLAY_COLLECT_REDUCE_BASE_CASE_SIZE || parallelism < .0001 || shift_bits == hash_bits) {
    return collect_reduce_serial(In, Out, helper, shift_bits);
  }
  internal::timer t;

  // 1. sampling
  auto g = [&](const in_type &kv) { return kv.first; };
  auto heavy_seq = sample_heavy_keys(In, g, helper.hash, helper.equal, shift_bits);
  size_t heavy_id_size = size_t{1} << log2_up(heavy_seq.size() * 5 + 1);
  size_t heavy_id_mask = heavy_id_size - 1;
  sequence<std::pair<size_t, size_t>> heavy_id(heavy_id_size, {ULLONG_MAX, ULLONG_MAX});
  for (auto [k, v] : heavy_seq) {
    size_t idx = helper.hash(In[k].first) >> shift_bits & heavy_id_mask;
    while (heavy_id[idx].first != ULLONG_MAX) {
      idx = (idx + 1) & heavy_id_mask;
    }
    heavy_id[idx] = {k, v};
  }
  auto lookup = [&](size_t k) {
    size_t idx = helper.hash(In[k].first) >> shift_bits & heavy_id_mask;
    while (heavy_id[idx].first != ULLONG_MAX && !helper.equal(In[heavy_id[idx].first].first, In[k].first)) {
      idx = (idx + 1) & heavy_id_mask;
    }
    return heavy_id[idx].second;
  };

  // if (parallelism == 1.0) t.next("sampling");

  // 2. count the number of light/heavy keys
  size_t LOG2_LIGHT_KEYS = std::min<size_t>(hash_bits - shift_bits, 10);
  size_t LIGHT_MASK = (1 << LOG2_LIGHT_KEYS) - 1;
  size_t light_buckets = 1 << LOG2_LIGHT_KEYS;
  size_t heavy_buckets = heavy_seq.size();
  size_t num_buckets = heavy_buckets + light_buckets;
  size_t num_blocks = 1 + n * sizeof(in_type) / (num_buckets * 5000);
  size_t block_size = (n + num_blocks - 1) / num_blocks;

  size_t m = num_blocks * light_buckets;
  size_t m2 = num_blocks * heavy_buckets;
#ifdef BREAKDOWN
  if (parallelism == 1.0) {
    printf("### heavy_buckets: %zu\n", heavy_buckets);
    printf("### light_buckets: %zu\n", light_buckets);
    printf("### num_blocks: %zu\n", num_blocks);
    printf("### m: %zu\n", m);
    printf("### m2: %zu\n", m2);
  }
#endif
  sequence<s_size_t> counts(m + 1, 0);
  sequence<val_type> block_aggregation(m2 + 1, helper.monoid.identity);
  parallel_for(
      0, num_blocks,
      [&](size_t i) {
        size_t start = i * block_size;
        size_t end = std::min(start + block_size, n);
        sequence<s_size_t> local_counts(light_buckets, 0);
        sequence<val_type> local_aggregation(heavy_buckets, helper.monoid.identity);
        for (size_t j = start; j < end; j++) {
          const auto it = lookup(j);
          size_t id;
          if (it != ULLONG_MAX) {
            // In[j] is a heavy key
            size_t id = it;
            local_aggregation[id] = helper.monoid(local_aggregation[id], helper.get_value(In[j]));
          } else {
            // In[j] is a light key
            size_t hash_v = helper.hash(In[j].first) >> shift_bits;
            id = hash_v & LIGHT_MASK;
            local_counts[id]++;
          }
        }
        for (size_t j = 0; j < light_buckets; j++) {
          counts[i * light_buckets + j] = local_counts[j];
        }
        for (size_t j = 0; j < heavy_buckets; j++) {
          block_aggregation[i * heavy_buckets + j] = local_aggregation[j];
        }
      },
      1);

  if (parallelism == 1.0) t.next("counting");

  // 3. distribute
  internal::timer t_dis;
  auto tmp = sequence<s_size_t>::uninitialized(m + 1);
  auto tmp2 = sequence<val_type>::uninitialized(m2 + 1);
  internal::transpose<uninitialized_move_tag, typename sequence<s_size_t>::iterator,
                      typename sequence<s_size_t>::iterator>(counts.begin(), tmp.begin())
      .trans(num_blocks, light_buckets);
  internal::transpose<uninitialized_move_tag, typename sequence<val_type>::iterator,
                      typename sequence<val_type>::iterator>(block_aggregation.begin(), tmp2.begin())
      .trans(num_blocks, heavy_buckets);

  // if (parallelism == 1.0) t_dis.next("component of distribute: first
  // transpose");

  scan_inplace(make_slice(tmp));
  parallel_for(0, heavy_buckets, [&](size_t i) {
    assign_dispatch(Out[i].first, In[heavy_seq[i].first].first, uninitialized_copy_tag());
    Out[i].second = reduce(tmp2.cut(i * num_blocks, (i + 1) * num_blocks), helper.monoid);
  });

  // if (parallelism == 1.0) t_dis.next("component of distribute: scan");

  internal::transpose<copy_assign_tag, typename sequence<s_size_t>::iterator, typename sequence<s_size_t>::iterator>(
      tmp.begin(), counts.begin())
      .trans(light_buckets, num_blocks);

  // if (parallelism == 1.0) t_dis.next("component of distribute: second
  // transpose");

  auto bucket_offsets = delayed_seq<size_t>(num_buckets + 1, [&](size_t i) { return tmp[i * num_blocks]; });
  size_t light_keys = bucket_offsets[light_buckets];
  auto Tmp = sequence<in_type>::uninitialized(light_keys);

  // if (parallelism == 1.0) t_dis.next("component of distribute:
  // initialization");

  parallel_for(0, num_blocks, [&](size_t i) {
    size_t start = i * block_size;
    size_t end = std::min(start + block_size, n);
    for (size_t j = start; j < end; j++) {
      const auto it = lookup(j);
      size_t id;
      if (it == ULLONG_MAX) {
        // In[j] is a light key
        size_t hash_v = helper.hash(In[j].first) >> shift_bits;
        id = hash_v & LIGHT_MASK;
        auto &pos = counts[i * light_buckets + id];
        assign_dispatch(Tmp[pos], In[j], uninitialized_copy_tag());
        pos++;
      }
    }
  });

  // if (parallelism == 1.0) t_dis.next("component of distribute: copy");

  // if (parallelism == 1.0) t.next("distribute");

  // 4. sort within each bucket
  auto output_offsets = sequence<size_t>::uninitialized(light_buckets + 1);
  output_offsets[light_buckets] = 0;
  auto Out2 = sequence<result_type>::uninitialized(light_keys);
  parallel_for(0, light_buckets, [&](size_t i) {
    size_t start = bucket_offsets[i];
    size_t end = bucket_offsets[i + 1];
    if (start != end) {
      output_offsets[i] = collect_reduce_<s_size_t>(Tmp.cut(start, end), Out2.cut(start, end), helper,
                                                    shift_bits + LOG2_LIGHT_KEYS, parallelism);
    } else {
      output_offsets[i] = 0;
    }
  });
  if (parallelism == 1.0) t.next("local sort");

  scan_inplace(make_slice(output_offsets));
  parallel_for(0, light_buckets, [&](size_t i) {
    size_t start = output_offsets[i];
    size_t end = output_offsets[i + 1];
    if (start != end) {
      parallel_for(
          0, end - start,
          [&](size_t j) {
            assign_dispatch(Out[heavy_buckets + output_offsets[i] + j], Out2[bucket_offsets[i] + j],
                            uninitialized_copy_tag());
          },
          BLOCK_SIZE);
    }
  });

  if (parallelism == 1.0) t.next("pack");
  size_t num_distinct_keys = output_offsets[light_buckets] + heavy_buckets;
  return num_distinct_keys;
}

template<typename arg_type, typename Hash, typename Equal, typename Monoid>
struct collect_reduce_helper {
  using in_type = arg_type;
  using key_type = std::tuple_element_t<0, in_type>;
  using val_type = std::tuple_element_t<1, in_type>;
  using result_type = std::pair<key_type, val_type>;
  Hash hash;
  Equal equal;
  Monoid monoid;
  collect_reduce_helper(Hash const &h, Equal const &e, Monoid const &m) : hash(h), equal(e), monoid(m){};
  template<typename T>
  static const key_type &get_key(const T &p) {
    return std::get<0>(p);
  }
  template<typename T>
  static key_type &get_key(T &p) {
    return std::get<0>(p);
  }
  static const val_type get_value(const in_type &p) { return std::get<1>(p); }
  static val_type get_value(in_type &p) { return std::get<1>(p); }
};

template<PARLAY_RANGE_TYPE R, typename Hash = parlay::hash<typename range_value_type_t<R>::first_type>,
         typename Equal = std::equal_to<>,
         typename Monoid = parlay::plus<typename std::tuple_element_t<1, range_value_type_t<R>>>>
auto collect_reduce(const R &In, const Hash &hash = {}, const Equal &equal = {}, const Monoid &monoid = {}) {
  auto helper = collect_reduce_helper<range_value_type_t<R>, Hash, Equal, Monoid>{hash, equal, monoid};
  using result_type =
      std::pair<typename range_value_type_t<R>::first_type, typename std::tuple_element_t<1, range_value_type_t<R>>>;
  auto Out = sequence<result_type>::uninitialized(In.size());
  size_t max32 = static_cast<size_t>((std::numeric_limits<uint32_t>::max)());
  if (In.size() < max32) {
    size_t num_distinct_keys = collect_reduce_<uint32_t>(In, make_slice(Out), helper);
    Out.resize(num_distinct_keys);
    return Out;
  } else {
    size_t num_distinct_keys = collect_reduce_<size_t>(In, make_slice(Out), helper);
    Out.resize(num_distinct_keys);
    return Out;
  }
}

template<typename arg_type, typename sum_type, typename Hash, typename Equal, typename Monoid>
struct histogram_helper {
  using in_type = arg_type;
  using key_type = std::tuple_element_t<0, in_type>;
  using val_type = sum_type;
  using result_type = std::pair<key_type, val_type>;
  Hash hash;
  Equal equal;
  Monoid monoid;
  histogram_helper(Hash const &h, Equal const &e, Monoid const &m) : hash(h), equal(e), monoid(m){};
  template<typename T>
  static const key_type &get_key(const T &p) {
    return std::get<0>(p);
  }
  template<typename T>
  static key_type &get_key(T &p) {
    return std::get<0>(p);
  }
  static constexpr val_type get_value(const in_type &) { return 1; }
  static constexpr val_type get_value(in_type &) { return 1; }
};

template<typename sum_type = size_t, typename Iterator, typename Hash, typename Equal,
         typename Monoid = parlay::plus<sum_type>>
auto histogram(slice<Iterator, Iterator> In, Hash hash, Equal equal, Monoid monoid = {}) {
  using result_type = std::pair<typename slice<Iterator, Iterator>::value_type::first_type, sum_type>;
  auto Out = sequence<result_type>::uninitialized(In.size());
  auto helper = histogram_helper<typename slice<Iterator, Iterator>::value_type, sum_type, Hash, Equal, Monoid>{
      hash, equal, monoid};
  size_t max32 = static_cast<size_t>((std::numeric_limits<uint32_t>::max)());
  if (In.size() < max32) {
    size_t num_distinct_keys = collect_reduce_<uint32_t>(In, make_slice(Out), helper);
    Out.resize(num_distinct_keys);
    return Out;
  } else {
    size_t num_distinct_keys = collect_reduce_<size_t>(In, make_slice(Out), helper);
    Out.resize(num_distinct_keys);
    return Out;
  }
}

}  // namespace parlay

#endif  // COLLECT_REDUCE_H_
