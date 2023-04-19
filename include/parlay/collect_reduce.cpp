#include "collect_reduce.h"

#include <fstream>
#include <utility>

#include "generator.h"
#include "internal/get_time.h"
#include "random.h"
using namespace std;
using namespace parlay;

size_t n = 1e9;
size_t kNumTests = 2;
constexpr int NUM_ROUNDS = 3;

std::string test_name(int id) {
  switch (id) {
    case 0: return "Ours";
    case 1: return "PLCR";
    default: assert(0);
  }
  return "";
}

template<class K, class V>
void check_correctness(const sequence<pair<K, V>> &in) {
  auto out = collect_reduce(make_slice(in), [](const K &k) { return _hash(k); });
  auto out2 = parlay_collect_reduce(make_slice(in), [](const K &k) { return _hash(k); });
  sort_inplace(out);
  sort_inplace(out2);
  if (out.size() != out2.size()) {
    printf("out.size()=%zu, out2.size()=%zu\n", out.size(), out2.size());
  }
  assert(out.size() == out2.size());
  // for (size_t i = 0; i < out.size(); i++) {
  // printf("(%u,%lu) (%u,%lu)\n", out[i].first, out[i].second, out2[i].first,
  // out2[i].second);
  //}
  parallel_for(0, out.size(), [&](size_t i) {
    if (out[i].first != out2[i].first || out[i].second != out2[i].second) {
      // printf("(%u,%u) (%u,%u)\n", out[i].first, out[i].second, out2[i].first,
      // out2[i].second); fflush(stdout);
    }
    assert(out[i].first == out2[i].first);
    assert(out[i].second == out2[i].second);
  });
  printf("Pass\n");
}

template<typename arg_type, typename Hash, typename Equal, typename Monoid>
struct collect_reduce_helper2 {
  using in_type = arg_type;
  using key_type = std::tuple_element_t<0, in_type>;
  using val_type = std::tuple_element_t<1, in_type>;
  using result_type = std::pair<key_type, val_type>;
  Hash hash;
  Equal equal;
  Monoid monoid;
  collect_reduce_helper2(Hash const &h, Equal const &e, Monoid const &m) : hash(h), equal(e), monoid(m){};
  template<typename T>
  static const key_type &get_key(const T &p) {
    return std::get<0>(p);
  }
  template<typename T>
  static key_type &get_key(T &p) {
    return std::get<0>(p);
  }
  void init(result_type &p, in_type const &kv) const { assign_uninitialized(std::get<1>(p), std::get<1>(kv)); }
  void update(result_type &p, in_type const &kv) const { std::get<1>(p) = monoid(std::get<1>(p), std::get<1>(kv)); }
  static void destruct_val(in_type &) {}
  template<typename Range>
  result_type reduce(Range &S) const {
    auto &key = std::get<0>(S[0]);
    auto sum = internal::reduce(internal::delayed_map(S, [&](in_type const &kv) { return std::get<1>(kv); }), monoid);
    return result_type(key, sum);
  }
};

template<typename R, typename Hash = parlay::hash<typename range_value_type_t<R>::first_type>,
         typename Equal = std::equal_to<>,
         typename Monoid = parlay::plus<std::tuple_element_t<1, range_value_type_t<R>>>>
auto parlay_collect_reduce(R &&A, const Hash &hash = {}, const Equal &equal = {}, const Monoid &monoid = {}) {
  static_assert(is_random_access_range_v<R>);
  auto helper = collect_reduce_helper2<range_value_type_t<R>, Hash, Equal, Monoid>{hash, equal, monoid};
  return internal::collect_reduce_sparse(std::forward<R>(A), helper);
}

template<typename T>
double test(const sequence<pair<T, T>> &seq, int id) {
  std::cout << "test_name: " << test_name(id) << std::endl;
  double total_time = 0;
  for (int i = 0; i <= NUM_ROUNDS; i++) {
    internal::timer t;
    switch (id) {
      case 0: collect_reduce(make_slice(seq), [](const T &k) { return _hash(k); }); break;
      case 1: parlay_collect_reduce(make_slice(seq), [](const T &k) { return _hash(k); }); break;
      default: assert(0);
    }
    t.stop();
    if (i == 0) {
      printf("Warmup: %f\n", t.total_time());
    } else {
      printf("Round %d: %f\n", i, t.total_time());
      total_time += t.total_time();
    }
  }
  double avg = total_time / NUM_ROUNDS;
  printf("Average: %f\n", avg);
  return avg;
}

template<class T>
void run_all(const sequence<pair<T, T>> &seq, int id = -1) {
  vector<double> times;
  if (id == -1) {
    for (size_t i = 0; i < kNumTests; i++) {
      times.push_back(test(seq, i));
      check_correctness(seq);
      printf("\n");
    }
  } else {
    times.push_back(test(seq, id));
    check_correctness(seq);
    printf("\n");
  }
  ofstream ofs("collect-reduce.tsv", ios::app);
  for (auto t : times) {
    ofs << t << '\t';
  }
  ofs << '\n';
  ofs.close();
}

template<class T>
void run_all_dist(int id = -1) {
  // uniform distribution
  vector<size_t> num_keys{10, 1000, 100000, 10000000, 1000000000};
  for (auto v : num_keys) {
    auto seq = uniform_generator<T>(v);
    run_all(seq, id);
  }

  // exponential distribution
  vector<double> lambda{0.0001, 0.00007, 0.00005, 0.00002, 0.00001};
  for (auto v : lambda) {
    auto seq = exponential_generator<T>(v);
    run_all(seq, id);
  }

  // zipfian distribution
  vector<double> s{1.5, 1.2, 1, 0.8, 0.6};
  for (auto v : s) {
    auto seq = zipfian_generator<T>(v);
    run_all(seq, id);
  }
}

int main(int argc, char *argv[]) {
  if (argc >= 2) {
    n = atoll(argv[1]);
  }
  printf("n: %zu\n", n);

  int id = -1;
  run_all_dist<uint64_t>(id);

  return 0;
}
