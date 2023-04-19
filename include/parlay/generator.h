#include <fcntl.h>
#include <malloc.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <utility>
using namespace std;

extern size_t n;
constexpr int base = 1e9 + 7;

uint32_t _hash(const string &s) {
  uint32_t ret = 0;
  for (auto c : s) {
    ret = ret * base + (islower(c) ? c - 'a' : 26);
  }
  return ret;
}

__uint128_t _hash(__uint128_t v) {
  __uint128_t w = parlay::hash64(2 * v);
  w = w << 64 | parlay::hash64(2 * v + 1);
  return w;
}

uint64_t _hash(uint64_t v) {
  return parlay::hash64(v);
}

uint32_t _hash(uint32_t v) {
  return parlay::hash32(v);
}

uint16_t _hash(uint16_t v) {
  return parlay::hash32(v);
}

uint8_t _hash(uint8_t v) {
  return parlay::hash32(v);
}

template<class T>
parlay::sequence<pair<T, T>> uniform_generator(size_t num_keys) {
  printf("uniform distribution with num_keys: %zu\n", num_keys);
  parlay::sequence<pair<T, T>> seq(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    size_t v = i % num_keys;
    seq[i] = {_hash(static_cast<T>(v)), _hash(static_cast<T>(i + n))};
    // seq[i] = {static_cast<T>(v), _hash(static_cast<T>(i + n))};
  });
  return random_shuffle(seq);
}

template<class T>
parlay::sequence<pair<T, T>> exponential_generator(double lambda) {
  printf("exponential distribution with lambda: %.10f\n", lambda);
  size_t cutoff = n;
  parlay::sequence<size_t> nums(cutoff + 1, 0);

  parlay::parallel_for(0, cutoff, [&](size_t i) { nums[i] = max(1.0, n * (lambda * exp(-lambda * (i + 0.5)))); });
  size_t tot = scan_inplace(make_slice(nums));
  assert(tot >= n);

  parlay::sequence<pair<T, T>> seq(n);
  parlay::parallel_for(0, cutoff, [&](size_t i) {
    parlay::parallel_for(nums[i], min(n, nums[i + 1]), [&](size_t j) {
      seq[j] = {_hash(static_cast<T>(i)), _hash(static_cast<T>(j + n))};
      // seq[j] = {static_cast<T>(i), _hash(static_cast<T>(j + n))};
    });
  });
  return random_shuffle(seq);
}

template<class T>
parlay::sequence<pair<T, T>> zipfian_generator(double s) {
  printf("zipfian distribution with s: %f\n", s);
  size_t cutoff = n;
  auto harmonic = parlay::delayed_seq<double>(cutoff, [&](size_t i) { return 1.0 / pow(i + 1, s); });
  double sum = parlay::reduce(make_slice(harmonic));
  double v = n / sum;
  parlay::sequence<size_t> nums(cutoff + 1, 0);
  parlay::parallel_for(0, cutoff, [&](size_t i) { nums[i] = max(1.0, v / pow(i + 1, s)); });
  size_t tot = scan_inplace(make_slice(nums));
  assert(tot >= n);
  parlay::sequence<pair<T, T>> seq(n);
  parlay::parallel_for(0, cutoff, [&](size_t i) {
    parlay::parallel_for(nums[i], min(n, nums[i + 1]), [&](size_t j) {
      seq[j] = {_hash(static_cast<T>(i)), _hash(static_cast<T>(j + n))};
      // seq[j] = {static_cast<T>(i), _hash(static_cast<T>(j + n))};
    });
  });
  return random_shuffle(seq);
}

template<class T>
void get_distribution(parlay::sequence<pair<T, T>> seq) {
  parlay::sort_inplace(make_slice(seq));
  auto flags =
      parlay::delayed_seq<bool>(seq.size(), [&](size_t i) { return i == 0 || seq[i].first != seq[i - 1].first; });
  auto index = parlay::pack_index<size_t>(flags);
  size_t num_keys = index.size();
  parlay::sequence<size_t> nums(num_keys);
  parlay::parallel_for(0, num_keys, [&](size_t i) {
    if (i == num_keys - 1) {
      nums[i] = seq.size() - index[i];
    } else {
      nums[i] = index[i + 1] - index[i];
    }
  });
  size_t max_occur = parlay::reduce(make_slice(nums), parlay::maxm<size_t>());
  printf("# num of distinct keys: %zu\n", num_keys);
  printf("# num of max occurrences: %zu\n", max_occur);
}

auto read_graph(const string filename) {
  ifstream ifs(filename);
  if (!ifs.is_open()) {
    cerr << "Error: Cannot open file " << filename << '\n';
    abort();
  }
  size_t n, m, sizes;
  ifs.read(reinterpret_cast<char *>(&n), sizeof(size_t));
  ifs.read(reinterpret_cast<char *>(&m), sizeof(size_t));
  ifs.read(reinterpret_cast<char *>(&sizes), sizeof(size_t));
  assert(sizes == (n + 1) * 8 + m * 4 + 3 * 8);

  auto offset = parlay::sequence<uint64_t>(n + 1);
  ifs.read(reinterpret_cast<char *>(offset.begin()), sizeof(uint64_t) * (n + 1));
  auto csr = parlay::sequence<uint32_t>(m);
  ifs.read(reinterpret_cast<char *>(csr.begin()), sizeof(uint32_t) * m);
  auto edges = parlay::sequence<pair<uint32_t, uint32_t>>::uninitialized(m);
  parlay::parallel_for(0, n, [&](uint32_t i) {
    parlay::parallel_for(offset[i], offset[i + 1], [&](size_t j) { edges[j] = make_pair(csr[j], i); });
  });
  if (ifs.peek() != EOF) {
    cerr << "Error: Bad data\n";
    abort();
  }
  ifs.close();
  return edges;
}

auto read_ngram(const string filename) {
  ifstream ifs(filename);
  if (!ifs.is_open()) {
    cerr << "Error: Cannot open file " << filename << '\n';
    abort();
  }
  size_t n;
  ifs >> n;
  parlay::sequence<pair<string, string>> pre_sub(n);
  for (size_t i = 0; i < n; i++) {
    size_t v;
    ifs >> v >> pre_sub[i].first >> pre_sub[i].second;
  }
  ifs.close();
  return random_shuffle(pre_sub);
}

template<class T>
void write_to_file(const parlay::sequence<pair<T, T>> &seq) {
  ofstream ofs("sequence.bin");
  ofs.write(reinterpret_cast<const char *>(seq.begin()), sizeof(pair<T, T>) * n);
  ofs.close();
}

template<class T>
parlay::sequence<pair<T,T>> read_from_file() {
  ifstream ifs("sequence.bin");
  parlay::sequence<pair<T, T>> seq(n);
  ifs.read(reinterpret_cast<char *>(seq.begin()), sizeof(pair<T, T>) * n);
  ifs.close();
  return seq;
}
