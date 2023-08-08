#include "semisort.h"

#include <fstream>
#include <utility>

#include "generator.h"
#include "internal/get_time.h"
#include "random.h"
using namespace std;
using namespace parlay;

size_t n = 1e9;
#ifndef NGRAM
size_t kNumTests = 4;
#else
size_t kNumTests = 2;
#endif
constexpr int NUM_ROUNDS = 1;

std::string test_name(int id) {
  switch (id) {
    case 0: return "Ours=";
    case 1: return "Ours<";
    case 3: return "Ours=-i";
    case 4: return "Ours<-i";
    default: assert(0);
  }
  return "";
}

template<class K, class V>
auto get_occurence(const sequence<pair<K, V>> &seq) {
  auto flags = delayed_seq<bool>(seq.size(), [&](size_t i) { return i == 0 || seq[i].first != seq[i - 1].first; });
  auto index = pack_index(flags);
  size_t num_keys = index.size();
  sequence<pair<size_t, K>> occur_key(num_keys);
  parallel_for(0, num_keys, [&](size_t i) {
    if (i == num_keys - 1) {
      occur_key[i] = {seq.size() - index[i], seq[index[i]].first};
    } else {
      occur_key[i] = {index[i + 1] - index[i], seq[index[i]].first};
    }
  });
  sort_inplace(occur_key);
  return occur_key;
}

template<class K, class V>
void check_correctness(const sequence<pair<K, V>> &in, int id) {
  auto out = in;
  parallel_for(0, out.size(), [&](size_t i) { out[i].second = i; });
  switch (id) {
    case 0:
      semisort_equal_inplace(
          make_slice(out), [](const pair<K, V> &kv) { return kv.first; }, [](const K &k) { return _hash(k); });
      break;
    case 1:
      semisort_less_inplace(
          make_slice(out), [](const pair<K, V> &kv) { return kv.first; }, [](const K &k) { return _hash(k); });
      break;
    case 2:
      semisort_equal_inplace(
          make_slice(out), [](const pair<K, V> &kv) { return kv.first; }, [](const K &k) { return k; });
      break;
    case 3:
      semisort_less_inplace(
          make_slice(out), [](const pair<K, V> &kv) { return kv.first; }, [](const K &k) { return k; });
      break;
    default: assert(0);
  }

  size_t SIZE = in.size() * 1.2;
  constexpr uint32_t MAX_VAL = numeric_limits<uint32_t>::max();
  sequence<size_t> vis(SIZE, MAX_VAL);
  parallel_for(0, in.size(), [&](size_t i) {
    if (i == 0 || out[i].first != out[i - 1].first) {
      uint32_t v = _hash(out[i].first) % SIZE;
      while (true) {
        if (vis[v] != MAX_VAL) {
          if (out[vis[v]].first == out[i].first) {
            fprintf(stderr, "Error: not semisorted\n");
            exit(EXIT_FAILURE);
          } else {
            v = (v + 1) % SIZE;
          }
        } else {
          if (__sync_bool_compare_and_swap(&vis[v], MAX_VAL, i)) {
            break;
          } else {
            v = (v + 1) % SIZE;
          }
        }
      }
    }
  });
  parallel_for(1, out.size(), [&](size_t i) {
    if (out[i - 1].first == out[i].first) {
      assert(out[i - 1].second < out[i].second);
    }
  });
  auto out2 = sort(make_slice(in));
  assert(get_occurence(out) == get_occurence(out2));
  printf("Pass\n");
}

template<typename K, typename V>
double test(const sequence<pair<K, V>> &in, int id) {
  std::cout << "test_name: " << test_name(id) << std::endl;
  double total_time = 0;
  for (int i = 0; i <= NUM_ROUNDS; i++) {
    auto out = in;
    internal::timer t;
    switch (id) {
      case 0:
        semisort_equal_inplace(
            make_slice(out), [](const pair<K, V> &kv) { return kv.first; }, [](const K &k) { return _hash(k); });
        break;
      case 1:
        semisort_less_inplace(
            make_slice(out), [](const pair<K, V> &kv) { return kv.first; }, [](const K &k) { return _hash(k); });
        break;
      case 2:
        semisort_equal_inplace(
            make_slice(out), [](const pair<K, V> &kv) { return kv.first; }, [](const K &k) { return k; });
        break;
      case 3:
        semisort_less_inplace(
            make_slice(out), [](const pair<K, V> &kv) { return kv.first; }, [](const K &k) { return k; });
        break;
      default: assert(0);
    }
    t.stop();
    if (i == 0) {
      printf("Warmup round: %f\n", t.total_time());
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
void run_all(const sequence<T> &seq, int id = -1) {
  // get_distribution(seq);
  vector<double> times;
  if (id == -1) {
    for (size_t i = 0; i < kNumTests; i++) {
      // times.push_back(test(seq, i));
      check_correctness(seq, i);
      printf("\n");
    }
  } else {
    // times.push_back(test(seq, id));
    check_correctness(seq, id);
    printf("\n");
  }
  ofstream ofs("Ours.tsv", ios::app);
  for (auto t : times) {
    ofs << t << '\t';
  }
  ofs << '\n';
  ofs.close();
}

template<class T>
void run_all_dist(int id = -1) {
  // uniform distribution
  vector<size_t> num_keys{1000000000, 10000000, 100000, 1000, 10};
  for (auto v : num_keys) {
    auto seq = uniform_pairs_generator<T>(v);
    run_all(seq, id);
  }

  // exponential distribution
  vector<double> lambda{0.00001, 0.00002, 0.00005, 0.00007, 0.0001};
  for (auto v : lambda) {
    auto seq = exponential_pairs_generator<T>(v);
    run_all(seq, id);
  }

  // zipfian distribution
  vector<double> s{0.6, 0.8, 1, 1.2, 1.5};
  for (auto v : s) {
    auto seq = zipfian_pairs_generator<T>(v);
    run_all(seq, id);
  }

  // bits exp distribution
  vector<size_t> rate{10, 30, 50, 100, 300};
  for (auto v : rate) {
    auto seq = bits_exp_pairs_generator<T>(v);
    run_all(seq, id);
  }
}

template<class T>
void run_scaling(int id = -1) {
  // uniform distribution
  vector<size_t> num_keys{1000, 10000000};
  for (auto v : num_keys) {
    auto seq = uniform_pairs_generator<T>(v);
    run_all(seq, id);
  }

  // exponential distribution
  vector<double> lambda{0.00007, 0.00002};
  for (auto v : lambda) {
    auto seq = exponential_pairs_generator<T>(v);
    run_all(seq, id);
  }

  // zipfian distribution
  vector<double> s{1.2, 0.8};
  for (auto v : s) {
    auto seq = zipfian_pairs_generator<T>(v);
    run_all(seq, id);
  }
}

template<class T>
void run_all_size(int id = -1) {
  vector<size_t> size{10000000, 20000000, 50000000, 100000000, 200000000, 500000000, 1000000000};
  for (auto input_size : size) {
    n = input_size;
    // uniform distribution
    vector<size_t> num_keys{1000, 10000000};
    for (auto v : num_keys) {
      auto seq = uniform_pairs_generator<T>(v);
      run_all(seq, id);
    }

    // exponential distribution
    vector<double> lambda{0.00007, 0.00002};
    for (auto v : lambda) {
      auto seq = exponential_pairs_generator<T>(v);
      run_all(seq, id);
    }

    // zipfian distribution
    vector<double> s{1.2, 0.8};
    for (auto v : s) {
      auto seq = zipfian_pairs_generator<T>(v);
      run_all(seq, id);
    }
  }
}

void run_all_graphs(int id = -1) {
  string path = "/data0/graphs/links/";
  vector<string> graphs = {"soc-LiveJournal1.bin", "twitter.bin", "Cosmo50_5.bin", "sd_arc.bin"};
  for (auto g : graphs) {
    printf("%s\n", g.c_str());
    auto seq = read_graph(path + g);
    run_all(seq, id);
  }
}

#ifdef NGRAM
void run_all_ngrams(int id = -1) {
  string path = "/data0/xdong038/";
  vector<string> ngrams = {"2gram.txt", "3gram.txt"};
  for (auto ng : ngrams) {
    auto seq = read_ngram(path + ng);
    run_all(seq, id);
  }
}
#endif

int main(int argc, char *argv[]) {
  if (argc >= 2) {
    n = atoll(argv[1]);
  }
  printf("n: %zu\n", n);

  int id = -1;
  run_all_dist<uint32_t>(id);
  run_all_dist<uint64_t>(id);
  // run_all_dist<__uint128_t>(id);

  // int id = -1;
  // run_all_graphs(id);

  // int id = -1;
  // run_all_ngrams(id);

  // int id = -1;
  // run_scaling<uint64_t>(id);

  // int id = -1;
  // run_all_size<uint64_t>(id);

  // using T = __uint32_t;
  // int id = 0;
  // auto seq = uniform_strings_generator<T>(1000000000);
  // auto seq = exponential_pairs_generator<T>(0.00001);
  // auto seq = bits_exp_strings_generator<T>(10);
  // run_all(seq, id);

  return 0;
}
