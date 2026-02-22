#include <bits/stdc++.h>
using namespace std;

struct Edge {
  int to;
  long long reward;
  Edge(int t = 0, long long r = 0) : to(t), reward(r) {}
};

struct Game {
  int n;
  // owner[v] = 0 (Max) or 1 (Min)
  vector<int> owner;
  // adjacency list
  vector<vector<Edge>> out;

  Game(int n_ = 0) { init(n_); }
  void init(int n_) {
    n = n_;
    owner.assign(n, 0);
    out.assign(n, {});
  }
  void set_owner(int v, int o) { owner[v] = o; }
  void add_edge(int u, int v, long long reward) {
    out[u].emplace_back(v, reward);
  }
  void clear() {
    owner.assign(n, 0);
    for (auto &vec : out) vec.clear();
  }
};

Game build_param_game(int N, int p, int q) {
  Game G;
  G.init(0);

  unordered_map<long long, int> base_id;
  unordered_map<long long, int> stpmin_id;
  unordered_map<long long, int> ltptop_id;

  auto make_base_key = [&](int d, int w, int mode) {
    return ((long long)d * (N + 1) + w) * 2 + mode;
  };
  auto make_stpmin_key = [&](int d, int w, int y) {
    return ((long long)d * (N + 1) + w) * (N + 1) + y;
  };
  auto make_ltptop_key = [&](int d, int w) {
    return (long long)d * (N + 1) + w;
  };

  auto add_node = [&](int owner) -> int {
    int id = G.n;
    G.n += 1;
    G.owner.push_back(owner);
    G.out.emplace_back();
    return id;
  };

  for (int d = 0; d <= N; ++d) {
    for (int w = 0; w <= d; ++w) {
      long long k0 = make_base_key(d, w, 0);
      int id0 = add_node(0);
      base_id[k0] = id0;
      long long k1 = make_base_key(d, w, 1);
      int id1 = add_node(0);
      base_id[k1] = id1;
    }
  }

  for (int d = 0; d <= N; ++d) {
    for (int w = 0; w <= d; ++w) {
      for (int y = 0; y <= N - w; ++y) {
        long long k = make_stpmin_key(d, w, y);
        int id = add_node(1);
        stpmin_id[k] = id;
      }
    }
  }

  for (int d = 0; d <= N; ++d) {
    for (int w = 0; w <= d; ++w) {
      long long k = make_ltptop_key(d, w);
      int id = add_node(1);
      ltptop_id[k] = id;
    }
  }

  int LTP_bot_node = add_node(1);

  long long key_11_stp = make_base_key(N, N, 0);
  long long key_11_ltp = make_base_key(N, N, 1);
  int node_11_stp = base_id.at(key_11_stp);
  int node_11_ltp = base_id.at(key_11_ltp);

  long long reward_int_bot = (long long)(p - 2 * q) * (long long)N;
  G.add_edge(LTP_bot_node, node_11_stp, reward_int_bot);
  G.add_edge(LTP_bot_node, node_11_ltp, reward_int_bot);

  for (int d = 0; d <= N; ++d) {
    for (int w = 0; w <= d; ++w) {
      int base_stp = base_id[make_base_key(d, w, 0)];
      for (int y = 0; y <= N - w; ++y) {
        int min_node = stpmin_id[make_stpmin_key(d, w, y)];
        G.add_edge(base_stp, min_node, 0);
      }
    }
  }

  for (int d = 0; d <= N; ++d) {
    for (int w = 0; w <= d; ++w) {
      for (int y = 0; y <= N - w; ++y) {
        int min_node = stpmin_id[make_stpmin_key(d, w, y)];

        if (y <= N - d) {
          int d2 = N - d - y;
          int w2 = d2;
          int dest_stp = base_id[make_base_key(d2, w2, 0)];
          int dest_ltp = base_id[make_base_key(d2, w2, 1)];
          long long opt_num = (long long)N + w - d + y;
          long long alg_num = (long long)2 * N - d + y;
          long long reward_int = (long long)p * opt_num - (long long)q * alg_num;
          G.add_edge(min_node, dest_stp, reward_int);
          G.add_edge(min_node, dest_ltp, reward_int);
        } else {
          int dest_stp = base_id[make_base_key(0, 0, 0)];
          int dest_ltp = base_id[make_base_key(0, 0, 1)];
          long long opt_num = (long long)w + 2LL * y;
          long long alg_num = (long long)N + 2LL * y;
          long long reward_int = (long long)p * opt_num - (long long)q * alg_num;
          G.add_edge(min_node, dest_stp, reward_int);
          G.add_edge(min_node, dest_ltp, reward_int);
        }

        for (int x = 1; x <= y; ++x) {
          int d2 = d + x;
          if (d2 > N) d2 = N;
          int w2 = w + x;
          int dest_ltp = base_id[make_base_key(d2, w2, 1)];
          long long reward_int = (long long)(p - 2LL * q) * (long long)x;
          G.add_edge(min_node, dest_ltp, reward_int);
        }
      }
    }
  }

  for (int d = 0; d <= N; ++d) {
    for (int w = 0; w <= d; ++w) {
      int base_ltp = base_id[make_base_key(d, w, 1)];
      int top_min = ltptop_id[make_ltptop_key(d, w)];
      G.add_edge(base_ltp, top_min, 0);
      G.add_edge(base_ltp, LTP_bot_node, 0);
    }
  }

  for (int d = 0; d <= N; ++d) {
    for (int w = 0; w <= d; ++w) {
      int top_min = ltptop_id[make_ltptop_key(d, w)];
      for (int x = 0; x <= d; ++x) {
        int d2 = N - d + x;
        d2 = max(0, min(N, d2));
        int w2 = min(N - w, d2);

        int dest_stp = base_id[make_base_key(d2, w2, 0)];
        int dest_ltp = base_id[make_base_key(d2, w2, 1)];
        long long opt_num = (long long)N - d + w + x;
        long long alg_num = (long long)2 * N - d + x;
        long long reward_int = (long long)p * opt_num - (long long)q * alg_num;
        G.add_edge(top_min, dest_stp, reward_int);
        G.add_edge(top_min, dest_ltp, reward_int);
      }
    }
  }

  return G;
}

// ============================================================
// Mean-payoff strategy iteration (ergodic assumption)
// ============================================================

struct PolicyResult {
  long double value;              // optimal gain g
  vector<int> sigma_max_edge_idx; // for Max states: chosen outgoing edge index, else -1
  vector<int> tau_min_edge_idx;   // for Min states: chosen outgoing edge index, else -1
};

// Build the "available edge list" for the induced 1-player game when Max is fixed to sigma.
// At Max nodes: only sigma edge is available.
// At Min nodes: all edges are available.
struct AvEdge {
  int u, v;
  long long r;
};

static vector<AvEdge> build_available_edges(const Game &G, const vector<int> &sigma) {
  vector<AvEdge> edges;
  edges.reserve(100000);

  for (int u = 0; u < G.n; ++u) {
    if (G.owner[u] == 0) {
      int ei = sigma[u];
      if (ei < 0 || ei >= (int)G.out[u].size()) continue;
      const auto &e = G.out[u][ei];
      edges.push_back({u, e.to, e.reward});
    } else {
      for (auto &e : G.out[u]) edges.push_back({u, e.to, e.reward});
    }
  }
  return edges;
}

// Karp's algorithm for minimum mean cycle in a directed graph (with all nodes present).
// Returns min cycle mean as long double.
static long double karp_min_mean_cycle(int n, const vector<AvEdge> &edges) {
  const long double INF = 1e300L;

  vector<vector<pair<int,long long>>> in(n);
  in.assign(n, {});
  in.shrink_to_fit();
  in.assign(n, {});
  for (auto &ed : edges) {
    in[ed.v].push_back({ed.u, ed.r});
  }

  // dp[k][v] = min weight of walk of length k ending at v
  vector<vector<long double>> dp(n + 1, vector<long double>(n, INF));
  for (int v = 0; v < n; ++v) dp[0][v] = 0.0L;

  for (int k = 1; k <= n; ++k) {
    for (int v = 0; v < n; ++v) {
      long double best = INF;
      for (auto &[u, w] : in[v]) {
        best = min(best, dp[k - 1][u] + (long double)w);
      }
      dp[k][v] = best;
    }
  }

  long double ans = INF;
  for (int v = 0; v < n; ++v) {
    if (dp[n][v] >= INF/2) continue;
    long double mx = -INF;
    for (int k = 0; k <= n - 1; ++k) {
      if (dp[k][v] >= INF/2) continue;
      long double denom = (long double)(n - k);
      long double val = (dp[n][v] - dp[k][v]) / denom;
      mx = max(mx, val);
    }
    ans = min(ans, mx);
  }
  return ans;
}

// Compute a bias/potential h for the induced game (Max fixed), given gain g.
// We compute shortest path cost-to-reference in graph with weights w' = r - g.
// Since g is min cycle mean, the transformed graph has no negative cycles (for the available edges).
static vector<long double> compute_bias_shortest_to_ref(
    const Game &G, const vector<int> &sigma, long double g, int ref = 0) {

  const long double INF = 1e300L;
  vector<long double> dist(G.n, INF);
  dist[ref] = 0.0L;

  // Bellman-Ford style relaxation for distances-to-ref:
  // dist[u] = min over (u->v) ( (r-g) + dist[v] )
  // Implemented by scanning available edges u->v each iteration.
  auto edges = build_available_edges(G, sigma);

  for (int it = 0; it < G.n - 1; ++it) {
    bool changed = false;
    for (auto &ed : edges) {
      int u = ed.u, v = ed.v;
      if (dist[v] >= INF/2) continue;
      long double wprime = (long double)ed.r - g;
      long double cand = wprime + dist[v];
      if (cand + 1e-18L < dist[u]) {
        dist[u] = cand;
        changed = true;
      }
    }
    if (!changed) break;
  }

  // Normalize to avoid huge numbers: shift so h[ref]=0
  long double shift = dist[ref];
  if (shift >= INF/2) shift = 0.0L;
  for (auto &x : dist) {
    if (x < INF/2) x -= shift;
  }
  return dist;
}

// Given (g,h) for the induced 1-player game (Max fixed),
// pick Min's greedy optimal policy w.r.t. the Bellman min equation:
// h(u) = min_e ( (r-g) + h(v) )
static vector<int> compute_min_best_response(
    const Game &G, const vector<int> &sigma, long double g, const vector<long double> &h) {

  vector<int> tau(G.n, -1);
  for (int u = 0; u < G.n; ++u) {
    if (G.owner[u] != 1) continue;
    int best_ei = 0;
    long double best = 1e300L;
    for (int ei = 0; ei < (int)G.out[u].size(); ++ei) {
      const auto &e = G.out[u][ei];
      long double cand = ((long double)e.reward - g) + h[e.to];
      if (cand + 1e-18L < best) {
        best = cand;
        best_ei = ei;
      }
    }
    tau[u] = best_ei;
  }
  return tau;
}

// Improve Max policy using current evaluation (g,h):
// For each Max state u, switch to edge maximizing (r-g + h[to]).
// If that improves strictly over current chosen edge, update.
static bool improve_max_policy(
    const Game &G, vector<int> &sigma, long double g, const vector<long double> &h) {

  bool improved = false;
  for (int u = 0; u < G.n; ++u) {
    if (G.owner[u] != 0) continue;

    int cur = sigma[u];
    if (cur < 0) cur = 0;

    // Current score
    long double curScore = -1e300L;
    if (!G.out[u].empty()) {
      const auto &ec = G.out[u][cur];
      curScore = ((long double)ec.reward - g) + h[ec.to];
    }

    int best = cur;
    long double bestScore = curScore;

    for (int ei = 0; ei < (int)G.out[u].size(); ++ei) {
      const auto &e = G.out[u][ei];
      long double s = ((long double)e.reward - g) + h[e.to];
      if (s > bestScore + 1e-18L) {
        bestScore = s;
        best = ei;
      }
    }

    if (best != cur) {
      sigma[u] = best;
      improved = true;
    }
  }
  return improved;
}

// One full evaluation of a fixed Max strategy sigma:
// - compute induced min mean cycle g
// - compute bias h
// - compute a greedy Min best-response tau
static tuple<long double, vector<long double>, vector<int>>
evaluate_fixed_max(const Game &G, const vector<int> &sigma, int ref = 0) {
  auto edges = build_available_edges(G, sigma);
  long double g = karp_min_mean_cycle(G.n, edges);
  auto h = compute_bias_shortest_to_ref(G, sigma, g, ref);
  auto tau = compute_min_best_response(G, sigma, g, h);
  return {g, h, tau};
}

// Strategy iteration (Hoffman–Karp style) for ergodic games.
PolicyResult mean_payoff_strategy_iteration(const Game &G, int max_iters = 100000) {
  PolicyResult res;
  res.sigma_max_edge_idx.assign(G.n, -1);
  res.tau_min_edge_idx.assign(G.n, -1);

  // Initial Max strategy: pick first outgoing edge for Max states.
  vector<int> sigma(G.n, -1);
  for (int u = 0; u < G.n; ++u) {
    if (G.owner[u] == 0) sigma[u] = 0;
  }

  long double g = 0.0L;
  vector<long double> h;
  vector<int> tau;

  for (int it = 0; it < max_iters; ++it) {
    tie(g, h, tau) = evaluate_fixed_max(G, sigma, /*ref=*/0);

    bool improved = improve_max_policy(G, sigma, g, h);
    if (!improved) {
      // Converged to optimal sigma. Recompute final tau against it.
      tie(g, h, tau) = evaluate_fixed_max(G, sigma, /*ref=*/0);
      break;
    }
  }

  res.value = g;
  res.sigma_max_edge_idx = sigma;
  res.tau_min_edge_idx = tau;

  // Clean non-owned entries to -1 for readability.
  for (int u = 0; u < G.n; ++u) {
    if (G.owner[u] != 0) res.sigma_max_edge_idx[u] = -1;
    if (G.owner[u] != 1) res.tau_min_edge_idx[u] = -1;
  }
  return res;
}

// Pretty-print a policy as "state -> (to,reward)" for owned states.
static void print_policy(const Game &G, const vector<int> &pol, int ownerFlag, const string &name) {
  cout << name << " policy (" << (ownerFlag==0 ? "Max" : "Min") << " states only):\n";
  for (int u = 0; u < G.n; ++u) {
    if (G.owner[u] != ownerFlag) continue;
    int ei = pol[u];
    if (ei < 0 || ei >= (int)G.out[u].size()) continue;
    const auto &e = G.out[u][ei];
    cout << "  state " << u << " -> edge#" << ei
         << " (to " << e.to << ", reward " << e.reward << ")\n";
  }
}

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  // Example usage:
  // Read p q from stdin and build the parameterized game.
  int N = 20;
  int p, q;
  if (!(cin >> p >> q)) {
    cerr << "Usage: provide p q on stdin.\n";
    return 1;
  }

  Game G = build_param_game(N, p, q);

  auto sol = mean_payoff_strategy_iteration(G);

  cout.setf(std::ios::fixed);
  cout << setprecision(15);

  print_policy(G, sol.sigma_max_edge_idx, 0, "Optimal");
  cout << "\n";
  print_policy(G, sol.tau_min_edge_idx, 1, "Optimal");

  cout << "Game value (optimal mean payoff gain g) = " << (double)sol.value << "\n\n";

  return 0;
}
