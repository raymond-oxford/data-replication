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
    for (auto &vec : out)
      vec.clear();
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
          long long reward_int =
              (long long)p * opt_num - (long long)q * alg_num;
          G.add_edge(min_node, dest_stp, reward_int);
          G.add_edge(min_node, dest_ltp, reward_int);
        } else {
          int dest_stp = base_id[make_base_key(0, 0, 0)];
          int dest_ltp = base_id[make_base_key(0, 0, 1)];
          long long opt_num = (long long)w + 2LL * y;
          long long alg_num = (long long)N + 2LL * y;
          long long reward_int =
              (long long)p * opt_num - (long long)q * alg_num;
          G.add_edge(min_node, dest_stp, reward_int);
          G.add_edge(min_node, dest_ltp, reward_int);
        }

        for (int x = 1; x <= y; ++x) {
          int d2 = d + x;
          if (d2 > N)
            d2 = N;
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
  long double value; // optimal gain g
  vector<int>
      sigma_max_edge_idx; // for Max states: chosen outgoing edge index, else -1
  vector<int>
      tau_min_edge_idx; // for Min states: chosen outgoing edge index, else -1
};

// Build the "available edge list" for the induced 1-player game when Max is
// fixed to sigma. At Max nodes: only sigma edge is available. At Min nodes: all
// edges are available.
struct AvEdge {
  int u, v;
  long long r;
};

static vector<AvEdge> build_available_edges(const Game &G,
                                            const vector<int> &sigma) {
  vector<AvEdge> edges;
  edges.reserve(100000);

  for (int u = 0; u < G.n; ++u) {
    if (G.owner[u] == 0) {
      int ei = sigma[u];
      if (ei < 0 || ei >= (int)G.out[u].size())
        continue;
      const auto &e = G.out[u][ei];
      edges.push_back({u, e.to, e.reward});
    } else {
      for (auto &e : G.out[u])
        edges.push_back({u, e.to, e.reward});
    }
  }
  return edges;
}

// Karp's algorithm for minimum mean cycle in a directed graph (with all nodes
// present). Returns min cycle mean as long double.
static long double karp_min_mean_cycle(int n, const vector<AvEdge> &edges) {
  const long double INF = 1e300L;

  vector<vector<pair<int, long long>>> in(n);
  in.assign(n, {});
  in.shrink_to_fit();
  in.assign(n, {});
  for (auto &ed : edges) {
    in[ed.v].push_back({ed.u, ed.r});
  }

  // dp[k][v] = min weight of walk of length k ending at v
  vector<vector<long double>> dp(n + 1, vector<long double>(n, INF));
  for (int v = 0; v < n; ++v)
    dp[0][v] = 0.0L;

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
    if (dp[n][v] >= INF / 2)
      continue;
    long double mx = -INF;
    for (int k = 0; k <= n - 1; ++k) {
      if (dp[k][v] >= INF / 2)
        continue;
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
// Since g is min cycle mean, the transformed graph has no negative cycles (for
// the available edges).
static vector<long double>
compute_bias_shortest_to_ref(const Game &G, const vector<int> &sigma,
                             long double g, int ref = 0) {

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
      if (dist[v] >= INF / 2)
        continue;
      long double wprime = (long double)ed.r - g;
      long double cand = wprime + dist[v];
      if (cand + 1e-18L < dist[u]) {
        dist[u] = cand;
        changed = true;
      }
    }
    if (!changed)
      break;
  }

  // Normalize to avoid huge numbers: shift so h[ref]=0
  long double shift = dist[ref];
  if (shift >= INF / 2)
    shift = 0.0L;
  for (auto &x : dist) {
    if (x < INF / 2)
      x -= shift;
  }
  return dist;
}

// Given (g,h) for the induced 1-player game (Max fixed),
// pick Min's greedy optimal policy w.r.t. the Bellman min equation:
// h(u) = min_e ( (r-g) + h(v) )
static vector<int> compute_min_best_response(const Game &G,
                                             const vector<int> &sigma,
                                             long double g,
                                             const vector<long double> &h) {

  vector<int> tau(G.n, -1);
  for (int u = 0; u < G.n; ++u) {
    if (G.owner[u] != 1)
      continue;
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
static bool improve_max_policy(const Game &G, vector<int> &sigma, long double g,
                               const vector<long double> &h) {

  bool improved = false;
  for (int u = 0; u < G.n; ++u) {
    if (G.owner[u] != 0)
      continue;

    int cur = sigma[u];
    if (cur < 0)
      cur = 0;

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
PolicyResult mean_payoff_strategy_iteration(const Game &G,
                                            int max_iters = 100000) {
  PolicyResult res;
  res.sigma_max_edge_idx.assign(G.n, -1);
  res.tau_min_edge_idx.assign(G.n, -1);

  // Initial Max strategy: pick first outgoing edge for Max states.
  vector<int> sigma(G.n, -1);
  for (int u = 0; u < G.n; ++u) {
    if (G.owner[u] == 0)
      sigma[u] = 0;
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
    if (G.owner[u] != 0)
      res.sigma_max_edge_idx[u] = -1;
    if (G.owner[u] != 1)
      res.tau_min_edge_idx[u] = -1;
  }
  return res;
}

// Pretty-print a policy as "state -> (to,reward)" for owned states.
static void print_policy(const Game &G, const vector<int> &pol, int ownerFlag,
                         const string &name) {
  cout << name << " policy (" << (ownerFlag == 0 ? "Max" : "Min")
       << " states only):\n";
  for (int u = 0; u < G.n; ++u) {
    if (G.owner[u] != ownerFlag)
      continue;
    int ei = pol[u];
    if (ei < 0 || ei >= (int)G.out[u].size())
      continue;
    const auto &e = G.out[u][ei];
    cout << "  state " << u << " -> edge#" << ei << " (to " << e.to
         << ", reward " << e.reward << ")\n";
  }
}

// continued fraction rational approximation: find p/q approx of x with q <=
// maxDen
pair<long long, long long> rational_approx(long double x, long long maxDen) {
  const long double EPS = 1e-18L;
  // continued fraction
  vector<long long> a;
  long double y = x;
  for (int i = 0; i < 100 && (long long)floor(y) <= (1LL << 60); ++i) {
    long long ai = (long long)floor(y + EPS);
    a.push_back(ai);
    long double frac = y - (long double)ai;
    if (fabsl(frac) < 1e-20L)
      break;
    y = 1.0L / frac;
  }
  // convergents
  long long p0 = 0, q0 = 1, p1 = 1, q1 = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    long long ai = a[i];
    long long p2 = ai * p1 + p0;
    long long q2 = ai * q1 + q0;
    if (q2 > maxDen) {
      // find final with bounding
      long long k = (maxDen - q0) / q1;
      if (k <= 0)
        break;
      p2 = k * p1 + p0;
      q2 = k * q1 + q0;
      return {p2, q2};
    }
    p0 = p1;
    q0 = q1;
    p1 = p2;
    q1 = q2;
  }
  // fallback
  if (q1 == 0)
    return {(long long)round(x), 1LL};
  return {p1, q1};
}

// Find the largest p/Q inside [c_lo,c_hi] that yields a negative game value.
// Returns pair(final_ratio, PolicyResult) where final_ratio = p_best / Q.
// If no p in the integer bracket yields g<0, returns final_ratio = -1 and a default PolicyResult.
pair<long double, PolicyResult> find_largest_negative_ratio_by_p_verbose(
    int N, long double c_lo, long double c_hi,
    int Q = 5000, int maxIter = 80, int neighborRadius = 3) {

  // Convert real bracket to integer p bounds (ceil/floor)
  long long p_lo = (long long)ceil(c_lo * (long double)Q - 1e-18L);
  long long p_hi = (long long)floor(c_hi * (long double)Q + 1e-18L);

  if (p_lo < 0) p_lo = 0;
  if (p_hi < p_lo) {
    cerr << "Empty integer bracket for chosen Q. Increase Q or widen [c_lo,c_hi].\n";
    return {-1.0L, PolicyResult()};
  }

  cout << fixed << setprecision(12);
  cout << "Searching largest negative p/Q with Q=" << Q
       << "  integer bracket p in [" << p_lo << "," << p_hi << "]"
       << "  (c_lo=" << (double)c_lo << ", c_hi=" << (double)c_hi << ")\n";

  PolicyResult bestNegSol;
  long long bestNegP = -1; // largest p found so far with g < 0
  long double bestNegPer = 0.0L;

  PolicyResult lastSol;
  long long last_p = -1;

  for (int it = 0; it < maxIter && p_lo <= p_hi; ++it) {
    long long p_mid = (p_lo + p_hi) >> 1; // integer mid
    int p = (int)p_mid;
    int q = Q;

    Game G = build_param_game(N, p, q);
    PolicyResult sol = mean_payoff_strategy_iteration(G);
    long double g = sol.value;            // scaled by q
    long double g_per = g / (long double)q; // normalized mean per step

    cout << "it=" << setw(3) << it
         << "  p_mid=" << setw(6) << p << "  p/q=" << setw(12) << (double)p / (double)q
         << "  g=" << setw(12) << (double)g
         << "  g/q=" << setw(12) << (double)g_per << "\n";

    lastSol = sol;
    last_p = p_mid;

    if (g < 0.0L) {
      // record as a candidate (we want the largest p with negative value)
      if (p_mid > bestNegP) {
        bestNegP = p_mid;
        bestNegSol = sol;
        bestNegPer = g_per;
      }
      // increase lower bound to search for larger negative p
      p_lo = p_mid + 1;
    } else {
      // g >= 0, move high down
      p_hi = p_mid - 1;
    }
  }

  // After binary loop: p_hi < p_lo OR we hit maxIter.
  // bestNegP is our best recorded negative p (largest seen in loop).
  // However, binary could exit with neighbor integers unexplored; check small neighborhood
  // around final p_hi and p_lo to be safe.
  long long neigh_lo = max(0LL, p_lo - neighborRadius - 2);
  long long neigh_hi = p_hi + neighborRadius + 2;
  neigh_lo = max(neigh_lo, 0LL);
  // clamp to reasonable range
  if (neigh_hi < neigh_lo) neigh_hi = neigh_lo;

  // expand search neighborhood a bit but avoid huge loops
  long long maxChecks = 1000;
  long long totalToCheck = neigh_hi - neigh_lo + 1;
  if (totalToCheck > maxChecks) {
    // limit to center window
    long long center = (neigh_lo + neigh_hi) / 2;
    neigh_lo = max(0LL, center - maxChecks/2);
    neigh_hi = neigh_lo + maxChecks - 1;
  }

  cout << "Checking small neighborhood p in [" << neigh_lo << "," << neigh_hi << "] for any larger negative p...\n";
  for (long long pp = neigh_lo; pp <= neigh_hi; ++pp) {
    int p = (int)pp;
    int q = Q;
    Game G = build_param_game(N, p, q);
    PolicyResult sol = mean_payoff_strategy_iteration(G);
    long double g = sol.value;
    long double g_per = g / (long double)q;
    cout << "  test p=" << setw(6) << p << "  p/q=" << setw(12) << (double)p / (double)q
         << "  g=" << setw(12) << (double)g << "  g/q=" << setw(12) << (double)g_per << "\n";
    if (g < 0.0L && pp > bestNegP) {
      bestNegP = pp;
      bestNegSol = sol;
      bestNegPer = g_per;
    }
  }

  if (bestNegP < 0) {
    cout << "No p in integer bracket produced negative game value.\n";
    return {-1.0L, PolicyResult()};
  } else {
    long double final_ratio = (long double)bestNegP / (long double)Q;
    cout << "Largest p with negative g is p=" << bestNegP << " (p/q=" << (double)final_ratio
         << ") with g/q=" << (double)bestNegPer << "\n";
    return {final_ratio, bestNegSol};
  }
}

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  int N = 40;
  long double c_lo = 7747.0L / 5000.0L;
  long double c_hi = 39.0L / 25.0L;
  int Q = 10000; // evaluation denominator; increase for finer resolution
  int maxIter = 60;

  auto res_pair = find_largest_negative_ratio_by_p_verbose(N, c_lo, c_hi, Q, maxIter);
  long double c_star = res_pair.first;
  PolicyResult finalSol = res_pair.second;

  cout << "\nEstimated c* ≈ " << (double)c_star << "\n";

  // Optional: approximate c_star by a "nice" rational with denominator <= 2000
  // auto frac = rational_approx(c_star, 2000);
  // cout << "Approx rational p/q = " << frac.first << " / " << frac.second
  //      << " ≈ " << (long double)frac.first / (long double)frac.second << "\n";

  cout << "Final game value at that rational (from solver) = "
       << (double)finalSol.value << "\n\n";

  // Print final policies
  // print_policy(build_param_game(N, (int)frac.first, (int)frac.second),
  // finalSol.sigma_max_edge_idx, 0, "Max"); print_policy(build_param_game(N,
  // (int)frac.first, (int)frac.second), finalSol.tau_min_edge_idx, 1, "Min");

  return 0;
}
