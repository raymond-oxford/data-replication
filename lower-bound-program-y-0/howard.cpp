#include "howard.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <numeric>
#include <omp.h>

HowardResult run_howard(const GameGraph& g, const std::vector<int>& sigma) {
    int n = g.num_nodes();
    std::vector<int> tau(n, 0);

    // Precompute Max successors (sigma is fixed for this call).
    std::vector<int>       max_to(n, -1);
    std::vector<long long> max_r(n, 0);
    for (int u = 0; u < n; ++u) {
        if (g.is_max(u)) {
            auto e = g.get_edge(u, sigma[u]);
            max_to[u] = e.to;
            max_r[u]  = e.reward;
        }
    }

    std::vector<int>       fn(n);
    std::vector<long long> fr(n);
    std::vector<long long> H(n, INF64);

    long long g_num = 0, g_den = 1;
    int ref_node = 0;

    for (;;) {
        // ------------------------------------------------------------------
        // Step A: build functional graph from current (sigma, tau).
        // ------------------------------------------------------------------
        for (int u = 0; u < n; ++u) {
            if (g.is_max(u)) {
                fn[u] = max_to[u];
                fr[u] = max_r[u];
            } else {
                auto e = g.get_edge(u, tau[u]);
                fn[u] = e.to;
                fr[u] = e.reward;
            }
        }

        // ------------------------------------------------------------------
        // Step B: find all cycles; compute min-mean cycle Γ*.
        // Chain-following on functional graph (each node has one successor).
        // ------------------------------------------------------------------
        struct CycleInfo {
            long long sum_r;
            int length;
            int entry; // first node of cycle in detection order
        };
        std::vector<CycleInfo> cycles;

        // color: 0=unvisited, 1=in_current_path, 2=done
        std::vector<int8_t> color(n, 0);
        std::vector<int>    path_idx(n, -1); // index in current path
        std::vector<int>    path;
        path.reserve(n);

        for (int start = 0; start < n; ++start) {
            if (color[start] != 0) continue;
            path.clear();
            int u = start;
            while (color[u] == 0) {
                color[u]    = 1;
                path_idx[u] = (int)path.size();
                path.push_back(u);
                u = fn[u];
            }
            if (color[u] == 1) {
                // Cycle: path[path_idx[u] .. end]
                int ci = path_idx[u];
                CycleInfo c;
                c.entry  = path[ci];
                c.length = (int)path.size() - ci;
                c.sum_r  = 0;
                for (int i = ci; i < (int)path.size(); ++i)
                    c.sum_r += fr[path[i]];
                cycles.push_back(c);
            }
            for (int v : path) {
                color[v]    = 2;
                path_idx[v] = -1;
            }
        }

        assert(!cycles.empty());

        // Find the min-mean cycle by cross-multiplication.
        int best_ci = 0;
        for (int i = 1; i < (int)cycles.size(); ++i) {
            const auto& a = cycles[i];
            const auto& b = cycles[best_ci];
            // a.sum_r/a.length < b.sum_r/b.length  <=>  a.sum_r*b.length < b.sum_r*a.length
            if (a.sum_r * (long long)b.length < b.sum_r * (long long)a.length)
                best_ci = i;
        }
        const CycleInfo& gamma = cycles[best_ci];

        {
            long long d = std::gcd(std::abs(gamma.sum_r), (long long)gamma.length);
            if (d == 0) d = 1;
            g_num = gamma.sum_r / d;
            g_den = gamma.length / d;
        }
        ref_node = gamma.entry;

        // ------------------------------------------------------------------
        // Step C: compute bias H (scaled by g_den: H[u] = h[u] * g_den).
        // ------------------------------------------------------------------
        std::fill(H.begin(), H.end(), INF64);

        // Assign H on Γ*: H[fn[u]] = H[u] + g_num - g_den * fr[u].
        {
            int u = gamma.entry;
            H[u] = 0;
            for (int k = 0; k < gamma.length - 1; ++k) {
                int v = fn[u];
                H[v] = H[u] + g_num - g_den * fr[u];
                u = v;
            }
        }

        // Propagate H to tails leading to Γ*.
        // hstatus: 0=unknown, 1=in_current_path, 2=done
        std::vector<int8_t> hs(n, 0);
        {
            int u = gamma.entry;
            for (int k = 0; k < gamma.length; ++k) {
                hs[u] = 2;
                u = fn[u];
            }
        }

        std::vector<int> hpath;
        hpath.reserve(n);
        for (int start = 0; start < n; ++start) {
            if (hs[start] != 0) continue;
            hpath.clear();
            int v = start;
            while (hs[v] == 0) {
                hs[v] = 1;
                hpath.push_back(v);
                v = fn[v];
            }
            // v is either done (hs==2) or in current path (hs==1, a different cycle).
            if (hs[v] == 2 && H[v] < INF64) {
                // Assign backward: H[u] = g_den*fr[u] - g_num + H[fn[u]].
                for (int i = (int)hpath.size() - 1; i >= 0; --i) {
                    int u = hpath[i];
                    H[u] = g_den * fr[u] - g_num + H[fn[u]];
                }
            }
            // else: nodes stay INF64.
            for (int u : hpath) hs[u] = 2;
        }

        // ------------------------------------------------------------------
        // Step D: improve Min.
        // Minimize g_den*r - g_num + H[dest] over outgoing edges.
        // ------------------------------------------------------------------
        bool improved = false;
        #pragma omp parallel for reduction(||: improved) schedule(dynamic, 64)
        for (int u = 0; u < n; ++u) {
            if (g.is_max(u)) continue;

            auto cur_e = g.get_edge(u, tau[u]);
            long long cur_score = (H[cur_e.to] < INF64)
                ? g_den * cur_e.reward - g_num + H[cur_e.to]
                : INF64;

            int       best_ei    = tau[u];
            long long best_score = cur_score;

            int ne = g.num_edges(u);
            for (int ei = 0; ei < ne; ++ei) {
                auto e = g.get_edge(u, ei);
                if (H[e.to] >= INF64) continue;
                long long score = g_den * e.reward - g_num + H[e.to];
                if (score < best_score) {
                    best_score = score;
                    best_ei    = ei;
                }
            }

            if (best_score < cur_score) {
                tau[u]   = best_ei;
                improved = true;
            }
        }

        if (!improved) break;
    }

    return HowardResult{g_num, g_den, ref_node, std::move(tau)};
}
