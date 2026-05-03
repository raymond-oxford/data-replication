#include "bellman_ford.h"
#include <omp.h>

std::vector<long long> run_bellman_ford(
    const GameGraph&        g,
    const std::vector<int>& sigma,
    const std::vector<int>& /*tau*/,   // identifies ref via Howard's; not used in BF itself
    long long               g_num,
    long long               g_den,
    int                     ref)
{
    int n = g.num_nodes();
    std::vector<long long> H(n, INF64);
    H[ref] = 0;

    // Precompute Max successors (sigma is fixed).
    std::vector<int>       max_to(n, -1);
    std::vector<long long> max_r(n, 0);
    for (int u = 0; u < n; ++u) {
        if (g.is_max(u)) {
            auto e  = g.get_edge(u, sigma[u]);
            max_to[u] = e.to;
            max_r[u]  = e.reward;
        }
    }

    // Reverse Bellman-Ford: shortest-path distance to ref under
    // modified weight g_den*r - g_num on each edge.
    // Max nodes follow sigma; Min nodes take the best available edge.
    // No negative cycles exist (g is the min cycle mean under sigma).
    // Double-buffered so parallel reads (H) and writes (H_new) don't race.
    std::vector<long long> H_new(n);
    for (;;) {
        bool changed = false;
        #pragma omp parallel for reduction(||: changed) schedule(static)
        for (int u = 0; u < n; ++u) {
            long long best = H[u];
            if (g.is_max(u)) {
                if (H[max_to[u]] < INF64) {
                    long long val = g_den * max_r[u] - g_num + H[max_to[u]];
                    if (val < best) best = val;
                }
            } else {
                int ne = g.num_edges(u);
                for (int ei = 0; ei < ne; ++ei) {
                    auto e = g.get_edge(u, ei);
                    if (H[e.to] >= INF64) continue;
                    long long val = g_den * e.reward - g_num + H[e.to];
                    if (val < best) best = val;
                }
            }
            H_new[u] = best;
            if (best < H[u]) changed = true;
        }
        H = H_new;
        if (!changed) break;
    }

    return H;
}
