#pragma once
#include "game_graph.h"
#include <vector>

// Compute integer bias H[u] (representing h[u] = H[u]/g_den) via reverse
// Bellman-Ford from reference node ref under modified weights g_den*r - g_num.
// ref must lie on a minimum-mean cycle of F(sigma, tau).
// Nodes that cannot reach ref retain H[u] = INF64.
std::vector<long long> run_bellman_ford(
    const GameGraph&        g,
    const std::vector<int>& sigma,
    const std::vector<int>& tau,
    long long               g_num,
    long long               g_den,
    int                     ref);
