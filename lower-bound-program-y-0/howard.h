#pragma once
#include "game_graph.h"
#include <vector>

struct HowardResult {
    long long g_num;   // numerator of gain: sum of rewards on min-mean cycle Γ*
    long long g_den;   // denominator of gain: length of Γ* (always > 0)
    int ref;           // any node on Γ* (min-mean cycle)
    std::vector<int> tau; // converged Min strategy (index into out-edges per Min node)
};

// Run Howard's policy iteration for Min given a fixed Max strategy sigma.
// sigma[u] is the chosen edge index for each Max-owned node u.
// Returns gain g = g_num/g_den (reduced), converged tau, and a reference node on Γ*.
HowardResult run_howard(const GameGraph& g,
                        const std::vector<int>& sigma);
