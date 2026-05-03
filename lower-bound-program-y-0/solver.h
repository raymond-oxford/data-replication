#pragma once
#include "game_graph.h"
#include <string>
#include <vector>

// Improve Max's strategy using the integer bias vector H.
// For each Max node, picks the edge maximizing g_den*reward + H[dest]
// among edges whose destination has finite H.
// Returns true if any sigma[u] was updated.
bool improve_max_policy(
    const GameGraph&        g,
    std::vector<int>&       sigma,
    const std::vector<long long>& H,
    long long               g_den);

// Extract a new candidate ratio from the min-score cycle of F(sigma, tau).
// Computes score(Γ) = p*Σopt_num - q*Σalg_num for each cycle and picks the minimum.
// If Σopt_num of the best cycle is <= 0, returns {p, q} unchanged.
// Otherwise returns {Σalg_num / gcd, Σopt_num / gcd} as the new (p, q).
std::pair<long long,long long> extract_cycle_ratio(
    const GameGraph&        g,
    const std::vector<int>& sigma,
    const std::vector<int>& tau,
    long long               p,
    long long               q);

// Verify the optimality certificate (sigma, tau, g=g_num/g_den, H).
// Checks four conditions:
//   0. All H[u] finite.
//   1. Max local optimality.
//   2. Min local optimality.
//   3. Bellman equality on chosen edges.
//   4. Every cycle of F(sigma,tau) has mean exactly g_num/g_den.
// Returns true on success; prints the failing condition and returns false otherwise.
bool verify_certificate(
    const GameGraph&              g,
    const std::vector<int>&       sigma,
    const std::vector<int>&       tau,
    long long                     g_num,
    long long                     g_den,
    const std::vector<long long>& H);

// Write optimal policy to a CSV file in the same format as ./game --fast-search.
// Header: mode,d,w,action,y_value,x_value
// One STP row and one LTP row per (d,w) pair with d=0..N, w=0..d.
void save_policy(
    const GameGraph&        g,
    const std::vector<int>& sigma,
    const std::vector<int>& tau,
    const std::string&      filename);
