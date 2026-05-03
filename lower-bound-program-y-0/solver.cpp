#include "solver.h"
#include <algorithm>
#include <omp.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

// ---------------------------------------------------------------------------
// Shared helper: find all cycles in the functional graph F(sigma, tau).
// Calls visit(entry_node, sum_reward, sum_opt, sum_alg, length) for each cycle.
// ---------------------------------------------------------------------------
template<typename Visit>
static void find_functional_cycles(
    const GameGraph&        g,
    const std::vector<int>& sigma,
    const std::vector<int>& tau,
    Visit                   visit)
{
    int n = g.num_nodes();
    std::vector<int8_t> color(n, 0); // 0=unvisited, 1=in_path, 2=done
    std::vector<int>    path_idx(n, -1);
    std::vector<int>    path;
    path.reserve(n);

    // Per-path edge info (reward, opt_num, alg_num for the edge leaving path[i]).
    std::vector<long long> path_r, path_opt, path_alg;
    path_r.reserve(n); path_opt.reserve(n); path_alg.reserve(n);

    for (int start = 0; start < n; ++start) {
        if (color[start] != 0) continue;
        path.clear(); path_r.clear(); path_opt.clear(); path_alg.clear();

        int u = start;
        while (color[u] == 0) {
            color[u]    = 1;
            path_idx[u] = (int)path.size();
            path.push_back(u);
            auto e = g.is_max(u) ? g.get_edge(u, sigma[u]) : g.get_edge(u, tau[u]);
            path_r.push_back(e.reward);
            path_opt.push_back(e.opt_num);
            path_alg.push_back(e.alg_num);
            u = e.to;
        }
        if (color[u] == 1) {
            int ci = path_idx[u];
            int len = (int)path.size() - ci;
            long long sr = 0, so = 0, sa = 0;
            for (int i = ci; i < (int)path.size(); ++i) {
                sr += path_r[i]; so += path_opt[i]; sa += path_alg[i];
            }
            visit(path[ci], sr, so, sa, len);
        }
        for (int v : path) { color[v] = 2; path_idx[v] = -1; }
    }
}

// ---------------------------------------------------------------------------
bool improve_max_policy(
    const GameGraph&              g,
    std::vector<int>&             sigma,
    const std::vector<long long>& H,
    long long                     g_den)
{
    bool improved = false;
    int n = g.num_nodes();
    #pragma omp parallel for reduction(||: improved) schedule(dynamic, 64)
    for (int u = 0; u < n; ++u) {
        if (!g.is_max(u)) continue;

        auto cur_e = g.get_edge(u, sigma[u]);
        long long cur_score = (H[cur_e.to] < INF64)
            ? g_den * cur_e.reward + H[cur_e.to]
            : -INF64; // any finite alternative is better

        int       best_ei    = sigma[u];
        long long best_score = cur_score;

        int ne = g.num_edges(u);
        for (int ei = 0; ei < ne; ++ei) {
            auto e = g.get_edge(u, ei);
            if (H[e.to] >= INF64) continue;
            long long score = g_den * e.reward + H[e.to];
            if (score > best_score) {
                best_score = score;
                best_ei    = ei;
            }
        }

        if (best_score > cur_score) {
            sigma[u] = best_ei;
            improved  = true;
        }
    }
    return improved;
}

// ---------------------------------------------------------------------------
std::pair<long long,long long> extract_cycle_ratio(
    const GameGraph&        g,
    const std::vector<int>& sigma,
    const std::vector<int>& tau,
    long long               p,
    long long               q)
{
    // Find the cycle minimizing score = (p*sum_opt - q*sum_alg) / length.
    bool        found    = false;
    long long   best_num = 0, best_len = 1; // best score numerator / length
    long long   best_alg = 0, best_opt = 0;

    find_functional_cycles(g, sigma, tau,
        [&](int /*entry*/, long long sr, long long so, long long sa, int len) {
            (void)sr;
            // score = (p*so - q*sa) / len; compare by cross-multiply
            long long num = p * so - q * sa;
            if (!found || num * best_len < best_num * (long long)len) {
                best_num = num;
                best_len = (long long)len;
                best_alg = sa;
                best_opt = so;
                found    = true;
            }
        });

    if (!found || best_opt <= 0)
        return {p, q};

    long long d = std::gcd(std::abs(best_alg), std::abs(best_opt));
    if (d == 0) d = 1;
    return {best_alg / d, best_opt / d};
}

// ---------------------------------------------------------------------------
bool verify_certificate(
    const GameGraph&              g,
    const std::vector<int>&       sigma,
    const std::vector<int>&       tau,
    long long                     g_num,
    long long                     g_den,
    const std::vector<long long>& H)
{
    int n = g.num_nodes();

    // Condition 0: all H finite.
    for (int u = 0; u < n; ++u) {
        if (H[u] >= INF64) {
            std::cerr << "Certificate FAIL cond 0: H[" << u << "] = +inf\n";
            return false;
        }
    }

    // Conditions 1 & 2: local optimality, and condition 3: Bellman equality.
    for (int u = 0; u < n; ++u) {
        int chosen = g.is_max(u) ? sigma[u] : tau[u];
        auto ce = g.get_edge(u, chosen);
        long long chosen_score = g_den * ce.reward + H[ce.to];

        // Condition 3: Bellman equality on chosen edge.
        long long bellman = g_den * ce.reward - g_num + H[ce.to];
        if (H[u] != bellman) {
            std::cerr << "Certificate FAIL cond 3: node " << u
                      << " H[u]=" << H[u] << " expected=" << bellman << "\n";
            return false;
        }

        // Conditions 1 & 2: no strictly better edge exists.
        int ne = g.num_edges(u);
        for (int ei = 0; ei < ne; ++ei) {
            auto e = g.get_edge(u, ei);
            long long score = g_den * e.reward + H[e.to];
            if (g.is_max(u) && score > chosen_score) {
                std::cerr << "Certificate FAIL cond 1: Max node " << u
                          << " ei=" << ei << " score=" << score
                          << " > chosen=" << chosen_score << "\n";
                return false;
            }
            if (!g.is_max(u) && score < chosen_score) {
                std::cerr << "Certificate FAIL cond 2: Min node " << u
                          << " ei=" << ei << " score=" << score
                          << " < chosen=" << chosen_score << "\n";
                return false;
            }
        }
    }

    // Condition 4: every cycle of F(sigma,tau) has mean exactly g_num/g_den.
    bool cond4_ok = true;
    find_functional_cycles(g, sigma, tau,
        [&](int entry, long long sr, long long /*so*/, long long /*sa*/, int len) {
            if (!cond4_ok) return;
            // g_den * sr == len * g_num ?
            if (g_den * sr != (long long)len * g_num) {
                std::cerr << "Certificate FAIL cond 4: cycle from node " << entry
                          << " sum_r=" << sr << " len=" << len
                          << " expected mean=" << g_num << "/" << g_den << "\n";
                cond4_ok = false;
            }
        });

    return cond4_ok;
}

// ---------------------------------------------------------------------------
void save_policy(
    const GameGraph&        g,
    const std::vector<int>& sigma,
    const std::vector<int>& tau,
    const std::string&      filename)
{
    int N = g.N();

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open " << filename << "\n";
        return;
    }

    ofs << "mode,d,w,action,y_value,x_value\n";
    ofs << std::fixed << std::setprecision(10);

    for (int d = 0; d <= N; ++d) {
        for (int w = 0; w <= d; ++w) {
            double d_scaled = (double)d / (double)N;
            double w_scaled = (double)w / (double)N;

            // STP row
            {
                int node = g.base_node(d, w, 0);
                int ei = (node >= 0) ? sigma[node] : -1;
                double y_scaled = (ei >= 0) ? (double)ei / (double)N : 0.0;
                std::string action_str = (ei >= 0) ? std::to_string(ei) : "";

                double x_scaled = -1.0;
                if (ei >= 0) {
                    int stpmin = g.stpmin_node(d, w, ei);
                    if (stpmin >= 0) {
                        int min_ei = tau[stpmin];
                        if (min_ei >= 2)
                            x_scaled = (double)(min_ei - 1) / (double)N;
                    }
                }
                ofs << "STP," << d_scaled << "," << w_scaled << ","
                    << action_str << "," << y_scaled << "," << x_scaled << "\n";
            }

            // LTP row
            {
                int node = g.base_node(d, w, 1);
                int ei = (node >= 0) ? sigma[node] : -1;
                std::string action_str = (ei >= 0) ? std::to_string(ei) : "";

                double x_scaled = -1.0;
                int ltptop = g.ltptop_node(d, w);
                if (ltptop >= 0) {
                    int min_ei = tau[ltptop];
                    if (min_ei >= 0)
                        x_scaled = (double)(min_ei / 2) / (double)N;
                }
                ofs << "LTP," << d_scaled << "," << w_scaled << ","
                    << action_str << ",0.0000000000," << x_scaled << "\n";
            }
        }
    }

    ofs.close();
    std::cerr << "Saved scaled Max policy CSV to " << filename << "\n";
}
