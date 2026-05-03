#pragma once
#include "alg.hpp"
#include "bellman_ford.hpp"
#include <vector>
#include <unordered_map>
#include <optional>
#include <atomic>
#include <algorithm>
#include <fstream>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// check_potential
//
// For each reachable node u and each edge u→v, checks the BF inequality:
//   pot(u) + (alpha * opt - beta * online) >= pot(v)
//
// Edges where either endpoint returns nullopt are skipped (partial potential).
// ---------------------------------------------------------------------------

struct PotentialCheckResult {
    int num_edges_checked;  // edges where both endpoints have defined potential
    int num_violations;     // edges where the inequality is violated
    int worst_slack;        // min(pot(u) + weight - pot(v)) over checked edges
                            // negative means violated; 0 means tight; positive is slack
};

// check_potential
//
// For each reachable node u and each edge u→v, checks the BF inequality:
//   pot(u) + (alpha * opt - beta * online) >= pot(v)
//
// Edges where either endpoint returns nullopt are skipped (partial potential).
//
// If violations_path is non-empty, all violated edges are written to a CSV
// sorted by slack ascending (worst violations first).  CSV columns:
//   src_<field>... , dst_<field>... , opt, online, weight, pot_src, pot_dst, slack
template<typename N, typename PotFn>
PotentialCheckResult check_potential(
    const std::vector<N>& reachable,
    const Algorithm<N>& alg,
    int alpha, int beta,
    PotFn pot,
    int lamb = 1,
    const std::string& violations_path = ""
) {
    struct ViolationRow {
        N src, dst;
        int opt, online, weight, pot_src, pot_dst, slack;
    };

    int num_edges_checked = 0;
    int num_violations    = 0;
    int worst_slack       = std::numeric_limits<int>::max();
    std::vector<ViolationRow> rows;

    for (const auto& u : reachable) {
        auto pu = pot(u);
        if (!pu.has_value()) continue;

        for (const auto& edge : alg.transition(u)) {
            const N& v = edge.get_state();
            auto pv = pot(v);
            if (!pv.has_value()) continue;

            int opt    = edge.get_opt();
            int online = edge.get_online();
            int weight = alpha * opt - beta * online;
            int slack  = *pu + weight - *pv;

            ++num_edges_checked;
            if (slack < worst_slack) worst_slack = slack;
            if (slack < 0) {
                ++num_violations;
                if (!violations_path.empty())
                    rows.push_back({ u, v, opt, online, weight, *pu, *pv, slack });
            }
        }
    }

    if (worst_slack == std::numeric_limits<int>::max())
        worst_slack = 0;  // no edges checked

    if (!violations_path.empty() && !rows.empty()) {
        std::sort(rows.begin(), rows.end(),
                  [](const ViolationRow& a, const ViolationRow& b) {
                      return a.slack < b.slack;
                  });

        std::ofstream out(violations_path);
        // Header: src_ prefix for source fields, dst_ prefix for dest fields.
        for (const auto& h : N::csv_header()) out << "src_" << h << ",";
        for (const auto& h : N::csv_header()) out << "dst_" << h << ",";
        out << "opt,online,weight,pot_src,pot_dst,slack\n";

        for (const auto& r : rows) {
            for (auto v : r.src.csv_values(lamb)) out << v << ",";
            for (auto v : r.dst.csv_values(lamb)) out << v << ",";
            out << r.opt << "," << r.online << "," << r.weight << ","
                << r.pot_src << "," << r.pot_dst << "," << r.slack << "\n";
        }
    }

    return { num_edges_checked, num_violations, worst_slack };
}

// ---------------------------------------------------------------------------
// compute_distances_seeded
//
// BF shortest-path seeded with a partial potential map (node.code -> int).
// Seeded nodes enter the frontier with their given initial distances;
// all other nodes start at INF.
//
// Result: dist[v.code] = min over seed nodes s reachable to v of
//           seed[s] + sum(weights along path s→v).
//
// If any seeded node ends up with dist < its seed value, a negative cycle
// passes through the seeds (the partial potential is inconsistent); a
// warning is printed to stderr.
// ---------------------------------------------------------------------------
template<typename N>
std::vector<int> compute_distances_seeded(
    const std::unordered_map<uint64_t, int>& seed,
    const Algorithm<N>& alg,
    int alpha, int beta
) {
    constexpr int max_code = 1 << N::size;
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif

    std::vector<std::atomic<int>>      dist(max_code);
    std::vector<std::atomic<uint32_t>> update_count(max_code);
    std::vector<std::atomic<uint8_t>>  in_queue(max_code);
    std::vector<std::atomic<uint8_t>>  visited(max_code);
    std::atomic<int>                   num_discovered{0};

    for (int i = 0; i < max_code; ++i) {
        dist[i].store(INF, std::memory_order_relaxed);
        update_count[i].store(0, std::memory_order_relaxed);
        in_queue[i].store(0, std::memory_order_relaxed);
        visited[i].store(0, std::memory_order_relaxed);
    }

    // Seed the distance vector and build the initial frontier.
    std::vector<N> frontier;
    frontier.reserve(seed.size());
    for (const auto& [code, val] : seed) {
        dist[code].store(val, std::memory_order_relaxed);
        visited[code].store(1, std::memory_order_relaxed);
        in_queue[code].store(1, std::memory_order_relaxed);
        num_discovered.fetch_add(1, std::memory_order_relaxed);
        // Reconstruct a node from its code to push onto the frontier.
        N node;
        node.code = code;
        frontier.push_back(node);
    }

    std::vector<std::vector<N>> local_next(nthreads);
    bool cycle_warned = false;

    while (!frontier.empty()) {
        std::atomic<bool> cycle_found{false};

#pragma omp parallel for schedule(dynamic, 32)
        for (int idx = 0; idx < (int)frontier.size(); ++idx) {
            N u = frontier[idx];
            in_queue[u.code].store(0, std::memory_order_relaxed);

            for (const auto& edge : alg.transition(u)) {
                int weight = alpha * edge.get_opt() - beta * edge.get_online();
                const N& v  = edge.get_state();
                int     c_v = v.code;

                uint8_t was_unvisited = 0;
                if (visited[c_v].compare_exchange_strong(was_unvisited, 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    num_discovered.fetch_add(1, std::memory_order_relaxed);
                }

                int du       = dist[u.code].load(std::memory_order_relaxed);
                int new_dist = (du == INF) ? INF : du + weight;

                if (new_dist < INF && atomic_relax_min(dist[c_v], new_dist)) {
                    uint32_t cnt = update_count[c_v].fetch_add(1, std::memory_order_relaxed) + 1;
                    int nd = num_discovered.load(std::memory_order_relaxed);
                    if ((int)cnt >= nd)
                        cycle_found.store(true, std::memory_order_relaxed);

                    uint8_t expected_q = 0;
                    if (in_queue[c_v].compare_exchange_strong(expected_q, 1,
                            std::memory_order_relaxed, std::memory_order_relaxed)) {
#ifdef _OPENMP
                        local_next[omp_get_thread_num()].push_back(v);
#else
                        local_next[0].push_back(v);
#endif
                    }
                }
            }
        }

        frontier.clear();
        for (auto& lv : local_next) {
            for (auto& node : lv) frontier.push_back(node);
            lv.clear();
        }

        if (cycle_found.load(std::memory_order_relaxed) && !cycle_warned) {
            std::cerr << "Warning: negative cycle detected through seed nodes "
                         "(partial potential is inconsistent)\n";
            cycle_warned = true;
        }
    }

    std::vector<int> result(max_code);
    for (int i = 0; i < max_code; ++i)
        result[i] = dist[i].load(std::memory_order_relaxed);
    return result;
}

// ---------------------------------------------------------------------------
// save_potential_range_csv
//
// Writes a CSV with pot_min and pot_max columns for each reachable node.
//
//   pot_min(v) = d_rev[v]                    (reverse BF, already >= 0)
//   pot_max(v) = (d_fwd[v] - min_d) + c      (forward BF, normalized + aligned)
//
// Both columns are divided by scale = beta * lamb.
// Nodes where either distance is infinite are omitted.
// ---------------------------------------------------------------------------
template<typename N>
void save_potential_range_csv(
    const std::vector<N>& reachable,
    const std::vector<int>& d_fwd,
    const std::vector<int>& d_rev,
    int min_d,
    int c,
    int beta, int lamb,
    const std::string& filename
) {
    constexpr int NEG_INF = std::numeric_limits<int>::min();
    double scale = static_cast<double>(beta) * lamb;

    std::ofstream out(filename);
    for (const auto& h : N::csv_header()) out << h << ",";
    out << "pot_min,pot_max\n";

    for (const auto& node : reachable) {
        int df = d_fwd[node.code];
        int dr = d_rev[node.code];
        if (df == INF || dr == NEG_INF) continue;

        double pot_min = static_cast<double>(dr) / scale;
        double pot_max = static_cast<double>((df - min_d) + c) / scale;

        for (auto v : node.csv_values(lamb)) out << v << ",";
        out << pot_min << "," << pot_max << "\n";
    }
}

// ---------------------------------------------------------------------------
// save_phi_range_violations_csv
//
// Writes a CSV containing only nodes where the candidate potential phi
// lies outside [D_min, D_max].  Columns: node fields + pot_min, pot_max, pot_phi.
// phi_vals is indexed by node.code; INF means the potential is undefined at
// that node (those nodes are skipped).
// ---------------------------------------------------------------------------
template<typename N>
void save_phi_range_violations_csv(
    const std::vector<N>& reachable,
    const std::vector<int>& d_fwd,
    const std::vector<int>& d_rev,
    int min_d,
    int c,
    int beta, int lamb,
    const std::string& filename,
    const std::vector<int>& phi_vals  // indexed by node.code; INF = undefined
) {
    constexpr int NEG_INF = std::numeric_limits<int>::min();
    double scale = static_cast<double>(beta) * lamb;

    std::ofstream out(filename);
    for (const auto& h : N::csv_header()) out << h << ",";
    out << "pot_min,pot_max,pot_phi\n";

    for (const auto& node : reachable) {
        int df = d_fwd[node.code];
        int dr = d_rev[node.code];
        if (df == INF || dr == NEG_INF) continue;

        int phi = phi_vals[node.code];
        if (phi == INF) continue;  // undefined

        int d_min = dr;
        int d_max = (df - min_d) + c;
        if (phi >= d_min && phi <= d_max) continue;  // in range, skip

        for (auto v : node.csv_values(lamb)) out << v << ",";
        out << static_cast<double>(d_min) / scale << ","
            << static_cast<double>(d_max) / scale << ","
            << static_cast<double>(phi)   / scale << "\n";
    }
}
