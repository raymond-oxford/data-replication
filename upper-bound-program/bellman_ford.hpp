#pragma once
#include "alg.hpp"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <limits>
#include <functional>
#include <algorithm>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

constexpr int INF = std::numeric_limits<int>::max();

template<typename N>
struct CycleStep {
    N node;
    Transition<N> edge;

    CycleStep(const N& n, const Transition<N>& e) : node(n), edge(e) {}

    void print(std::ostream& out) const {
        out << "CycleStep:\n  Node: ";
        node.print(out);
        out << "  Edge: ";
        edge.print(out);
    }
};

template<typename N>
std::pair<int, int> compute_cycle_totals(const std::vector<CycleStep<N>>& cycle) {
    int total_opt = 0;
    int total_online = 0;

    for (const auto& step : cycle) {
        total_opt += step.edge.get_opt();
        total_online += step.edge.get_online();
    }

    return {total_opt, total_online};
}

// ---------------------------------------------------------------------------
// Level-parallel Bellman-Ford helpers
// ---------------------------------------------------------------------------

// Atomic min via CAS loop.  Returns true if dist[c_v] was improved.
inline bool atomic_relax_min(std::atomic<int>& slot, int new_val) {
    int old_val = slot.load(std::memory_order_relaxed);
    while (new_val < old_val) {
        if (slot.compare_exchange_weak(old_val, new_val,
                std::memory_order_relaxed, std::memory_order_relaxed))
            return true;
    }
    return false;
}

// Atomic max via CAS loop.  Returns true if dist[c_v] was improved.
inline bool atomic_relax_max(std::atomic<int>& slot, int new_val) {
    int old_val = slot.load(std::memory_order_relaxed);
    while (new_val > old_val) {
        if (slot.compare_exchange_weak(old_val, new_val,
                std::memory_order_relaxed, std::memory_order_relaxed))
            return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// find_negative_cycle  (minimize, forward transition)
// Returns {cycle, distances}.  When a cycle is found, distances is empty
// (intermediate state).  When no cycle is found, distances holds the final
// converged BF distances and cycle is empty.
// ---------------------------------------------------------------------------
template<typename N>
std::pair<std::vector<CycleStep<N>>, std::vector<int>> find_negative_cycle(
    const std::vector<N>& roots,
    const Algorithm<N>& alg,
    int alpha,
    int beta
) {
    constexpr int max_code = 1 << N::size;
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    static bool printed_threads = false;
    if (!printed_threads) {
        std::cerr << "Bellman-Ford using " << nthreads << " thread(s)\n";
        printed_threads = true;
    }

    std::vector<std::atomic<int>>      dist(max_code);
    std::vector<std::atomic<uint32_t>> update_count(max_code);
    std::vector<std::atomic<uint8_t>>  in_queue(max_code);
    std::vector<std::atomic<uint8_t>>  visited(max_code);
    std::vector<N>                     parent_node(max_code);
    std::vector<Transition<N>>         parent_edge(max_code);
    std::atomic<int>                   num_discovered{0};

    for (int i = 0; i < max_code; ++i) {
        dist[i].store(INF, std::memory_order_relaxed);
        update_count[i].store(0, std::memory_order_relaxed);
        in_queue[i].store(0, std::memory_order_relaxed);
        visited[i].store(0, std::memory_order_relaxed);
    }

    std::vector<N> frontier;
    frontier.reserve(roots.size());
    for (const auto& root : roots) {
        int c = root.code;
        dist[c].store(0, std::memory_order_relaxed);
        visited[c].store(1, std::memory_order_relaxed);
        num_discovered.fetch_add(1, std::memory_order_relaxed);
        in_queue[c].store(1, std::memory_order_relaxed);
        frontier.push_back(root);
    }

    auto extract_dist = [&]() {
        std::vector<int> result(max_code);
        for (int i = 0; i < max_code; ++i)
            result[i] = dist[i].load(std::memory_order_relaxed);
        return result;
    };

    std::vector<std::vector<N>> local_next(nthreads);
    N cycle_head = N();
    size_t size_check = SIZE_MAX;  // skip growing-frontier heuristic on first level

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

                // Mark visited (first-visit only, atomic CAS)
                uint8_t was_unvisited = 0;
                if (visited[c_v].compare_exchange_strong(was_unvisited, 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    num_discovered.fetch_add(1, std::memory_order_relaxed);
                }

                int du       = dist[u.code].load(std::memory_order_relaxed);
                int new_dist = du + weight;

                if (atomic_relax_min(dist[c_v], new_dist)) {
                    // intentional racy write: only needed for cycle reconstruction
                    parent_node[c_v] = u;
                    parent_edge[c_v] = edge;

                    uint32_t cnt = update_count[c_v].fetch_add(1, std::memory_order_relaxed) + 1;
                    int nd = num_discovered.load(std::memory_order_relaxed);
                    if ((int)cnt >= nd)
                        cycle_found.store(true, std::memory_order_relaxed);

                    // Enqueue into thread-local next (dedup via in_queue CAS)
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

        // Merge thread-local lists into next frontier
        frontier.clear();
        for (auto& lv : local_next) {
            for (auto& node : lv) frontier.push_back(node);
            lv.clear();
        }

        // Primary cycle detection: update_count >= num_discovered
        if (cycle_found.load(std::memory_order_relaxed) && !cycle_head.is_valid()) {
            int nd = num_discovered.load(std::memory_order_relaxed);
            for (int c = 0; c < max_code && !cycle_head.is_valid(); ++c) {
                uint32_t uc = update_count[c].load(std::memory_order_relaxed);
                if (uc > 0 && (int)uc >= nd) {
                    N x = parent_node[c];
                    if (!x.is_valid()) continue;
                    for (int k = 0; k < nd; ++k) {
                        x = parent_node[x.code];
                        if (!x.is_valid()) break;
                    }
                    if (x.is_valid()) cycle_head = x;
                }
            }
        }

        // Secondary heuristic: growing frontier suggests a cycle
        if (!cycle_head.is_valid() && frontier.size() > size_check) {
            int nd = num_discovered.load(std::memory_order_relaxed);
            if (!frontier.empty()) {
                N cycle_node = frontier[0];
                for (int k = 0; k < nd; ++k) {
                    cycle_node = parent_node[cycle_node.code];
                    if (!cycle_node.is_valid()) break;
                }
                if (cycle_node.is_valid()) cycle_head = cycle_node;
            }
        }

        if (cycle_head.is_valid()) {
            std::vector<CycleStep<N>> cycle;
            N cur = cycle_head;
            int guard = max_code;
            do {
                N par = parent_node[cur.code];
                if (!par.is_valid()) break;
                Transition<N> e = parent_edge[cur.code];
                cycle.emplace_back(par, e);
                cur = par;
                if (--guard < 0) break;
            } while (cur != cycle_head);
            std::reverse(cycle.begin(), cycle.end());
            return {cycle, {}};
        }

        size_check = frontier.size();
    }
    return {{}, extract_dist()};
}

// ---------------------------------------------------------------------------
// compute_distances  (minimize, forward transition; warns but does NOT abort)
// ---------------------------------------------------------------------------
template<typename N>
std::vector<int> compute_distances(
    const std::vector<N>& roots,
    const Algorithm<N>& alg,
    int alpha,
    int beta
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
    std::vector<N>                     parent_node(max_code);
    std::vector<Transition<N>>         parent_edge(max_code);
    std::atomic<int>                   num_discovered{0};

    for (int i = 0; i < max_code; ++i) {
        dist[i].store(INF, std::memory_order_relaxed);
        update_count[i].store(0, std::memory_order_relaxed);
        in_queue[i].store(0, std::memory_order_relaxed);
        visited[i].store(0, std::memory_order_relaxed);
    }

    std::vector<N> frontier;
    frontier.reserve(roots.size());
    for (const auto& root : roots) {
        int c = root.code;
        dist[c].store(0, std::memory_order_relaxed);
        visited[c].store(1, std::memory_order_relaxed);
        num_discovered.fetch_add(1, std::memory_order_relaxed);
        in_queue[c].store(1, std::memory_order_relaxed);
        frontier.push_back(root);
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
                int new_dist = du + weight;

                if (atomic_relax_min(dist[c_v], new_dist)) {
                    parent_node[c_v] = u;
                    parent_edge[c_v] = edge;

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
            std::cerr << "There is a negative cycle";
            cycle_warned = true;
        }
    }

    std::vector<int> result(max_code);
    for (int i = 0; i < max_code; ++i)
        result[i] = dist[i].load(std::memory_order_relaxed);
    return result;
}

// ---------------------------------------------------------------------------
// compute_distances_reverse  (maximize, reverse_transition; returns on cycle)
// ---------------------------------------------------------------------------
template<typename N>
std::vector<int> compute_distances_reverse(
    const std::vector<N>& roots,
    const Algorithm<N>& alg,
    int alpha,
    int beta
) {
    constexpr int max_code = 1 << N::size;
    constexpr int NEG_INF  = std::numeric_limits<int>::min();
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif

    std::vector<std::atomic<int>>      dist(max_code);
    std::vector<std::atomic<uint32_t>> update_count(max_code);
    std::vector<std::atomic<uint8_t>>  in_queue(max_code);
    std::vector<std::atomic<uint8_t>>  visited(max_code);
    std::vector<N>                     parent_node(max_code);
    std::vector<Transition<N>>         parent_edge(max_code);
    std::atomic<int>                   num_discovered{0};

    for (int i = 0; i < max_code; ++i) {
        dist[i].store(NEG_INF, std::memory_order_relaxed);
        update_count[i].store(0, std::memory_order_relaxed);
        in_queue[i].store(0, std::memory_order_relaxed);
        visited[i].store(0, std::memory_order_relaxed);
    }

    std::vector<N> frontier;
    frontier.reserve(roots.size());
    for (const auto& root : roots) {
        int c = root.code;
        dist[c].store(0, std::memory_order_relaxed);
        visited[c].store(1, std::memory_order_relaxed);
        num_discovered.fetch_add(1, std::memory_order_relaxed);
        in_queue[c].store(1, std::memory_order_relaxed);
        frontier.push_back(root);
    }

    std::vector<std::vector<N>> local_next(nthreads);

    auto extract_dist = [&]() {
        std::vector<int> result(max_code);
        for (int i = 0; i < max_code; ++i)
            result[i] = dist[i].load(std::memory_order_relaxed);
        return result;
    };

    while (!frontier.empty()) {
        std::atomic<bool> cycle_found{false};

#pragma omp parallel for schedule(dynamic, 32)
        for (int idx = 0; idx < (int)frontier.size(); ++idx) {
            N u = frontier[idx];
            in_queue[u.code].store(0, std::memory_order_relaxed);

            for (const auto& edge : alg.reverse_transition(u)) {
                int weight = alpha * edge.get_opt() - beta * edge.get_online();
                const N& v  = edge.get_state();
                int     c_v = v.code;

                uint8_t was_unvisited = 0;
                if (visited[c_v].compare_exchange_strong(was_unvisited, 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    num_discovered.fetch_add(1, std::memory_order_relaxed);
                }

                int du      = dist[u.code].load(std::memory_order_relaxed);
                int new_val = du - weight;

                if (atomic_relax_max(dist[c_v], new_val)) {
                    parent_node[c_v] = u;
                    parent_edge[c_v] = edge;

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

        if (cycle_found.load(std::memory_order_relaxed)) {
            std::cerr << "There is a negative cycle";
            return extract_dist();
        }
    }
    return extract_dist();
}

// ---------------------------------------------------------------------------
// compute_distances_reversed  (maximize, forward transition; returns on cycle)
// ---------------------------------------------------------------------------
template<typename N>
std::vector<int> compute_distances_reversed(
    const std::vector<N>& roots,
    const Algorithm<N>& alg,
    int alpha,
    int beta
) {
    constexpr int max_code = 1 << N::size;
    constexpr int NEG_INF  = std::numeric_limits<int>::min();
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif

    std::vector<std::atomic<int>>      dist(max_code);
    std::vector<std::atomic<uint32_t>> update_count(max_code);
    std::vector<std::atomic<uint8_t>>  in_queue(max_code);
    std::vector<std::atomic<uint8_t>>  visited(max_code);
    std::vector<N>                     parent_node(max_code);
    std::vector<Transition<N>>         parent_edge(max_code);
    std::atomic<int>                   num_discovered{0};

    for (int i = 0; i < max_code; ++i) {
        dist[i].store(NEG_INF, std::memory_order_relaxed);
        update_count[i].store(0, std::memory_order_relaxed);
        in_queue[i].store(0, std::memory_order_relaxed);
        visited[i].store(0, std::memory_order_relaxed);
    }

    std::vector<N> frontier;
    frontier.reserve(roots.size());
    for (const auto& root : roots) {
        int c = root.code;
        dist[c].store(0, std::memory_order_relaxed);
        visited[c].store(1, std::memory_order_relaxed);
        num_discovered.fetch_add(1, std::memory_order_relaxed);
        in_queue[c].store(1, std::memory_order_relaxed);
        frontier.push_back(root);
    }

    std::vector<std::vector<N>> local_next(nthreads);

    auto extract_dist = [&]() {
        std::vector<int> result(max_code);
        for (int i = 0; i < max_code; ++i)
            result[i] = dist[i].load(std::memory_order_relaxed);
        return result;
    };

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

                int du      = dist[u.code].load(std::memory_order_relaxed);
                int new_val = du - weight;

                if (atomic_relax_max(dist[c_v], new_val)) {
                    parent_node[c_v] = u;
                    parent_edge[c_v] = edge;

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

        if (cycle_found.load(std::memory_order_relaxed)) {
            std::cerr << "There is a negative cycle";
            return extract_dist();
        }
    }
    return extract_dist();
}

// ---------------------------------------------------------------------------
// Non-hot-path helpers — left sequential
// ---------------------------------------------------------------------------

template<typename N>
bool check_transition_consistency(
    const std::vector<N>& roots,
    const Algorithm<N>& forward_alg,
    const Algorithm<N>& backward_alg
) {
    std::queue<N> q;
    constexpr int max_code = 1 << N::size;
    std::vector<bool> visited(max_code, false);

    std::cerr << "=== Backward Consistency Check ===\n";
    for (const auto& root : roots) {
        if (visited[root.code]) continue;
        q.push(root);
        visited[root.code] = true;
        while (!q.empty()) {
            N v = q.front(); q.pop();
            for (const auto& e : backward_alg.transition(v)) {
                N u = e.get_state();
                if (!visited[u.code]) {
                    visited[u.code] = true;
                    q.push(u);
                }
                bool found = false;
                for (const auto& fwd_e : forward_alg.transition(u)) {
                    if (fwd_e.get_state().code == v.code) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "[Extraneous backward] Incoming edge from:\n";
                    u.print(std::cerr);
                    std::cerr << "  to:\n";
                    v.print(std::cerr);
                    std::cerr << " with flip: " << e.get_flip();
                    return false;
                }
            }
        }
    }

    std::cerr << "=== Forward Consistency Check ===\n";
    std::fill(visited.begin(), visited.end(), false);
    for (const auto& root : roots) {
        if (visited[root.code]) continue;
        q.push(root);
        visited[root.code] = true;
        while (!q.empty()) {
            N u = q.front(); q.pop();
            for (const auto& e : forward_alg.transition(u)) {
                N v = e.get_state();
                if (!visited[v.code]) {
                    visited[v.code] = true;
                    q.push(v);
                }
                bool found = false;
                for (const auto& back_e : backward_alg.transition(v)) {
                    if (back_e.get_state().code == u.code) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "[Missing backward] Edge from:\n";
                    u.print(std::cerr);
                    std::cerr << "  to:\n";
                    v.print(std::cerr);
                    std::cerr << " with flip: " << e.get_flip();
                    return false;
                }
            }
        }
    }

    std::cerr << "Transition consistency check completed.\n";
    return true;
}

template<typename N>
bool check_transition_consistency_full(
    const std::vector<N>& roots,
    const Algorithm<N>& forward_alg,
    const Algorithm<N>& backward_alg
) {
    std::queue<N> q;
    constexpr int max_code = 1 << N::size;
    std::vector<bool> visited(max_code, false);

    std::cerr << "=== Backward Consistency Check ===\n";
    for (const auto& root : roots) {
        if (visited[root.code]) continue;
        q.push(root);
        visited[root.code] = true;
        while (!q.empty()) {
            N v = q.front(); q.pop();
            for (const auto& e : backward_alg.transition(v)) {
                N u = e.get_state();
                if (!visited[u.code]) {
                    visited[u.code] = true;
                    q.push(u);
                }
                bool found = false;
                for (const auto& fwd_e : forward_alg.transition(u)) {
                    if (fwd_e.get_state().code == v.code &&
                        fwd_e.get_opt() == e.get_opt() &&
                        fwd_e.get_online() == e.get_online() &&
                        fwd_e.get_flip() == e.get_flip()) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "[Extraneous backward] Incoming edge from:\n";
                    u.print(std::cerr);
                    std::cerr << "  to:\n";
                    v.print(std::cerr);
                    std::cerr << " with flip: " << e.get_flip()
                              << ", opt: " << e.get_opt()
                              << ", online: " << e.get_online() << "\n";
                    return false;
                }
            }
        }
    }

    std::cerr << "=== Forward Consistency Check ===\n";
    std::fill(visited.begin(), visited.end(), false);
    for (const auto& root : roots) {
        if (visited[root.code]) continue;
        q.push(root);
        visited[root.code] = true;
        while (!q.empty()) {
            N u = q.front(); q.pop();
            for (const auto& e : forward_alg.transition(u)) {
                N v = e.get_state();
                if (!visited[v.code]) {
                    visited[v.code] = true;
                    q.push(v);
                }
                bool found = false;
                for (const auto& back_e : backward_alg.transition(v)) {
                    if (back_e.get_state().code == u.code &&
                        back_e.get_opt() == e.get_opt() &&
                        back_e.get_online() == e.get_online() &&
                        back_e.get_flip() == e.get_flip()) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "[Missing backward] Edge from:\n";
                    u.print(std::cerr);
                    std::cerr << "  to:\n";
                    v.print(std::cerr);
                    std::cerr << " with flip: " << e.get_flip()
                              << ", opt: " << e.get_opt()
                              << ", online: " << e.get_online() << "\n";
                    return false;
                }
            }
        }
    }

    std::cerr << "Transition consistency check completed.\n";
    return true;
}

// Single-alg overload: uses alg.transition() and alg.reverse_transition()
template<typename N>
bool check_transition_consistency_full(
    const std::vector<N>& roots,
    const Algorithm<N>& alg
) {
    std::queue<N> q;
    constexpr int max_code = 1 << N::size;
    std::vector<bool> visited(max_code, false);

    std::cerr << "=== Backward Consistency Check ===\n";
    for (const auto& root : roots) {
        if (visited[root.code]) continue;
        q.push(root);
        visited[root.code] = true;
        while (!q.empty()) {
            N v = q.front(); q.pop();
            for (const auto& e : alg.reverse_transition(v)) {
                N u = e.get_state();
                if (!visited[u.code]) {
                    visited[u.code] = true;
                    q.push(u);
                }
                bool found = false;
                for (const auto& fwd_e : alg.transition(u)) {
                    if (fwd_e.get_state().code == v.code &&
                        fwd_e.get_opt() == e.get_opt() &&
                        fwd_e.get_online() == e.get_online() &&
                        fwd_e.get_flip() == e.get_flip()) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "[Extraneous backward] Incoming edge from:\n";
                    u.print(std::cerr);
                    std::cerr << "  to:\n";
                    v.print(std::cerr);
                    std::cerr << " with flip: " << e.get_flip()
                              << ", opt: " << e.get_opt()
                              << ", online: " << e.get_online() << "\n";
                    return false;
                }
            }
        }
    }

    std::cerr << "=== Forward Consistency Check ===\n";
    std::fill(visited.begin(), visited.end(), false);
    for (const auto& root : roots) {
        if (visited[root.code]) continue;
        q.push(root);
        visited[root.code] = true;
        while (!q.empty()) {
            N u = q.front(); q.pop();
            for (const auto& e : alg.transition(u)) {
                N v = e.get_state();
                if (!visited[v.code]) {
                    visited[v.code] = true;
                    q.push(v);
                }
                bool found = false;
                for (const auto& back_e : alg.reverse_transition(v)) {
                    if (back_e.get_state().code == u.code &&
                        back_e.get_opt() == e.get_opt() &&
                        back_e.get_online() == e.get_online() &&
                        back_e.get_flip() == e.get_flip()) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "[Missing backward] Edge from:\n";
                    u.print(std::cerr);
                    std::cerr << "  to:\n";
                    v.print(std::cerr);
                    std::cerr << " with flip: " << e.get_flip()
                              << ", opt: " << e.get_opt()
                              << ", online: " << e.get_online() << "\n";
                    return false;
                }
            }
        }
    }

    std::cerr << "Transition consistency check completed.\n";
    return true;
}

template<typename N>
std::vector<N> bfs_reachable_nodes(
    const std::vector<N>& roots,
    const Algorithm<N>& alg
) {
    constexpr int max_code = 1 << N::size;
    std::vector<uint8_t> visited(max_code, false);
    std::vector<N> discovered;
    std::queue<N> q;

    for (const auto& root : roots) {
        if (!visited[root.code]) {
            visited[root.code] = true;
            discovered.push_back(root);
            q.push(root);
        }
    }

    while (!q.empty()) {
        N u = q.front(); q.pop();
        for (const auto& edge : alg.transition(u)) {
            const N& v = edge.get_state();
            if (!visited[v.code]) {
                visited[v.code] = true;
                discovered.push_back(v);
                q.push(v);
            }
        }
    }

    return discovered;
}

template<typename N>
void save_distances_csv(
    const std::vector<int>& dist,
    const std::vector<N>& reachable_nodes,
    int lamb,
    int beta,
    const std::string& filename,
    bool reverse = false
) {
    std::ofstream out(filename);
    for (const auto& h : N::csv_header()) out << h << ",";
    out << "distance\n";

    double scale = static_cast<double>(beta) * lamb;
    if (!reverse) {
        // find min/max among finite distances (skip INF); normalize by shifting
        int min_d = 0, max_d = 0;
        bool found = false;
        for (const auto& node : reachable_nodes) {
            int d = dist[node.code];
            if (d == INF) continue;
            if (!found) { min_d = max_d = d; found = true; }
            else { min_d = std::min(min_d, d); max_d = std::max(max_d, d); }
        }
        std::cout << "Forward distances (raw): min=" << min_d << ", max=" << max_d << "\n";
        for (const auto& node : reachable_nodes) {
            int d = dist[node.code];
            if (d == INF) continue;
            for (auto v : node.csv_values(lamb)) out << v << ",";
            out << static_cast<double>(d - min_d) / scale << "\n";
        }
    } else {
        // reverse distances are already >= 0; output raw scaled values
        for (const auto& node : reachable_nodes) {
            int d = dist[node.code];
            if (d == std::numeric_limits<int>::min()) continue;
            for (auto v : node.csv_values(lamb)) out << v << ",";
            out << static_cast<double>(d) / scale << "\n";
        }
    }
}
