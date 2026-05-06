#include "../bellman_ford.hpp"
#include "csv_alg.hpp"
#include "naive_alg.hpp"
#include "analytical_alg.hpp"

#include <iostream>
#include <vector>
#include <string>

#ifndef LAMBDA_VAL
#define LAMBDA_VAL 100
#endif

using N = Node<LAMBDA_VAL>;

int find_worst_cycle(Algorithm<N>& alg,
                     const std::string& save_path = "",
                     bool reverse = false,
                     bool all_roots = false) {
    constexpr int Lambda = LAMBDA_VAL;
    int alpha = 3;
    int beta = 2;

    auto roots = alg.initial_nodes();
    std::vector<CycleStep<N>> prev_cycle;

    while (true) {
        auto [cycle, cached_dist] = find_negative_cycle(roots, alg, alpha, beta);

        if (!cycle.empty()) {
            std::cout << "Negative cycle found:\n";
            auto [total_opt, total_online] = compute_cycle_totals(cycle);
            std::cout << "The ratio is " << total_online << "/" << total_opt << " = " << static_cast<double>(total_online) / total_opt << '\n';
            alpha = total_online;
            beta = total_opt;
            prev_cycle = cycle;
        } else {
            std::cout << "No negative cycle found.\n";
            std::cout << "The ratio is " << alpha << "/" << beta << " = " << static_cast<double>(alpha) / beta << '\n';
            for (const auto& step : prev_cycle) {
                step.print(std::cout);
            }
            if (!save_path.empty()) {
                auto reachable = bfs_reachable_nodes(roots, alg);
                std::vector<int> d;
                if (!reverse && !all_roots) {
                    d = std::move(cached_dist);
                } else {
                    auto bf_roots = all_roots ? reachable : roots;
                    d = reverse
                        ? compute_distances_reverse(bf_roots, alg, alpha, beta)
                        : compute_distances(bf_roots, alg, alpha, beta);
                }
                save_distances_csv(d, reachable, Lambda, beta, save_path, reverse);
                std::cout << "Distances saved to " << save_path << "\n";
            }
            return 0;
        }
    }
}

int main(int argc, char* argv[]) {
    constexpr int Lambda = LAMBDA_VAL;
    std::string csv_policy;
    std::string save_path;

    bool reverse = false;
    bool all_roots = false;
    bool bound_mode = false;
    // bool use_naive = false;
    bool use_naive = true;
    int csv_grid = 50;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            save_path = argv[++i];
        } else if (arg == "--reverse") {
            reverse = true;
        } else if (arg == "--all-roots") {
            all_roots = true;
        } else if (arg == "--bound-mode") {
            bound_mode = true;
        } else if (arg == "--naive") {
            use_naive = true;
        } else if (arg == "--csv-grid" && i + 1 < argc) {
            csv_grid = std::stoi(argv[++i]);
        } else {
            csv_policy = arg;
        }
    }

    if (csv_policy.empty()) {
        if (use_naive) {
            NaiveAlgorithm<Lambda> alg(bound_mode);
            return find_worst_cycle(alg, save_path, reverse, all_roots);
        }
        AnalyticalAlgorithm<Lambda> alg(bound_mode);
        return find_worst_cycle(alg, save_path, reverse, all_roots);
    }
    CSVAlgorithm<Lambda> alg(csv_policy, csv_grid, bound_mode);
    return find_worst_cycle(alg, save_path, reverse, all_roots);
}
