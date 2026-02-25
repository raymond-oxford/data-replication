#include "bellman_ford.hpp"
#include "csv_alg.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <string>


int find_worst_cycle() {
    constexpr int Lambda = 2000;
    // int alpha = 410;
    // int beta = 265;
    // int alpha = 2048;
    // int beta = 1322;
    int alpha = 31;
    int beta = 20;

    CSVAlgorithm<Lambda> alg("policy-501.csv");
    auto roots = alg.initial_nodes_small();
    std::vector<CycleStep<Lambda>> prev_cycle;

    while (true) {
      std::vector<CycleStep<Lambda>> cycle = find_negative_cycle(roots, alg, alpha, beta);

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
          return 0;
      }
    }
}

int main () {
  find_worst_cycle();
  return 0;
}
