#include "bellman_ford.h"
#include "game_graph.h"
#include "howard.h"
#include "solver.h"
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <string>

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <N> <p> <q> [--verbose]\n";
    std::cerr << "  N   : grid size (integer >= 1)\n";
    std::cerr << "  p,q : initial ratio p/q (positive integers)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) { usage(argv[0]); return 1; }

    int N       = std::stoi(argv[1]);
    long long p = std::stoll(argv[2]);
    long long q = std::stoll(argv[3]);
    bool verbose = (argc >= 5 && std::string(argv[4]) == "--verbose");

    if (N < 1 || p <= 0 || q <= 0) {
        std::cerr << "Error: N >= 1 and p, q > 0 required.\n";
        return 1;
    }

    std::cout << "Threads: " << omp_get_max_threads() << "\n";

    GameGraph g(N);
    g.build(p, q); // sets topology; node IDs are fixed henceforth

    int n = g.num_nodes();
    if (verbose)
        std::cout << "N=" << N << "  initial p/q=" << p << "/" << q
                  << "  nodes=" << n << "\n";

    std::vector<int> sigma(n, 0); // warm-start Max strategy (carried across iters)

    for (int iter = 1; iter <= 200; ++iter) {
        g.build(p, q); // recompute rewards for current (p, q)

        // Step 1: Howard's policy iteration (Min, fixed sigma) → gain + converged tau.
        auto hr     = run_howard(g, sigma);
        long long g_num = hr.g_num;
        long long g_den = hr.g_den;
        int       ref   = hr.ref;
        auto&     tau_h = hr.tau;

        if (verbose)
            std::cout << "iter " << iter << "  p/q=" << p << "/" << q
                      << "  g=" << g_num << "/" << g_den << "\n";

        // Step 2: if gain > 0 the current ratio is too small; pivot.
        if (g_num > 0) {
            auto [np, nq] = extract_cycle_ratio(g, sigma, tau_h, p, q);
            p = np; q = nq;
            continue;
        }

        // Step 3: compute integer bias via Bellman-Ford anchored at ref.
        auto H = run_bellman_ford(g, sigma, tau_h, g_num, g_den, ref);

        // Step 4: try to improve Max's strategy.
        bool improved = improve_max_policy(g, sigma, H, g_den);
        if (improved) continue;

        // No Max improvement possible.
        if (g_num == 0) {
            // Potential optimum — verify the certificate.
            if (verify_certificate(g, sigma, tau_h, 0, g_den, H)) {
                save_policy(g, sigma, tau_h, "policy.csv");
                std::cout << "p*/q* = " << p << "/" << q << "\n";
                std::cout << "Certificate verified.\n";
                return 0;
            } else {
                std::cerr << "Certificate verification FAILED at iter " << iter << ".\n";
                return 1;
            }
        }

        // g_num < 0: ratio is too large; pivot via cycle ratio extraction.
        auto [np, nq] = extract_cycle_ratio(g, sigma, tau_h, p, q);
        p = np; q = nq;
    }

    std::cerr << "Failed to converge in 200 iterations.\n";
    return 1;
}
