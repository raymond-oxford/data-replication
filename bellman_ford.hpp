#pragma once
#include "node.hpp"
#include "alg.hpp"
#include <queue>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

constexpr int INF = std::numeric_limits<int>::max();

template<int Lambda>
struct CycleStep {
    Node<Lambda> node;
    Transition<Lambda> edge;  // specify template argument

    CycleStep(const Node<Lambda>& n, const Transition<Lambda>& e) : node(n), edge(e) {}

    void print(std::ostream& out) const {
            out << "CycleStep:\n  Node: ";
            node.print(out);
            out << "  Edge: ";
            edge.print(out);
        }
};

template<int Lambda>
std::pair<int, int> compute_cycle_totals(const std::vector<CycleStep<Lambda>>& cycle) {
    int total_opt = 0;
    int total_online = 0;

    for (const auto& step : cycle) {
        total_opt += step.edge.get_opt();
        total_online += step.edge.get_online();
    }

    return {total_opt, total_online};
}

template<int Lambda>
std::vector<CycleStep<Lambda>> find_negative_cycle(
    const std::vector<Node<Lambda>>& roots,
    const Algorithm<Lambda>& alg,
    int alpha,
    int beta
) {
    using N = Node<Lambda>;
    constexpr int max_code = 1 << (2 * N::L + 3);
    int num_discovered = 0;
    int i = 0;
    int size = INF;

    std::vector<int> dist(max_code, INF);
    std::vector<uint32_t> update_count(max_code, 0);
    std::vector<uint8_t> in_queue(max_code, false);
    std::vector<uint8_t> visited(max_code, false);
    std::vector<N> parent_node(max_code);                  // Store predecessor node
    std::vector<Transition<Lambda>> parent_edge(max_code); // Store edge leading to node

    std::queue<Node<Lambda>> q1, q2;

    // Initialize distances and stack with roots
    for (const auto& root : roots) {
        int c = root.code;
        dist[c] = 0;
        visited[c] = true;
        ++num_discovered;
        q1.push(root);
        in_queue[c] = true;
    }

    N cycle_head = N();

    while (!q1.empty() || !q2.empty()) {
      while (!q1.empty()) {
          N u = q1.front(); q1.pop();
          in_queue[u.code] = false;

          for (const auto& edge : alg.transition(u)) {
              int weight = alpha * edge.get_opt() - beta * edge.get_online();
              const N& v = edge.get_state();
              int c_v = v.code;

              if (!visited[c_v]) {
                  visited[c_v] = true;
                  ++num_discovered;
              }

              int new_dist = dist[u.code] + weight;
              if (new_dist < dist[c_v]) {
                  dist[c_v] = new_dist;
                  parent_node[c_v] = u;
                  parent_edge[c_v] = edge;
                  update_count[c_v]++;

                  if (!in_queue[c_v]) {
                      q2.push(v);
                      in_queue[c_v] = true;
                  }

                  if (update_count[c_v] >= num_discovered) {
                      // Negative cycle detected: reconstruct cycle
                      N x = v;
                      for (int i = 0; i < num_discovered; ++i)
                          x = parent_node[x.code];
                      cycle_head = x;
                      break;
                  }

              }
          }
          if (cycle_head.is_valid()) {
            while (!q1.empty()) {
                q2.push(q1.front());
                q1.pop();
            }
            break;
          }
      }
      std::swap(q1, q2);
      i++;
      if (q1.size() > size && !cycle_head.is_valid()) {
          const N candidate = q1.front();
          N cycle_node = candidate;
         for (int i = 0; i < num_discovered; ++i) {
            cycle_node = parent_node[cycle_node.code];
            if (!cycle_node.is_valid())
              break;
         }
         if (cycle_node.is_valid()) {
            cycle_head = cycle_node;
         }
      }
      if (cycle_head.is_valid()) {
          std::vector<CycleStep<Lambda>> cycle;
          N cur = cycle_head;
          do {
              N par = parent_node[cur.code];
              Transition e = parent_edge[cur.code];
              cycle.emplace_back(par, e);
              cur = par;
          } while (cur != cycle_head);
          std::reverse(cycle.begin(), cycle.end());
          assert (q2.empty());
          return cycle;
      }
      size = q1.size();
    }
    return {}; // no negative cycle
}
