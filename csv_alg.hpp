#pragma once
#include "alg.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>

struct PolicyKey {
  std::string mode;
  double d;
  double w;

  bool operator<(const PolicyKey &other) const {
    if (mode != other.mode)
      return mode < other.mode;
    if (d != other.d)
      return d < other.d;
    return w < other.w;
  }
};

struct PolicyValue {
  std::string action; // e.g. "top", "bottom", "double", "hybrid"
  double y_val;       // -1 if not applicable
};

static std::map<PolicyKey, PolicyValue> policy_table;

inline std::string trim(const std::string &s) {
  size_t start = s.find_first_not_of(" \t\r\n");
  size_t end = s.find_last_not_of(" \t\r\n");
  return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

inline double round6(double x) { return std::round(x * 1e6) / 1e6; }

static void load_policy_csv(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("Could not open " + filename);

  std::string line;
  std::getline(file, line); // skip header

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string mode, d_str, w_str, action, y_str;
    std::getline(ss, mode, ',');
    std::getline(ss, d_str, ',');
    std::getline(ss, w_str, ',');
    std::getline(ss, action, ',');
    std::getline(ss, y_str, ',');
    y_str = trim(y_str);

    double d = std::stod(d_str);
    d = round6(d);
    double w = std::stod(w_str);
    w = round6(w);

    PolicyKey key{mode, d, w};
    PolicyValue val{action, y_str.empty() ? -1.0 : round6(std::stod(y_str))};
    policy_table[key] = val;
  }
}

template <int Lambda> class CSVAlgorithm : public Algorithm<Lambda> {
public:
  using NodeType = Node<Lambda>;
  static constexpr int lamb = Lambda;

  CSVAlgorithm(const std::string &csv_file) {
    if (policy_table.empty()) {
      load_policy_csv(csv_file);
    }
  }

  std::vector<Transition<Lambda>> transition(const NodeType &curr) const {
    std::vector<Transition<Lambda>> transitions;
    auto next_reqs = next_requests(curr);
    for (const auto &[time, location, pred] : next_reqs) {
      auto [costs, next_state] = choose_config(curr, time, location, pred);
      transitions.emplace_back(location, costs.first, costs.second,
                               std::move(next_state));
    }
    return transitions;
  }

  std::vector<Node<Lambda>> initial_nodes() const {
    std::vector<Node<Lambda>> nodes;

    for (int time = 1; time <= Lambda + 1; ++time) {
      nodes.emplace_back(1, time, 0, 0, 0);
      nodes.emplace_back(1, time, 1, 0, 0);
    }

    for (int time = 1; time < Lambda; ++time) {
      nodes.emplace_back(1, time, 0, 1, 0);
      nodes.emplace_back(1, time, 1, 1, 0);
    }

    return nodes;
  }

  std::vector<Node<Lambda>> initial_nodes_small() const {
    std::vector<Node<Lambda>> nodes;
    nodes.emplace_back(0, Lambda, 0, 0, Lambda);
    return nodes;
  }

  bool get_stay_from_csv(int dist, int work) const {
    double d = static_cast<double>(dist) / lamb;
    if (d > 1) {
      d = 1;
    }
    d = round6(d);
    double w = static_cast<double>(work) / lamb;
    w = round6(w);
    PolicyKey key{"LTP", d, w};
    auto it = policy_table.find(key);
    if (it == policy_table.end()) {
      std::ostringstream oss;
      oss << "LTP policy not found for d=" << d << ", w=" << w;
      throw std::runtime_error(oss.str());
    }
    return it->second.action == "bottom";
  }

  std::vector<std::tuple<int, int, int>>
  next_requests(const NodeType &curr) const {
    std::vector<std::tuple<int, int, int>> requests;

    bool top_pred = curr.get_pred1();
    bool bot_pred = curr.get_pred2();

    if (bot_pred) {
      int limit = lamb - curr.get_dist() + 1;
      for (int time = 0; time < limit; ++time) {
        requests.emplace_back(time, 1, 0);
        requests.emplace_back(time, 1, 1);
        if (top_pred) {
          requests.emplace_back(time, 0, 0);
          requests.emplace_back(time, 0, 1);
        }
      }
    } else {
      if (top_pred) {
        for (int time = 0; time <= lamb; ++time) {
          requests.emplace_back(time, 0, 0);
          requests.emplace_back(time, 0, 1);
        }
        int start_time =
            std::max(lamb - curr.get_dist(),
                     0); // changed this line from lambda - curr.get_dist() + 1
        for (int time = start_time; time <= lamb + 1;
             ++time) { // changed this line from < lamb
          requests.emplace_back(time, 1, 0);
          requests.emplace_back(time, 1, 1);
        }
      } else {
        requests.emplace_back(lamb + 1, 0, 0);
        requests.emplace_back(lamb + 1, 0, 1);
        requests.emplace_back(lamb, 0, 0); // added this line
        requests.emplace_back(lamb, 0, 1); // added this line

        int start_time =
            std::max(lamb - curr.get_dist(),
                     0); // changed this line from lambda - curr.get_dist() + 1
        for (int time = start_time; time <= lamb + 1; ++time) {
          requests.emplace_back(time, 1, 0);
          requests.emplace_back(time, 1, 1);
        }
      }
    }
    return requests;
  }

private:
  double get_y_from_csv(int dist, int work) const {
    double d = static_cast<double>(dist) / lamb;
    if (d > 1) {
      d = 1;
    }
    d = round6(d);
    double w = static_cast<double>(work) / lamb;
    w = round6(w);
    PolicyKey key{"STP", d, w};
    auto it = policy_table.find(key);
    if (it == policy_table.end())
      throw std::runtime_error("STP policy not found");
    return it->second.y_val;
  }

  std::pair<std::pair<int, int>, NodeType>
  choose_config(const NodeType &curr, int time, int location, int pred) const {
    // assert(time > 0);
    assert (time >= 0);
    assert(curr.get_work() >= 0 && curr.get_work() <= lamb);
    // assert(curr.get_dist() >= 1 && curr.get_dist() <= lamb + 1);
    assert(curr.get_dist() >= 0 && curr.get_dist() <= lamb);

    int config = curr.get_config();
    int dist = curr.get_dist();
    int work = curr.get_work();
    int pred1 = curr.get_pred1();
    int pred2 = curr.get_pred2();

    std::pair<int, int> next_P;
    if (location == 0) {
      next_P = {pred, pred2};
    } else {
      next_P = {pred, pred1};
    }

    int next_dist, next_w, one_w, both_w;
    if (location == 0) {
      next_dist = std::min(lamb, dist + time);
      one_w = time;
      both_w = std::min(time + lamb - 1, work + 2 * time); // changed this line from time + lamb
      next_w = both_w - one_w;
    } else {
      next_dist = std::min(time, lamb);
      one_w = std::min(time + lamb - 1, work + time); // changed this line from time + lamb
      both_w = std::min(work + 2 * time, time + lamb - 1); // changed this line from time + lamb
      next_w = both_w - one_w;
    }

    int online = 0;
    int next_C = 0;

    if (config == 0) {
      assert(pred2 == 0);
      if (location == 0) {
        online = time;
        next_C = 0;
      } else {
        online = time + lamb + 1; // changed this line from time + lamb
        next_C = 1;
      }
    } else {
      if (!pred1 && !pred2) {
        assert(dist - work >= 0);
        bool stay = get_stay_from_csv(dist, work);
        if ((location ^ stay) != 0) {
          online = time + lamb + 1; // changed this line from time + lamb
          next_C = 1;
        } else {
          online = time;
          next_C = 0;
        }
      } else if (!pred1 && pred2) {
        if (location != 1) {
          curr.print(std::cerr);
        }
        assert(location == 1);
        online = time;
        next_C = 0;
      } else if (pred1 && !pred2) {
        double csv_y = get_y_from_csv(dist, work);

        if (csv_y * lamb >= time) {
          online = 2 * time;
          next_C = 1;
        } else {
          online = time + static_cast<int>(csv_y * lamb);
          next_C = 0;
          if (location == 1) {
            online += lamb + 1; // changed this line from lamb to lamb + 1
            next_C = 1;
          }
        }
      } else {
        online = 2 * time;
        next_C = 1;
      }
    }

    int opt = one_w;

    NodeType next_state(next_C, next_dist, next_P.first, next_P.second, next_w);
    return {{opt, online}, next_state};
  }
};
