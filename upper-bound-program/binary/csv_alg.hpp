#pragma once
#include "alg.hpp"
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

struct PolicyKey {
    std::string mode;
    double d;
    double w;

    bool operator<(const PolicyKey& other) const {
        if (mode != other.mode) return mode < other.mode;
        if (d != other.d) return d < other.d;
        return w < other.w;
    }
};

struct PolicyValue {
    std::string action;  // e.g. "top", "bottom", "double", "hybrid"
    double y_val;        // -1 if not applicable
};

static std::map<PolicyKey, PolicyValue> policy_table;

inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end   = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

inline double round6(double x) {
    return std::round(x * 1e6) / 1e6;
}

static void load_policy_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open " + filename);

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

template<int Lambda>
class CSVAlgorithm : public BinaryAlgorithm<Lambda> {
public:
    using NodeType = Node<Lambda>;

    CSVAlgorithm(const std::string& csv_file, int csv_grid = 50, bool bound_mode = false)
        : BinaryAlgorithm<Lambda>(bound_mode), csv_grid_(csv_grid) {
        if (policy_table.empty()) load_policy_csv(csv_file);
    }

    std::vector<Node<Lambda>> initial_nodes() const override {
        // return { Node<Lambda>(0, Lambda, 0, 0, Lambda) };
        std::vector<NodeType> nodes;
        for (int c = 0; c <= 1; c++) {
            for (int d = 0; d <= Lambda; d++) {
                for (int w = 0; w <= d; w++) {
                    for (int pred1 = 0; pred1 <= 1; pred1++) {
                        for (int pred2 = 0; pred2 <= 1; pred2++) {
                            nodes.emplace_back(c, d, pred1, pred2, w);
                        }
                    }
                }
            }
        }
        return nodes;
    }

    bool get_stay(int dist, int work) const override {
        double d = std::min(1.0, static_cast<double>(dist) / Lambda);
        double w = static_cast<double>(work) / Lambda;
        d = snap_to_csv_grid(d);
        w = snap_to_csv_grid(w);
        PolicyKey key{"LTP", d, w};
        auto it = policy_table.find(key);
        if (it == policy_table.end()) {
            std::ostringstream oss;
            oss << "LTP policy not found for d=" << d << ", w=" << w;
            throw std::runtime_error(oss.str());
        }
        return it->second.action == "bottom" || it->second.action == "1";
    }

    double get_y(int dist, int work) const override {
        double d = std::min(1.0, static_cast<double>(dist) / Lambda);
        double w = static_cast<double>(work) / Lambda;
        d = snap_to_csv_grid(d);
        w = snap_to_csv_grid(w);
        PolicyKey key{"STP", d, w};
        auto it = policy_table.find(key);
        if (it == policy_table.end()) throw std::runtime_error("STP policy not found");
        return it->second.y_val;
    }

private:
    int csv_grid_;

    double snap_to_csv_grid(double x) const {
        return std::round(x * csv_grid_) / csv_grid_;
    }
};
