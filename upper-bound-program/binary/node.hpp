#pragma once
#include <cstdint>
#include <iostream>
#include <cassert>
#include <vector>
#include <string>

template<int Lambda>
struct Node {
    static constexpr int L = []() {
        // int l = 0, x = Lambda + 1;
        int l = 0, x = Lambda;
        while (x > 0) { ++l; x >>= 1; }
        return l;
    }();

    static constexpr uint32_t INVALID_NODE_CODE = 1u << 31;

    static constexpr int size = 2 * L + 3;
    static_assert(size <= 32, "Node encoding won't fit in uint32_t");

    uint32_t code;

    Node() : code(INVALID_NODE_CODE) {
        // code=0 means:
        // config = 0, dist = 1 (since dist is stored as dist-1),
        // pred1 = 0, pred2 = 0, work = 0
        // This is a valid default Node with minimal values.
    }

    explicit Node(uint32_t raw_code) : code(raw_code) {}

    // Constructor
    Node(int config, int dist, int pred1, int pred2, int work) {
        assert(config == 0 || config == 1);
        assert(pred1 == 0 || pred1 == 1);
        assert(pred2 == 0 || pred2 == 1);
        // assert(dist >= 1 && dist <= Lambda + 1);
        // assert(dist >= 1 && dist <= Lambda);
        assert(dist >= 0 && dist <= Lambda);
        assert(work >= 0 && work <= Lambda);

        code = 0;
        // code |= (dist - 1);
        code |= dist;
        code |= (work << L);
        code |= (pred1 << (2 * L));
        code |= (pred2 << (2 * L + 1));
        code |= (config << (2 * L + 2));
    }

    // Accessors
    int get_config() const { return (code >> (2 * L + 2)) & 1; }
    int get_pred1()  const { return (code >> (2 * L)) & 1; }
    int get_pred2()  const { return (code >> (2 * L + 1)) & 1; }
    // int get_dist()   const { return (code & ((1 << L) - 1)) + 1; }
    int get_dist() const { return code & ((1 << L) - 1); }
    int get_work()   const { return (code >> L) & ((1 << L) - 1); }
    bool is_valid() const {return (code & INVALID_NODE_CODE) == 0;}

    // Comparison (for map keys, etc.)
    bool operator==(const Node &other) const = default;
    bool operator<(const Node &other) const { return code < other.code; }

    static std::vector<std::string> csv_header() {
        return {"config", "d", "pred1", "pred2", "w"};
    }

    std::vector<double> csv_values(int lamb) const {
        return {
            (double)get_config(),
            (double)get_dist() / lamb,
            (double)get_pred1(),
            (double)get_pred2(),
            (double)get_work() / lamb,
        };
    }

    void print(std::ostream& out) const {
        out << "config: " << get_config()
                  << ", dist: " << get_dist()
                  << ", predictions: (" << get_pred1() << ", " << get_pred2() << ")"
                  << ", work: " << get_work() << "\n";
    }
};

namespace std {
    template<int Lambda>
    struct hash<Node<Lambda>> {
        std::size_t operator()(const Node<Lambda>& n) const {
            return std::hash<uint32_t>()(n.code);
        }
    };
}
