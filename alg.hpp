#pragma once
#include "node.hpp"
#include <vector>
#include <tuple>
#include <cassert>
#include <algorithm>
#include <iostream>

// Transition struct is now global (outside the class)
template<int Lambda>
struct Transition {
    uint64_t code;

    static constexpr int L = Node<Lambda>::L;
    static constexpr int NODE_BITS   = 2 * L + 3;
    static constexpr int FLIP_BITS   = 1;
    static constexpr int OPT_BITS    = 15; // signed
    static constexpr int ONLINE_BITS = 15;

    static_assert(NODE_BITS + FLIP_BITS + OPT_BITS + ONLINE_BITS <= 64,
                  "Transition encoding won't fit in uint64_t");

    Transition() : code(0) {}

    Transition(int flip, int opt, int online, Node<Lambda> state) {
        assert(flip == 0 || flip == 1);
        assert(opt >= -(1 << (OPT_BITS - 1)) && opt < (1 << (OPT_BITS - 1)));
        assert(online >= 0 && online < (1 << ONLINE_BITS));

        uint64_t opt_u = static_cast<uint64_t>(static_cast<int64_t>(opt)) & ((1ULL << OPT_BITS) - 1);

        code = 0;
        code |= static_cast<uint64_t>(state.code);
        code |= static_cast<uint64_t>(flip)   << NODE_BITS;
        code |= opt_u                         << (NODE_BITS + FLIP_BITS);
        code |= static_cast<uint64_t>(online) << (NODE_BITS + FLIP_BITS + OPT_BITS);
    }

    int get_flip() const {
        return (code >> NODE_BITS) & 1;
    }

    int get_online() const {
        return (code >> (NODE_BITS + FLIP_BITS + OPT_BITS)) & ((1 << ONLINE_BITS) - 1);
    }

    int get_opt() const {
        int32_t raw = (code >> (NODE_BITS + FLIP_BITS)) & ((1 << OPT_BITS) - 1);
        // Sign-extend manually
        if (raw & (1 << (OPT_BITS - 1))) {
            raw |= ~((1 << OPT_BITS) - 1);
        }
        return raw;
    }

    Node<Lambda> get_state() const {
        return Node<Lambda>(code & ((1ULL << NODE_BITS) - 1));
    }

    void print(std::ostream& out) const {
        out << "Transition - flip: " << get_flip()
                  << ", opt: " << get_opt()
                  << ", online: " << get_online()
                  << ", state: ";
        get_state().print(out);
    }

    bool operator==(const Transition<Lambda>& other) const {
        return get_flip() == other.get_flip() &&
               get_opt() == other.get_opt() &&
               get_online() == other.get_online() &&
               get_state().code == other.get_state().code;
    }

};
namespace std {
    template<int Lambda>
    struct hash<Transition<Lambda>> {
        size_t operator()(const Transition<Lambda>& t) const {
            return std::hash<int>()(t.get_flip())
                 ^ (std::hash<int>()(t.get_opt()) << 1)
                 ^ (std::hash<int>()(t.get_online()) << 2)
                 ^ (std::hash<int>()(t.get_state().code) << 3);
        }
    };
}

template<int Lambda>
class Algorithm {
public:
    using NodeType = Node<Lambda>;
    virtual ~Algorithm() = default;

    virtual std::vector<Transition<Lambda>> predecessors_as_reversed_transitions(const Node<Lambda>& next_state) const {
      return {};
    }
    virtual std::vector<Transition<Lambda>> transition(const NodeType& curr) const = 0;
    virtual std::vector<Node<Lambda>> initial_nodes() const = 0;
    virtual std::vector<Node<Lambda>> initial_nodes_small() const = 0;
};

template<int Lambda>
class TAlgorithm : public Algorithm<Lambda> {
public:
    using NodeType = Node<Lambda>;
    static constexpr int lamb = Lambda;
    int threshold;
    int thresh;

    TAlgorithm(int t, int th) : threshold(t), thresh(th) {}

    std::vector<Transition<Lambda>> transition(const NodeType& curr) const {
        std::vector<Transition<Lambda>> transitions;
        auto next_reqs = next_requests(curr);
        for (const auto& [time, location, pred] : next_reqs) {
            auto [costs, next_state] = choose_config(curr, time, location, pred);
            transitions.emplace_back(location, costs.first, costs.second, std::move(next_state));
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
      nodes.emplace_back(1, Lambda, 0, 0, Lambda);
      return nodes;
    }

private:
    std::vector<std::tuple<int,int,int>> next_requests(const NodeType& curr) const {
        std::vector<std::tuple<int,int,int>> requests;

        bool top_pred = curr.get_pred1();
        bool bot_pred = curr.get_pred2();

        if (bot_pred) {
            int limit = lamb - curr.get_dist() + 1;
            for (int time = 1; time < limit; ++time) {
                requests.emplace_back(time, 1, 0);
                requests.emplace_back(time, 1, 1);
                if (top_pred) {
                    requests.emplace_back(time, 0, 0);
                    requests.emplace_back(time, 0, 1);
                }
            }
        } else {
            if (top_pred) {
                for (int time = 1; time <= lamb; ++time) {
                    requests.emplace_back(time, 0, 0);
                    requests.emplace_back(time, 0, 1);
                }
                int start_time = std::max(lamb - curr.get_dist() + 1, 1);
                for (int time = start_time; time < lamb; ++time) {
                    requests.emplace_back(time, 1, 0);
                    requests.emplace_back(time, 1, 1);
                }
            } else {
                requests.emplace_back(lamb + 1, 0, 0);
                requests.emplace_back(lamb + 1, 0, 1);

                int start_time = std::max(lamb - curr.get_dist() + 1, 1);
                for (int time = start_time; time <= lamb + 1; ++time) {
                    requests.emplace_back(time, 1, 0);
                    requests.emplace_back(time, 1, 1);
                }
            }
        }
        return requests;
    }

    std::pair<std::pair<int,int>, NodeType> choose_config(const NodeType& curr, int time, int location, int pred) const {
        assert(time > 0);
        assert(curr.get_work() >= 0 && curr.get_work() <= lamb);
        assert(curr.get_dist() >= 1 && curr.get_dist() <= lamb + 1);

        int config = curr.get_config();
        int dist = curr.get_dist();
        int work = curr.get_work();
        int pred1 = curr.get_pred1();
        int pred2 = curr.get_pred2();

        std::pair<int,int> next_P;
        if (location == 0) {
            next_P = {pred, pred2};
        } else {
            next_P = {pred, pred1};
        }

        int next_dist, next_w;
        if (location == 0) {
            next_dist = std::min(lamb + 1, dist + time);
            int one_w = time;
            int both_w = std::min(time + lamb, work + 2 * time);
            next_w = both_w - one_w;
        } else {
            next_dist = time;
            int one_w = std::min(time + lamb, work + time);
            int both_w = std::min(work + 2 * time, time + lamb);
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
                online = time + lamb;
                next_C = 1;
            }
        } else {
            if (!pred1 && !pred2) {
                assert(dist - work >= 0);
                bool stay = (std::min(dist, lamb) - work > threshold);
                if ((location ^ stay) != 0) {
                    online = time + lamb;
                    next_C = 1;
                } else {
                    online = time;
                    next_C = 0;
                }
            } else if (!pred1 && pred2) {
                assert(location == 1);
                online = time;
                next_C = 0;
            } else if (pred1 && !pred2) {
                if (std::min(dist, lamb) - work <= thresh) {
                    online = time;
                    next_C = 0;
                    if (location == 1) {
                        online += lamb;
                        next_C = 1;
                    }
                } else if (std::min(dist, lamb) - work >= thresh + time) {
                    online = 2 * time;
                    next_C = 1;
                } else {
                    online = time + (std::min(dist, lamb) - work - thresh);
                    next_C = 0;
                    if (location == 1) {
                        online += lamb;
                        next_C = 1;
                    }
                }
            } else {
                online = 2 * time;
                next_C = 1;
            }
        }

        int opt_curr = (config == 0) ? 0 : work;
        int one_w = 0, both_w = 0;
        if (location == 0) {
            one_w = time;
            both_w = std::min(time + lamb, work + 2 * time);
        } else {
            one_w = std::min(time + lamb, work + time);
            both_w = std::min(work + 2 * time, time + lamb);
        }

        // int opt_next = (next_C == 0) ? one_w : both_w;
        // int opt = opt_next - opt_curr;
        int opt = one_w;

        NodeType next_state(next_C, next_dist, next_P.first, next_P.second, next_w);
        return {{opt, online}, next_state};
    }
};
