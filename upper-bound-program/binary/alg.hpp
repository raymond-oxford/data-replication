#pragma once
#include "node.hpp"
#include "../alg.hpp"
#include <cmath>
#include <iomanip>
#include <vector>
#include <tuple>
#include <cassert>
#include <algorithm>
#include <iostream>

template<int Lambda>
class BinaryAlgorithm : public Algorithm<Node<Lambda>> {
public:
    using NodeType = Node<Lambda>;
    using TransType = Transition<Node<Lambda>>;
    static constexpr int lamb = Lambda;

    explicit BinaryAlgorithm(bool bound_mode = false) : bound_mode_(bound_mode) {}
    virtual ~BinaryAlgorithm() = default;

    virtual bool get_stay(int dist, int work) const = 0;
    virtual double get_y(int dist, int work) const = 0;
    virtual std::vector<NodeType> initial_nodes() const = 0;

    std::vector<TransType> reverse_transition(const NodeType& next_state) const override {
        std::vector<TransType> reversed;

        for (int config = 0; config <= 1; ++config) {
            for (int dist = 1; dist <= lamb; ++dist) {
                for (int pred1 = 0; pred1 <= 1; ++pred1) {
                    for (int pred2 = 0; pred2 <= config; ++pred2) {
                        for (int work = 0; work <= dist; ++work) {
                            if (work > dist) continue;
                            if (config == 0 && pred2 == 1) continue;

                            NodeType prev_node(config, dist, pred1, pred2, work);
                            if (!prev_node.is_valid()) continue;

                            auto outs = transition(prev_node);

                            for (const auto& t : outs) {
                                if (t.get_state() == next_state) {
                                    TransType rev(t.get_flip(), t.get_opt(), t.get_online(), prev_node);
                                    reversed.push_back(rev);
                                }
                            }
                        }
                    }
                }
            }
        }

        return reversed;
    }

    std::vector<TransType> transition(const NodeType& curr) const override {
        std::vector<TransType> transitions;
        auto next_reqs = next_requests(curr);
        for (const auto& [time, location, pred] : next_reqs) {
            auto [costs, next_state] = choose_config(curr, time, location, pred);
            transitions.emplace_back(location, costs.first, costs.second, std::move(next_state));
        }
        return transitions;
    }

protected:
    bool bound_mode_;

private:
    std::vector<std::tuple<int,int,int>> next_requests(const NodeType& curr) const {
        std::vector<std::tuple<int,int,int>> requests;

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
                int start_time = std::max(lamb - curr.get_dist(), 0);
                for (int time = start_time; time <= lamb + 1; ++time) {
                    requests.emplace_back(time, 1, 0);
                    requests.emplace_back(time, 1, 1);
                }
            } else {
                requests.emplace_back(lamb + 1, 0, 0);
                requests.emplace_back(lamb + 1, 0, 1);
                requests.emplace_back(lamb, 0, 0);
                requests.emplace_back(lamb, 0, 1);

                int start_time = std::max(lamb - curr.get_dist(), 0);
                for (int time = start_time; time <= lamb + 1; ++time) {
                    requests.emplace_back(time, 1, 0);
                    requests.emplace_back(time, 1, 1);
                }
            }
        }
        return requests;
    }

    std::pair<std::pair<int,int>, NodeType>
    choose_config(const NodeType& curr, int time, int location, int pred) const {
        assert(time >= 0);
        assert(curr.get_work() >= 0 && curr.get_work() <= lamb);
        assert(curr.get_dist() >= 0 && curr.get_dist() <= lamb);

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

        int next_dist, next_w, one_w, both_w;
        if (bound_mode_) {
            if (location == 0) {
                next_dist = std::min(lamb, dist + time);
                one_w = time;
                both_w = std::min(time + lamb - 1, work + 2 * time);
                next_w = both_w - one_w;
            } else {
                next_dist = std::min(time, lamb);
                one_w = std::min(time + lamb - 1, work + time);
                both_w = std::min(work + 2 * time, time + lamb - 1);
                next_w = both_w - one_w;
            }
        } else {
            if (location == 0) {
                next_dist = std::min(lamb, dist + time);
                one_w = time;
                both_w = std::min(time + lamb, work + 2 * time);
                next_w = both_w - one_w;
            } else {
                next_dist = std::min(time, lamb);
                one_w = std::min(time + lamb, work + time);
                both_w = std::min(work + 2 * time, time + lamb);
                next_w = both_w - one_w;
            }
        }

        int transfer_cost = bound_mode_ ? time + lamb + 1 : time + lamb;
        int stp_transfer  = bound_mode_ ? lamb + 1 : lamb;

        int online = 0;
        int next_C = 0;

        if (config == 0) {
            assert(pred2 == 0);
            if (location == 0) {
                online = time;
                next_C = 0;
            } else {
                online = transfer_cost;
                next_C = 1;
            }
        } else {
            if (!pred1 && !pred2) {
                assert(dist - work >= 0);
                bool stay = get_stay(dist, work);
                if ((location ^ stay) != 0) {
                    if (time != 0) {
                        online = transfer_cost;
                    }
                    next_C = 1;
                } else {
                    online = time;
                    next_C = 0;
                }
            } else if (!pred1 && pred2) {
                if (bound_mode_) {
                    if (location != 1) {
                        curr.print(std::cerr);
                    }
                }
                assert(location == 1);
                online = time;
                next_C = 0;
            } else if (pred1 && !pred2) {
                double y = get_y(dist, work);
                if (std::round(y * lamb) >= time) {
                    online = 2 * time;
                    next_C = 1;
                } else {
                    online = time + static_cast<int>(std::round(y * lamb));
                    next_C = 0;
                    if (location == 1) {
                        online += stp_transfer;
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
