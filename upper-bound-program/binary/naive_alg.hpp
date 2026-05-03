#pragma once
#include "alg.hpp"

template<int Lambda>
class NaiveAlgorithm : public BinaryAlgorithm<Lambda> {
public:
    NaiveAlgorithm(bool bound_mode = false) : BinaryAlgorithm<Lambda>(bound_mode) {}

    std::vector<Node<Lambda>> initial_nodes() const override {
        return { Node<Lambda>(0, Lambda, 0, 0, Lambda) };
    }

    // LTP: always go to top (no replication)
    bool get_stay(int /*dist*/, int /*work*/) const override { return false; }

    // STP: y=0 means switch immediately (no waiting in double)
    double get_y(int /*dist*/, int /*work*/) const override { return 0.0; }
};
