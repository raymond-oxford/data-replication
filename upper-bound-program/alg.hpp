#pragma once
#include "transition.hpp"
#include <vector>

template<typename N>
class Algorithm {
public:
    using NodeType = N;
    using TransType = Transition<N>;

    virtual ~Algorithm() = default;

    virtual std::vector<TransType> transition(const N& curr) const = 0;
    virtual std::vector<N> initial_nodes() const { return {}; }
    virtual std::vector<TransType> reverse_transition(const N& curr) const { return {}; }

    // Optional: pre-build reverse graph from a known reachable node set.
    // Called before compute_distances_reverse when --reverse is used.
    // Default is a no-op; override when brute-forcing reverse_transition is infeasible.
    virtual void build_reverse_graph(const std::vector<N>& /*nodes*/) const {}
};
