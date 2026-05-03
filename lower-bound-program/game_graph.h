#pragma once
#include "types.h"
#include <vector>

class GameGraph {
public:
    explicit GameGraph(int N);

    // (Re)compute all edge rewards for the given (p, q).
    // Node IDs, topology, and opt_num/alg_num are fixed after construction.
    void build(long long p, long long q);

    // Number of nodes in the graph.
    int num_nodes() const { return (int)meta_.size(); }

    // True if node u is Max-owned (BASE nodes).
    bool is_max(int u) const { return meta_[u].type == NodeType::BASE; }

    // Out-degree of node u.
    int num_edges(int u) const { return meta_[u].num_edges; }

    // Return edge info for edge index ei out of node u.
    // O(1). Valid for 0 <= ei < num_edges(u).
    EdgeInfo get_edge(int u, int ei) const;

    // Node metadata accessor.
    const NodeMeta& meta(int u) const { return meta_[u]; }

    int N() const { return N_; }

    // Public node-ID lookups (return -1 if out of range).
    int base_node(int d, int w, int mode) const;
    int stpmin_node(int d, int w, int y) const;
    int ltptop_node(int d, int w, int y) const;

private:
    int N_;

    // Flat lookup arrays: value is node ID, -1 if invalid.
    // Indexed by the integer keys described in the spec.
    std::vector<int> base_arr_;   // size: ((N+1)*(N+1)*2)
    std::vector<int> stpmin_arr_; // size: ((N+1)*(N+1)*(N+1))
    std::vector<int> ltptop_arr_; // size: ((N+1)*(N+1)*N)

    int ltp_bot_id_;

    std::vector<NodeMeta> meta_;

    // Current p, q (set by build()).
    long long p_, q_;

    // Helpers to index lookup arrays.
    int base_idx(int d, int w, int mode) const {
        return ((d * (N_ + 1) + w) * 2) + mode;
    }
    int stpmin_idx(int d, int w, int y) const {
        return ((d * (N_ + 1) + w) * (N_ + 1)) + y;
    }
    int ltptop_idx(int d, int w, int y) const {
        return (d * (N_ + 1) + w) * N_ + y;
    }

    // Safe lookup helpers (return -1 if out of range).
    int base_lookup(int d, int w, int mode) const;
    int stpmin_lookup(int d, int w, int y) const;
    int ltptop_lookup(int d, int w, int y) const;

    // Edge computation helpers (return EdgeInfo with reward already set).
    EdgeInfo edge_base(int u, int ei) const;
    EdgeInfo edge_stpmin(int u, int ei) const;
    EdgeInfo edge_ltptop(int u, int ei) const;
    EdgeInfo edge_ltpbot(int u, int ei) const;
};
