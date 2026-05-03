#pragma once
#include <cstdint>
#include <limits>
#include <vector>

enum class NodeType { BASE, STP_MIN, LTP_TOP_MIN, LTP_BOT };

// Returned by GameGraph::get_edge(u, ei)
struct EdgeInfo {
    int to;       // destination node ID
    long long reward;   // p * opt_num - q * alg_num
    long long opt_num;  // coefficient of p in reward decomposition
    long long alg_num;  // coefficient of q in reward decomposition
};

// Metadata stored per node
struct NodeMeta {
    NodeType type;
    int d, w;       // base parameters (y stored separately for STP_MIN)
    int y;          // only for STP_MIN
    int mode;       // only for BASE (0=STP, 1=LTP)
    int num_edges;  // out-degree
};

static constexpr long long INF64 = std::numeric_limits<long long>::max() / 2;
