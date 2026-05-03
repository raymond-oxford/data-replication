#include "game_graph.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>

GameGraph::GameGraph(int N) : N_(N), p_(0), q_(1) {
    int Np1 = N + 1;
    base_arr_.assign(Np1 * Np1 * 2, -1);
    stpmin_arr_.assign(Np1 * Np1 * Np1, -1);
    ltptop_arr_.assign(Np1 * Np1 * N, -1);
    ltp_bot_id_ = -1;

    int id = 0;

    // 1. BASE nodes: d=0..N, w=0..d, mode=0,1
    for (int d = 0; d <= N; ++d) {
        for (int w = 0; w <= d; ++w) {
            for (int mode = 0; mode <= 1; ++mode) {
                base_arr_[base_idx(d, w, mode)] = id;
                NodeMeta m{};
                m.type = NodeType::BASE;
                m.d = d; m.w = w; m.mode = mode; m.y = 0;
                int num_y = (d < N) ? (d + 1) : N;
                m.num_edges = (mode == 0) ? (N - w + 1) : (num_y + 1);
                meta_.push_back(m);
                ++id;
            }
        }
    }

    // 2. STP_MIN nodes: d=0..N, w=0..d, y=0..N-w
    for (int d = 0; d <= N; ++d) {
        for (int w = 0; w <= d; ++w) {
            for (int y = 0; y <= N - w; ++y) {
                stpmin_arr_[stpmin_idx(d, w, y)] = id;
                NodeMeta m{};
                m.type = NodeType::STP_MIN;
                m.d = d; m.w = w; m.y = y; m.mode = 0;
                m.num_edges = y + 2;
                meta_.push_back(m);
                ++id;
            }
        }
    }

    // 3. LTP_TOP_MIN nodes: d=0..N, w=0..d, y in {0} ∪ [N-d, N-1]
    for (int d = 0; d <= N; ++d) {
        for (int w = 0; w <= d; ++w) {
            auto add_ltptop = [&](int y) {
                ltptop_arr_[ltptop_idx(d, w, y)] = id;
                NodeMeta m{};
                m.type = NodeType::LTP_TOP_MIN;
                m.d = d; m.w = w; m.y = y; m.mode = 0;
                int x_min = std::max(y + 1, N - d);
                m.num_edges = 2 * (N - x_min + 1);
                meta_.push_back(m);
                ++id;
            };
            add_ltptop(0);
            int y_range_start = (d < N) ? (N - d) : 1;
            for (int y = y_range_start; y <= N - 1; ++y)
                add_ltptop(y);
        }
    }

    // 4. LTP_BOT: single node
    ltp_bot_id_ = id;
    NodeMeta m{};
    m.type = NodeType::LTP_BOT;
    m.d = 0; m.w = 0; m.y = 0; m.mode = 0;
    m.num_edges = 2;
    meta_.push_back(m);
    ++id;
}

void GameGraph::build(long long p, long long q) {
    p_ = p;
    q_ = q;
}

int GameGraph::base_node(int d, int w, int mode) const   { return base_lookup(d, w, mode); }
int GameGraph::stpmin_node(int d, int w, int y) const    { return stpmin_lookup(d, w, y); }
int GameGraph::ltptop_node(int d, int w, int y) const    { return ltptop_lookup(d, w, y); }

int GameGraph::base_lookup(int d, int w, int mode) const {
    if (d < 0 || d > N_ || w < 0 || w > d || mode < 0 || mode > 1) return -1;
    return base_arr_[base_idx(d, w, mode)];
}

int GameGraph::stpmin_lookup(int d, int w, int y) const {
    if (d < 0 || d > N_ || w < 0 || w > d || y < 0 || y > N_ - w) return -1;
    return stpmin_arr_[stpmin_idx(d, w, y)];
}

int GameGraph::ltptop_lookup(int d, int w, int y) const {
    if (d < 0 || d > N_ || w < 0 || w > d || y < 0 || y >= N_) return -1;
    return ltptop_arr_[ltptop_idx(d, w, y)];
}

EdgeInfo GameGraph::edge_base(int u, int ei) const {
    const NodeMeta& m = meta_[u];
    EdgeInfo e{};
    e.opt_num = 0;
    e.alg_num = 0;
    e.reward = 0;
    if (m.mode == 0) {
        // STP mode: ei = y, destination = stpmin[d, w, y]
        int y = ei;
        e.to = stpmin_lookup(m.d, m.w, y);
    } else {
        // LTP mode: ei=0..num_y-1 -> ltptop[d,w,y], ei=num_y -> LTP_BOT
        int num_y = (m.d < N_) ? (m.d + 1) : N_;
        if (ei < num_y) {
            int y;
            if (m.d == N_)    y = ei;
            else if (ei == 0) y = 0;
            else              y = N_ - m.d + (ei - 1);
            e.to = ltptop_lookup(m.d, m.w, y);
        } else {
            e.to = ltp_bot_id_;
        }
    }
    assert(e.to != -1);
    return e;
}

EdgeInfo GameGraph::edge_stpmin(int u, int ei) const {
    const NodeMeta& m = meta_[u];
    int d = m.d, w = m.w, y = m.y;
    EdgeInfo e{};

    if (ei >= 2) {
        // x-loop edge: ei = x+1, x = ei-1 >= 1
        int x = ei - 1;
        int dd = std::min(d + x, N_);
        int ww = w + x;
        e.to = base_lookup(dd, ww, 1);
        assert(e.to != -1);
        e.opt_num = x;
        e.alg_num = 2 * x;
        e.reward = p_ * e.opt_num - q_ * e.alg_num;
        return e;
    }

    // ei = 0 or 1
    bool case_a = (y <= N_ - d - 1);
    if (case_a) {
        int d2 = N_ - d - y - 1;
        e.opt_num = N_ + w - d + y + 1;
        e.alg_num = 2 * N_ - d + y;
        e.reward = p_ * e.opt_num - q_ * e.alg_num;
        e.to = base_lookup(d2, d2, ei); // ei=0 -> mode 0, ei=1 -> mode 1
    } else {
        // case B: y >= N-d
        int minval = std::min(w + y + 1, N_);
        e.opt_num = y + 1 + minval;
        e.alg_num = N_ + 2 * y + 1;
        e.reward = p_ * e.opt_num - q_ * e.alg_num;
        e.to = base_lookup(0, 0, ei); // ei=0 -> mode 0, ei=1 -> mode 1
    }
    assert(e.to != -1);
    return e;
}

EdgeInfo GameGraph::edge_ltptop(int u, int ei) const {
    const NodeMeta& m = meta_[u];
    int d = m.d, w = m.w, y = m.y;
    int x_min = std::max(y + 1, N_ - d);
    int x    = x_min + ei / 2;
    int mode = ei % 2;
    int w2   = std::min(N_ - w, x);
    EdgeInfo e{};
    e.to      = base_lookup(x, w2, mode);
    assert(e.to != -1);
    e.opt_num = w + x;
    e.alg_num = x + y + N_;
    e.reward  = p_ * e.opt_num - q_ * e.alg_num;
    return e;
}

EdgeInfo GameGraph::edge_ltpbot(int /*u*/, int ei) const {
    EdgeInfo e{};
    e.to = base_lookup(N_, N_, ei); // ei=0 -> mode 0, ei=1 -> mode 1
    assert(e.to != -1);
    e.opt_num = N_;
    e.alg_num = 2 * N_;
    e.reward = p_ * e.opt_num - q_ * e.alg_num;
    return e;
}

EdgeInfo GameGraph::get_edge(int u, int ei) const {
    switch (meta_[u].type) {
        case NodeType::BASE:        return edge_base(u, ei);
        case NodeType::STP_MIN:     return edge_stpmin(u, ei);
        case NodeType::LTP_TOP_MIN: return edge_ltptop(u, ei);
        case NodeType::LTP_BOT:     return edge_ltpbot(u, ei);
    }
    throw std::logic_error("unknown node type");
}
