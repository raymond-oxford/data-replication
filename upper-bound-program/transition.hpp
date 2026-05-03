#pragma once
#include <cstdint>
#include <cassert>
#include <iostream>

template<typename N>
struct Transition {
    uint64_t code;

    static constexpr int NODE_BITS   = N::size;
    static constexpr int FLIP_BITS   = 2;
    static constexpr int OPT_BITS    = 15; // signed
    static constexpr int ONLINE_BITS = 15;

    static_assert(NODE_BITS + FLIP_BITS + OPT_BITS + ONLINE_BITS <= 64,
                  "Transition encoding won't fit in uint64_t");

    Transition() : code(0) {}

    Transition(int flip, int opt, int online, N state) {
        assert(flip >= 0 && flip < (1 << FLIP_BITS));
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
        return (code >> NODE_BITS) & ((1 << FLIP_BITS) - 1);
    }

    int get_online() const {
        return (code >> (NODE_BITS + FLIP_BITS + OPT_BITS)) & ((1 << ONLINE_BITS) - 1);
    }

    int get_opt() const {
        int32_t raw = (code >> (NODE_BITS + FLIP_BITS)) & ((1 << OPT_BITS) - 1);
        if (raw & (1 << (OPT_BITS - 1))) {
            raw |= ~((1 << OPT_BITS) - 1);
        }
        return raw;
    }

    N get_state() const {
        return N(code & ((1ULL << NODE_BITS) - 1));
    }

    void print(std::ostream& out) const {
        out << "Transition - flip: " << get_flip()
                  << ", opt: " << get_opt()
                  << ", online: " << get_online()
                  << ", state: ";
        get_state().print(out);
    }

    bool operator==(const Transition<N>& other) const {
        return get_flip() == other.get_flip() &&
               get_opt() == other.get_opt() &&
               get_online() == other.get_online() &&
               get_state().code == other.get_state().code;
    }
};

namespace std {
    template<typename N>
    struct hash<Transition<N>> {
        size_t operator()(const Transition<N>& t) const {
            return std::hash<int>()(t.get_flip())
                 ^ (std::hash<int>()(t.get_opt()) << 1)
                 ^ (std::hash<int>()(t.get_online()) << 2)
                 ^ (std::hash<int>()(t.get_state().code) << 3);
        }
    };
}
