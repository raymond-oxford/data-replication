#pragma once
#include "alg.hpp"

template <int Lambda>
class AnalyticalAlgorithm : public BinaryAlgorithm<Lambda> {
public:
  AnalyticalAlgorithm(bool bound_mode = false)
      : BinaryAlgorithm<Lambda>(bound_mode) {}

  std::vector<Node<Lambda>> initial_nodes() const override {
    return {Node<Lambda>(0, Lambda, 0, 0, Lambda)};
  }

  // LTP: always go to top (no replication)
  bool get_stay(int dist, int work) const override {
      int thresh = std::round(0.63 * Lambda);
      int k = dist - work;
      return k >= Lambda - thresh;
  }

  // STP: y=0 means switch immediately (no waiting in double)
  double get_y(int dist, int work) const override {
      int thresh = std::round(0.63 * Lambda);
      int k = dist - work;
      if (k >= Lambda - thresh) {
          return thresh - work;
      }
      else {
          return 0;
      }
  }
};
