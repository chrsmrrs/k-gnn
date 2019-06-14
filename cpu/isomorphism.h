#pragma once

#include <torch/extension.h>
#include <vector>

#include "adjacency.h"
#include "iterate.h"

using namespace at;
using namespace std;

inline int64_t pair(int64_t u, int64_t v) {
  return u >= v ? u * u + u + v : u + v * v;
}

inline Tensor convert(Tensor x) {
  auto range = torch::empty(x.size(1), x.options());
  arange_out(range, x.size(1));
  x = x * range.view({1, -1});
  return x.sum(1).toType(kLong);
}

template <int64_t K, bool connected> struct Isomorphism;

template <> struct Isomorphism<2, true> {
  static int64_t type(vector<int64_t> set, Tensor row, Tensor col, Tensor x,
                      int64_t num_labels) {
    auto x_data = x.data<int64_t>();
    vector<int64_t> labels = {x_data[set[0]], x_data[set[1]]};
    sort(labels.begin(), labels.end());

    return labels[0] * num_labels + labels[1];
  }
};

template <> struct Isomorphism<2, false> {
  static int64_t type(vector<int64_t> set, Tensor row, Tensor col, Tensor x,
                      int64_t num_labels) {
    auto x_data = x.data<int64_t>();
    vector<int64_t> labels = {x_data[set[0]], x_data[set[1]]};
    sort(labels.begin(), labels.end());

    return num_labels * num_labels * is_adjacent(set[0], set[1], row, col) +
           labels[0] * num_labels + labels[1];
  }
};

template <> struct Isomorphism<3, true> {
  static int64_t type(vector<int64_t> set, Tensor row, Tensor col, Tensor x,
                      int64_t num_labels) {
    auto x_data = x.data<int64_t>();
    vector<int64_t> labels = {x_data[set[0]], x_data[set[1]], x_data[set[2]]};
    sort(labels.begin(), labels.end());

    return num_labels * num_labels * num_labels *
               is_adjacent(set[2], set[0], row, col) +
           labels[0] * num_labels * num_labels + labels[1] * num_labels +
           labels[2];
  }
};

template <> struct Isomorphism<3, false> {
  static int64_t type(vector<int64_t> set, Tensor row, Tensor col, Tensor x,
                      int64_t num_labels) {
    // TODO
    printf("Not yet implemented.\n");
    return -1;
  }
};
