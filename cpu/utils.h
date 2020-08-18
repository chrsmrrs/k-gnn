#pragma once

#include <torch/extension.h>
#include <vector>

using namespace at;
using namespace std;

inline Tensor remove_self_loops(Tensor index) {
  auto row = index[0], col = index[1];
  auto mask = row != col;
  row = row.masked_select(mask), col = col.masked_select(mask);
  return stack({row, col}, 0);
}

inline Tensor coalesce(Tensor index, int64_t num_nodes) {
  Tensor row = index[0], col = index[1], unique, inv, perm;
  tie(unique, inv) = _unique(num_nodes * row + col, true, true);

  perm = torch::empty(inv.size(0), index.options());
  arange_out(perm, inv.size(0));
  perm = torch::empty(unique.size(0), index.options()).scatter_(0, inv, perm);

  row = row.index_select(0, perm);
  col = col.index_select(0, perm);

  return stack({row, col}, 0);
}

inline Tensor sort_by_row(Tensor index) {
  Tensor row = index[0], col = index[1], perm;
  tie(row, perm) = row.sort();
  col = col.index_select(0, perm);
  return stack({row, col}, 0);
}

inline Tensor degree(Tensor row, int64_t num_nodes) {
  auto zero = torch::zeros(num_nodes, row.options());
  auto one = torch::ones(row.size(0), row.options());
  return zero.scatter_add_(0, row, one);
}

inline tuple<Tensor, Tensor> to_csr(Tensor index, int64_t num_nodes) {
  index = sort_by_row(index);
  auto row = degree(index[0], num_nodes).cumsum(0);
  row = cat({torch::zeros(1, row.options()), row}, 0); // Prepend zero.
  return make_tuple(row, index[1]);
}

inline Tensor from_vector(vector<int64_t> src) {
  auto out = torch::empty((size_t)src.size(), torch::CPU(at::kLong));
  auto out_data = out.data<int64_t>();
  for (ptrdiff_t i = 0; i < out.size(0); i++) {
    out_data[i] = src[i];
  }
  return out;
}

template <int64_t K> struct MapToTensor;

template <> struct MapToTensor<2> {
  static Tensor get(map<vector<int64_t>, int64_t> set_to_id) {
    int64_t size = (int64_t)set_to_id.size();
    Tensor set = torch::empty(2 * size, torch::CPU(at::kLong));
    Tensor id = torch::empty(2 * size, torch::CPU(at::kLong));
    auto set_data = set.data<int64_t>(), id_data = id.data<int64_t>();

    int64_t i = 0;
    for (auto item : set_to_id) {
      set_data[2 * i] = item.first[0];
      set_data[2 * i + 1] = item.first[1];
      id_data[2 * i] = item.second;
      id_data[2 * i + 1] = item.second;
      i++;
    }

    return stack({set, id}, 0);
  }
};

template <> struct MapToTensor<3> {
  static Tensor get(map<vector<int64_t>, int64_t> set_to_id) {
    int64_t size = (int64_t)set_to_id.size();
    Tensor set = torch::empty(3 * size, torch::CPU(at::kLong));
    Tensor id = torch::empty(3 * size, torch::CPU(at::kLong));
    auto set_data = set.data<int64_t>(), id_data = id.data<int64_t>();

    int64_t i = 0;
    for (auto item : set_to_id) {
      set_data[3 * i] = item.first[0];
      set_data[3 * i + 1] = item.first[1];
      set_data[3 * i + 2] = item.first[2];
      id_data[3 * i] = item.second;
      id_data[3 * i + 1] = item.second;
      id_data[3 * i + 2] = item.second;
      i++;
    }

    return stack({set, id}, 0);
  }
};
