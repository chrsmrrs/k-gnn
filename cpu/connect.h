#pragma once

#include <torch/extension.h>

#include "iterate.h"
#include "utils.h"

using namespace at;
using namespace std;

#define ADD_SET(ID, SET)                                                       \
  [&] {                                                                        \
    sort(SET.begin(), SET.end());                                              \
    auto item2 = set_to_id.find(SET);                                          \
    if (item2 != set_to_id.end()) {                                            \
      rows.push_back(ID);                                                      \
      cols.push_back(item2->second);                                           \
      rows.push_back(item2->second);                                           \
      cols.push_back(ID);                                                      \
    }                                                                          \
  }()

template <int64_t K> struct Connect;

template <> struct Connect<2> {
  static Tensor local(Tensor row, Tensor col,
                      map<vector<int64_t>, int64_t> set_to_id) {
    auto row_data = row.data_ptr<int64_t>(), col_data = col.data_ptr<int64_t>();
    vector<int64_t> rows, cols;

    for (auto item : set_to_id) {
      ITERATE_NEIGHBORS(item.first[0], x, row_data, col_data, {
        vector<int64_t> set1 = {item.first[0], x};
        ADD_SET(item.second, set1);
        vector<int64_t> set2 = {item.first[1], x};
        ADD_SET(item.second, set2);
      });

      ITERATE_NEIGHBORS(item.first[1], x, row_data, col_data, {
        vector<int64_t> set1 = {item.first[0], x};
        ADD_SET(item.second, set1);
        vector<int64_t> set2 = {item.first[1], x};
        ADD_SET(item.second, set2);
      });
    }

    if (rows.size() == 0) {
      return torch::empty(0, row.options());
    }

    auto index = torch::stack({from_vector(rows), from_vector(cols)}, 0);
    return coalesce(remove_self_loops(index), (int64_t)set_to_id.size());
  }

  static Tensor malkin(Tensor row, Tensor col,
                       map<vector<int64_t>, int64_t> set_to_id) {
    auto row_data = row.data_ptr<int64_t>(), col_data = col.data_ptr<int64_t>();
    vector<int64_t> rows, cols;

    for (auto item : set_to_id) {
      ITERATE_NEIGHBORS(item.first[0], x, row_data, col_data, {
        vector<int64_t> set = {item.first[1], x};
        ADD_SET(item.second, set);
      });

      ITERATE_NEIGHBORS(item.first[1], x, row_data, col_data, {
        vector<int64_t> set = {item.first[0], x};
        ADD_SET(item.second, set);
      });
    }

    if (rows.size() == 0) {
      return torch::empty(0, row.options());
    }

    auto index = torch::stack({from_vector(rows), from_vector(cols)}, 0);
    return coalesce(remove_self_loops(index), (int64_t)set_to_id.size());
  }
};

template <> struct Connect<3> {
  static Tensor local(Tensor row, Tensor col,
                      map<vector<int64_t>, int64_t> set_to_id) {
    auto row_data = row.data_ptr<int64_t>(), col_data = col.data_ptr<int64_t>();
    vector<int64_t> rows, cols;

    for (auto item : set_to_id) {
      ITERATE_NEIGHBORS(item.first[0], x, row_data, col_data, {
        vector<int64_t> set1 = {item.first[0], item.first[1], x};
        ADD_SET(item.second, set1);
        vector<int64_t> set2 = {item.first[0], item.first[2], x};
        ADD_SET(item.second, set2);
        vector<int64_t> set3 = {item.first[1], item.first[2], x};
        ADD_SET(item.second, set3);
      });

      ITERATE_NEIGHBORS(item.first[1], x, row_data, col_data, {
        vector<int64_t> set1 = {item.first[0], item.first[1], x};
        ADD_SET(item.second, set1);
        vector<int64_t> set2 = {item.first[0], item.first[2], x};
        ADD_SET(item.second, set2);
        vector<int64_t> set3 = {item.first[1], item.first[2], x};
        ADD_SET(item.second, set3);
      });

      ITERATE_NEIGHBORS(item.first[2], x, row_data, col_data, {
        vector<int64_t> set1 = {item.first[0], item.first[1], x};
        ADD_SET(item.second, set1);
        vector<int64_t> set2 = {item.first[0], item.first[2], x};
        ADD_SET(item.second, set2);
        vector<int64_t> set3 = {item.first[1], item.first[2], x};
        ADD_SET(item.second, set3);
      });
    }

    if (rows.size() == 0) {
      return torch::empty(0, row.options());
    }

    auto index = torch::stack({from_vector(rows), from_vector(cols)}, 0);
    return coalesce(remove_self_loops(index), (int64_t)set_to_id.size());
  }

  static Tensor malkin(Tensor row, Tensor col,
                       map<vector<int64_t>, int64_t> set_to_id) {
    auto row_data = row.data_ptr<int64_t>(), col_data = col.data_ptr<int64_t>();
    vector<int64_t> rows, cols;

    for (auto item : set_to_id) {
      ITERATE_NEIGHBORS(item.first[0], x, row_data, col_data, {
        vector<int64_t> set = {item.first[1], item.first[2], x};
        ADD_SET(item.second, set);
      });

      ITERATE_NEIGHBORS(item.first[1], x, row_data, col_data, {
        vector<int64_t> set = {item.first[0], item.first[2], x};
        ADD_SET(item.second, set);
      });

      ITERATE_NEIGHBORS(item.first[2], x, row_data, col_data, {
        vector<int64_t> set = {item.first[0], item.first[1], x};
        ADD_SET(item.second, set);
      });
    }

    if (rows.size() == 0) {
      return torch::empty(0, row.options());
    }

    auto index = torch::stack({from_vector(rows), from_vector(cols)}, 0);
    return coalesce(remove_self_loops(index), (int64_t)set_to_id.size());
  }
};
