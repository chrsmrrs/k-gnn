#pragma once

#include <torch/extension.h>

#include "isomorphism.h"
#include "iterate.h"
#include "utils.h"

using namespace at;
using namespace std;

typedef tuple<map<vector<int64_t>, int64_t>, Tensor> AssignmentType;

template <int64_t K> struct Assignment;

template <> struct Assignment<2> {
  static AssignmentType unconnected(Tensor row, Tensor col, Tensor x,
                                    int64_t num_nodes) {
    map<vector<int64_t>, int64_t> set_to_id;
    vector<int64_t> iso_type;
    auto num_labels = x.size(1);
    x = convert(x);

    int64_t i = 0;
    ITERATE_NODES(0, u, num_nodes, {
      ITERATE_NODES(u + 1, v, num_nodes, {
        set_to_id.insert({{u, v}, i});
        iso_type.push_back(
            Isomorphism<2, false>::type({u, v}, row, col, x, num_labels));
        i++;
      });
    });

    return make_tuple(set_to_id, from_vector(iso_type));
  }

  static AssignmentType connected(Tensor row, Tensor col, Tensor x,
                                  int64_t num_nodes) {
    auto row_data = row.data_ptr<int64_t>(), col_data = col.data_ptr<int64_t>();
    map<vector<int64_t>, int64_t> set_to_id;
    vector<int64_t> iso_type;
    auto num_labels = x.size(1);
    x = convert(x);

    int64_t i = 0;
    ITERATE_NODES(0, u, num_nodes, {
      ITERATE_NEIGHBORS(u, v, row_data, col_data, {
        if (u >= v)
          continue;
        set_to_id.insert({{u, v}, i});
        iso_type.push_back(
            Isomorphism<2, true>::type({u, v}, row, col, x, num_labels));
        i++;
      });
    });

    return make_tuple(set_to_id, from_vector(iso_type));
  }
};

template <> struct Assignment<3> {
  static AssignmentType unconnected(Tensor row, Tensor col, Tensor x,
                                    int64_t num_nodes) {
    map<vector<int64_t>, int64_t> set_to_id;
    vector<int64_t> iso_type;
    auto num_labels = x.size(1);
    x = convert(x);

    int64_t i = 0;
    ITERATE_NODES(0, u, num_nodes, {
      ITERATE_NODES(u + 1, v, num_nodes, {
        ITERATE_NODES(v + 1, w, num_nodes, {
          set_to_id.insert({{u, v, w}, i});
          iso_type.push_back(
              Isomorphism<3, false>::type({u, v, w}, row, col, x, num_labels));
          i++;
        });
      });
    });

    return make_tuple(set_to_id, from_vector(iso_type));
  }

  static AssignmentType connected(Tensor row, Tensor col, Tensor x,
                                  int64_t num_nodes) {
    auto row_data = row.data_ptr<int64_t>(), col_data = col.data_ptr<int64_t>();
    map<vector<int64_t>, int64_t> set_to_id;
    vector<int64_t> iso_type;
    auto num_labels = x.size(1);
    x = convert(x);

    int64_t i = 0;
    ITERATE_NODES(0, u, num_nodes, {
      ITERATE_NEIGHBORS(u, v, row_data, col_data, {
        ITERATE_NEIGHBORS(v, w, row_data, col_data, {
          if (w == u)
            continue;
          vector<int64_t> set = {u, v, w};
          sort(set.begin(), set.end());
          auto iter = set_to_id.find(set);
          if (iter == set_to_id.end()) {
            set_to_id.insert({set, i});
            iso_type.push_back(
                Isomorphism<3, true>::type(set, row, col, x, num_labels));
            i++;
          }
        });
      });
    });

    return make_tuple(set_to_id, from_vector(iso_type));
  }
};
