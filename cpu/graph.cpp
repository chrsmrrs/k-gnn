#include <torch/extension.h>
#include <vector>

#include "assignment.h"
#include "connect.h"
#include "utils.h"

using namespace at;
using namespace std;

template <int64_t K>
vector<Tensor> local(Tensor index, Tensor x, int64_t num_nodes) {
  Tensor row, col;
  tie(row, col) = to_csr(index, num_nodes);
  Tensor assignment, iso_type;
  map<vector<int64_t>, int64_t> set_to_id;
  tie(set_to_id, iso_type) = Assignment<K>::unconnected(row, col, x, num_nodes);
  index = Connect<K>::local(row, col, set_to_id);
  assignment = MapToTensor<K>::get(set_to_id);
  return {index, assignment, iso_type};
}

template <int64_t K>
vector<Tensor> connected_local(Tensor index, Tensor x, int64_t num_nodes) {
  Tensor row, col;
  tie(row, col) = to_csr(index, num_nodes);
  Tensor assignment, iso_type;
  map<vector<int64_t>, int64_t> set_to_id;
  tie(set_to_id, iso_type) = Assignment<K>::connected(row, col, x, num_nodes);
  index = Connect<K>::local(row, col, set_to_id);
  assignment = MapToTensor<K>::get(set_to_id);
  return {index, assignment, iso_type};
}

template <int64_t K>
vector<Tensor> malkin(Tensor index, Tensor x, int64_t num_nodes) {
  Tensor row, col;
  tie(row, col) = to_csr(index, num_nodes);
  Tensor assignment, iso_type;
  map<vector<int64_t>, int64_t> set_to_id;
  tie(set_to_id, iso_type) = Assignment<K>::unconnected(row, col, x, num_nodes);
  index = Connect<K>::malkin(row, col, set_to_id);
  assignment = MapToTensor<K>::get(set_to_id);
  return {index, assignment, iso_type};
}

template <int64_t K>
vector<Tensor> connected_malkin(Tensor index, Tensor x, int64_t num_nodes) {
  Tensor row, col;
  tie(row, col) = to_csr(index, num_nodes);
  Tensor assignment, iso_type;
  map<vector<int64_t>, int64_t> set_to_id;
  tie(set_to_id, iso_type) = Assignment<K>::connected(row, col, x, num_nodes);
  index = Connect<K>::malkin(row, col, set_to_id);
  assignment = MapToTensor<K>::get(set_to_id);
  return {index, assignment, iso_type};
}

Tensor assignment_2to3(Tensor index, int64_t num_nodes) {
  Tensor row, col;
  tie(row, col) = to_csr(index, num_nodes);
  auto one = ones({num_nodes, 1}, index.options());
  map<vector<int64_t>, int64_t> set2_to_id =
      get<0>(Assignment<2>::unconnected(row, col, one, num_nodes));
  map<vector<int64_t>, int64_t> set3_to_id =
      get<0>(Assignment<3>::connected(row, col, one, num_nodes));

  vector<int64_t> rows, cols;
  for (auto item3 : set3_to_id) {
    int64_t u = item3.first[0], v = item3.first[1], w = item3.first[2];

    auto item2 = set2_to_id.find({u, v});
    rows.push_back(item2->second);
    cols.push_back(item3.second);

    item2 = set2_to_id.find({u, w});
    rows.push_back(item2->second);
    cols.push_back(item3.second);

    item2 = set2_to_id.find({v, w});
    rows.push_back(item2->second);
    cols.push_back(item3.second);
  }

  return stack({from_vector(rows), from_vector(cols)}, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("two_local", &local<2>, "2-Local");
  m.def("connected_two_local", &connected_local<2>, "Connected 2-Local");
  m.def("two_malkin", &malkin<2>, "2-Malkin");
  m.def("connected_two_malkin", &connected_malkin<2>, "Connected 2-Malkin");
  m.def("three_local", &local<3>, "3-Local");
  m.def("connected_three_local", &connected_local<3>, "Connected 3-Local");
  m.def("three_malkin", &malkin<3>, "3-Malkin");
  m.def("connected_three_malkin", &connected_malkin<3>, "Connected 3-Malkin");
  m.def("assignment_2to3", &assignment_2to3, "Assignment Two To Three Graph");
}
