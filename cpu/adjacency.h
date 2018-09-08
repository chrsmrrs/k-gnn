#pragma once

#include <torch/torch.h>

#include "iterate.h"

using namespace at;

inline int64_t is_adjacent(int64_t u, int64_t v, Tensor row, Tensor col) {
  ITERATE_NEIGHBORS(u, w, row.data<int64_t>(), col.data<int64_t>(), {
    if (v == w)
      return 1;
  });
  return 0;
}
