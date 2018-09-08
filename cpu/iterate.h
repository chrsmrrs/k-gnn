#pragma once

#define ITERATE_NODES(START, NAME, END, ...)                                   \
  {                                                                            \
    for (int64_t NAME = START; NAME < END; NAME++) {                           \
      __VA_ARGS__;                                                             \
    }                                                                          \
  }

#define ITERATE_NEIGHBORS(NODE, NAME, ROW, COL, ...)                           \
  {                                                                            \
    for (int64_t NAME##_i = ROW[NODE]; NAME##_i < ROW[NODE + 1]; NAME##_i++) { \
      auto NAME = COL[NAME##_i];                                               \
      __VA_ARGS__;                                                             \
    }                                                                          \
  }
