// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/insert_load_store.hpp"
using namespace ngraph::builder::subgraph;
namespace {
std::vector<Shape> inputShapes1{{1, 1, 2, 5, 1}, {1, 4, 1, 5, 1}};
std::vector<Shape> inputShapes2{{1, 1, 2, 5, 1}, {1, 4, 1, 5, 1}, {1, 4, 1, 5, 16}};
Shape exec_domain{1, 4, 2, 5, 16};
Shape emptyShape{};

INSTANTIATE_TEST_SUITE_P(BroadcastLoadOnInput1, SnippetsLoadStoreTests,
                         ::testing::Combine(
                                 ::testing::Values(exec_domain),
                                 ::testing::ValuesIn(inputShapes1),
                                 ::testing::ValuesIn(inputShapes1),
                                 ::testing::Values(emptyShape),
                                 ::testing::Values(exec_domain),
                                 ::testing::Values(exec_domain)),
                         SnippetsLoadStoreTests::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(BroadcastMoveOnInput1, SnippetsLoadStoreTests,
                         ::testing::Combine(
                                 ::testing::Values(exec_domain),
                                 ::testing::Values(Shape {1, 4, 1, 5, 16}),
                                 ::testing::ValuesIn(inputShapes2),
                                 ::testing::Values(emptyShape),
                                 ::testing::Values(exec_domain),
                                 ::testing::Values(exec_domain)),
                         SnippetsLoadStoreTests::getTestCaseName);
} // namespace