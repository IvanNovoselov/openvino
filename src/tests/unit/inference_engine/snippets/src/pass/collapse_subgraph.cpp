// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <subgraph_simple.hpp>
#include <snippets_helpers.hpp>
using namespace ngraph::builder::subgraph;
TEST_F(SnippetsCollapseSubgraphTests, EltwiseSubgraph) {
    const auto &f = EltwiseFunction(std::vector<Shape> {{2, 3}, {1, 3}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(SnippetsCollapseSubgraphTests, MatMulWithEltwiseBranchesSubgraph) {
    const auto &f = MatMulEltwiseBranchesFunction(std::vector<Shape> {{1, 3, 4, 4}, {1, 3, 4, 4}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(SnippetsCollapseSubgraphTests, AvoidLoopEltwiseSubgraphs) {
    const auto &f = EltwiseLogLoop(std::vector<Shape> {{2, 5}, {2, 1}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}