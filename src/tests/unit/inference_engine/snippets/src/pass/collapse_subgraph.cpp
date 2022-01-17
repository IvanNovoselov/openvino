// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <subgraph_eltwise.hpp>
#include <snippets_helpers.hpp>

TEST_F(SnippetsCollapseSubgraphTests, EltwiseSubgraph) {
    function = ngraph::builder::subgraph::EltwiseFunction::getOriginal();
    function_ref = ngraph::builder::subgraph::EltwiseFunction::getReference();
    run();
}

TEST_F(SnippetsCollapseSubgraphTests, MatMulWithEltwiseBranchesSubgraph) {
    function = ngraph::builder::subgraph::MatMulEltwiseBranchesFunction::getOriginal();
    function_ref = ngraph::builder::subgraph::MatMulEltwiseBranchesFunction::getReference();
    run();
}

TEST_F(SnippetsCollapseSubgraphTests, AvoidLoopEltwiseSubgraphs) {
    function = ngraph::builder::subgraph::EltwiseLogLoop::getOriginal();
    function_ref = ngraph::builder::subgraph::EltwiseLogLoop::getReference();
    run();
}