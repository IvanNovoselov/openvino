// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <subgraph_eltwise.hpp>
#include <snippets_helpers.hpp>
#include <ngraph_transformations/snippets_mark_skipped.hpp>

using namespace MKLDNNPlugin;
class SnippetsMarkSkippedTests : public SnippetsCollapseSubgraphTests {
public:
    void run(bool serialize_before = false, bool serialize_after = false, bool serialize_ref = false) {
        manager.register_pass<SnippetsMarkSkipped>();
        SnippetsCollapseSubgraphTests::run(serialize_before, serialize_after, serialize_ref);
    }
};

TEST_F(SnippetsMarkSkippedTests, SkipAfterInputs_EltwiseFunction) {
    function = ngraph::builder::subgraph::EltwiseFunction::getOriginal();
    // None subgraphs are expected, since the whole graph is an eltwise chain after input
    function_ref = ngraph::builder::subgraph::EltwiseFunction::getOriginal();
    run();
}

TEST_F(SnippetsMarkSkippedTests, SkipAfterInputs_MatMulEltwiseBranchesFunction) {
    function = ngraph::builder::subgraph::MatMulEltwiseBranchesFunction::getOriginal();
    // Fully tokenizable, since inputs are followed by MatMul
    function_ref = ngraph::builder::subgraph::MatMulEltwiseBranchesFunction::getReference();
    run();
}
