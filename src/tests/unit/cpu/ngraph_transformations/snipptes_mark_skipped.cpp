// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <subgraph_eltwise.hpp>
#include <subgraph_convolution.hpp>
#include <snippets_helpers.hpp>
#include <ngraph_transformations/snippets_mark_skipped.hpp>

using namespace MKLDNNPlugin;
using namespace ngraph::builder::subgraph;
class SnippetsMarkSkippedTests : public SnippetsCollapseSubgraphTests {
public:
    void run(bool serialize_before = false, bool serialize_after = false, bool serialize_ref = false) {
        manager.register_pass<SnippetsMarkSkipped>();
        SnippetsCollapseSubgraphTests::run(serialize_before, serialize_after, serialize_ref);
    }
};

TEST_F(SnippetsMarkSkippedTests, SkipAfterInputs_EltwiseFunction) {
    const auto &f = EltwiseFunction({{2, 3}, {1, 3}});
    function = f.getOriginal();
    // None subgraphs are expected, since the whole graph is an eltwise chain after input
    function_ref = f.getOriginal();
    run();
}

TEST_F(SnippetsMarkSkippedTests, SkipAfterInputs_MatMulEltwiseBranchesFunction) {
    const auto &f = MatMulEltwiseBranchesFunction(std::vector<Shape> {{1, 3, 4, 4}, {1, 3, 4, 4}});
    function = f.getOriginal();
    // Fully tokenizable, since inputs are followed by MatMul
    function_ref = f.getReference();
    run();
}

TEST_F(SnippetsMarkSkippedTests, SkipConvFused_ConvMulActivation) {
    const auto &f = ConvMulActivation(std::vector<Shape> {{1, 2, 16, 16}, {1, 2, 1, 16}});
    function = f.getOriginal();
    // Fully tokenizable, since Mul with 2 inputs isn't fused into Convolution
    function_ref = f.getReference();
    run();
}

TEST_F(SnippetsMarkSkippedTests, SkipConvFused_ConvSumActivation) {
    const auto &f = ConvMulActivation(std::vector<Shape> {{1, 2, 16, 16}, {1, 2, 1, 1}});
    function = f.getOriginal();
    // If I replace Multiply with Add in the original function it'll become non-tokenizable
    // due to FuseConvolutionSumAndConvolutionSumActivation fusing.
    const auto& ops = function->get_ops();
    std::shared_ptr<Node> mul = *std::find_if(ops.begin(), ops.end(),
                                             [](std::shared_ptr<Node> n) {
                                                        return is_type<op::v1::Multiply>(n);
                                                    });
    auto add = std::make_shared<op::v1::Add>(mul->get_input_source_output(0), mul->get_input_source_output(1));
    function->replace_node(mul, add);
    function_ref = function;
    run();
}
