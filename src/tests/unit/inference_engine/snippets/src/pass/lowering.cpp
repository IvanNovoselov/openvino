// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "pass/lowering.hpp"

using namespace ngraph::builder::subgraph;
TEST_F(SnippetsLoweringTests, EltwiseSubgraph) {
//    std::vector<Shape> inputShapes{{1, 32, 2, 3}, {1, 32, 1, 3}};
    Subgraph::BlockedShape is1{{1, 4, 5, 3, 8}, {0, 1, 2, 3, 1}, ngraph::element::f32};
    Subgraph::BlockedShape is2{{1, 4, 1, 3, 8}, {0, 1, 2, 3, 1}, ngraph::element::f32};
    std::vector<Shape> inputShapes{std::get<0>(is1), std::get<0>(is2)};
    // input for internal constant
    Subgraph::BlockedShape is3{{1, 4, 1, 3, 8}, {0, 1, 2, 3, 1}, ngraph::element::f32};
    Subgraph::BlockedShapeVector inputBlockedShapes{is1, is2, is3};
    Subgraph::BlockedShapeVector outputBlockedShapes{{{1, 4, 5, 3, 8}, {0, 1, 2, 3, 1}, ngraph::element::f32}};
    const auto &f = EltwiseFunctionLowered(inputShapes, inputBlockedShapes, outputBlockedShapes);
    input_blocked_shapes = f.getInputBlockedShapes();
    output_blocked_shapes = f.getOutputBlockedShapes();
    function = f.getOriginal();
    function_ref = f.getLowered();
    run();
}

TEST_F(SnippetsLoweringTests, AddNoBroadcast) {
    std::vector<Shape> inputShapes{{1, 8, 2, 3}, {1, 8, 2, 3}};
    const auto &f = AddFunctionLoweredBroadcast(inputShapes, {{}, {}});
    function = f.getOriginal();
    function_ref = f.getLowered();
    run();
}

TEST_F(SnippetsLoweringTests, AddBroadcast0) {
    std::vector<Shape> inputShapes{{1, 1, 2, 3}, {1, 8, 2, 3}};
    const auto &f = AddFunctionLoweredBroadcast(inputShapes, {{1, 8, 2, 3}, {}});
    function = f.getOriginal();
    function_ref = f.getLowered();
    run();
}

TEST_F(SnippetsLoweringTests, AddBroadcast1) {
    std::vector<Shape> inputShapes{{1, 8, 2, 3}, {1, 8, 1, 3}};
    const auto &f = AddFunctionLoweredBroadcast(inputShapes, {{}, {1, 8, 2, 3}});
    function = f.getOriginal();
    function_ref = f.getLowered();
    run();
}

TEST_F(SnippetsLoweringTests, AddBroadcast12) {
    std::vector<Shape> inputShapes{{1, 1, 2, 2}, {1, 8, 1, 2}};
    const auto &f = AddFunctionLoweredBroadcast(inputShapes, {{1, 8, 2, 2}, {1, 8, 2, 2}});
    function = f.getOriginal();
    function_ref = f.getLowered();
    run();
}