// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/insert_movebroadcast.hpp"
#include "common_test_utils/common_utils.hpp"
#include <subgraph_lowered.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

std::string SnippetsMoveBroadcastTests::getTestCaseName(testing::TestParamInfo<multiInputParams> obj) {
    std::vector<Shape> inputShapes(2);
    std::vector<Shape> broadcastShapes(2);
    std::tie(inputShapes[0], inputShapes[1], broadcastShapes[0], broadcastShapes[1]) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); i++)
        result << "IS[" << i << "]=" << CommonTestUtils::vec2str(inputShapes[i]) << "_";
    for (size_t i = 0; i < broadcastShapes.size(); i++)
        result << "BS[" << i << "]=" << CommonTestUtils::vec2str(broadcastShapes[i]) << "_";
    return result.str();
}

void SnippetsMoveBroadcastTests::SetUp() {
    TransformationTestsF::SetUp();
    input_shapes.resize(2);
    broadcast_shapes.resize(2);
    std::tie(input_shapes[0], input_shapes[1], broadcast_shapes[0], broadcast_shapes[1]) = this->GetParam();
}

TEST_P(SnippetsMoveBroadcastTests, CompareWithRefImpl) {
    const auto &f = AddFunctionLoweredBroadcast(input_shapes, broadcast_shapes);
    function = f.getOriginal();
    function_ref = f.getLowered();
    prepare();
    lower();
}

namespace {
using ov::Shape;
std::vector<Shape> inputShapes0 {{1, 1, 1, 3}, {1, 1, 2, 3}, {1, 8, 1, 3}};
std::vector<Shape> inputShapes1 {{1, 8, 2, 3}};
Shape broadcastShape {1, 8, 2, 3};
Shape emptyShape {};
INSTANTIATE_TEST_SUITE_P(BroadcastOnInput0, SnippetsMoveBroadcastTests,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes0),
                                 ::testing::ValuesIn(inputShapes1),
                                 ::testing::Values(broadcastShape),
                                 ::testing::Values(emptyShape)),
                         SnippetsMoveBroadcastTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(BroadcastOnInput1, SnippetsMoveBroadcastTests,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes1),
                                 ::testing::ValuesIn(inputShapes0),
                                 ::testing::Values(emptyShape),
                                 ::testing::Values(broadcastShape)),
                         SnippetsMoveBroadcastTests::getTestCaseName);

std::vector<Shape> inputShapesBoth0 {{4, 1, 2, 3}, {1, 8, 1, 3}, {1, 1, 2, 3}};
std::vector<Shape> inputShapesBoth1 {{1, 8, 1, 3}, {4, 1, 2, 3}, {4, 8, 1, 3}};
Shape broadcastShapeBoth{4, 8, 2, 3};
std::vector<multiInputParams> params = {std::make_tuple(inputShapesBoth0[0], inputShapesBoth1[0], broadcastShapeBoth, broadcastShapeBoth),
                                        std::make_tuple(inputShapesBoth0[1], inputShapesBoth1[1], broadcastShapeBoth, broadcastShapeBoth),
                                        std::make_tuple(inputShapesBoth0[2], inputShapesBoth1[2], broadcastShapeBoth, broadcastShapeBoth)};

INSTANTIATE_TEST_SUITE_P(BroadcastOnBothInputs, SnippetsMoveBroadcastTests,
                         ::testing::ValuesIn(params),
                         SnippetsMoveBroadcastTests::getTestCaseName);

std::vector<multiInputParams> paramsNo = {std::make_tuple(inputShapesBoth0[0], inputShapesBoth0[0], emptyShape, emptyShape),
                                        std::make_tuple(inputShapesBoth0[1], inputShapesBoth0[1], emptyShape, emptyShape),
                                        std::make_tuple(inputShapesBoth0[2], inputShapesBoth0[2], emptyShape, emptyShape)};

INSTANTIATE_TEST_SUITE_P(NoBroadcast, SnippetsMoveBroadcastTests,
                         ::testing::ValuesIn(params),
                         SnippetsMoveBroadcastTests::getTestCaseName);
} // namespace
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph