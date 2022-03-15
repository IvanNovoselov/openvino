// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/canonicalization.hpp"
#include "common_test_utils/common_utils.hpp"
#include <subgraph_lowered.hpp>

namespace ov {
namespace test {
namespace snippets {

std::string SnippetsCanonicalizationTests::getTestCaseName(testing::TestParamInfo<SnippetsCanonicalizationParamsInputs> obj) {
    std::vector<std::tuple<Shape, Subgraph::BlockedShape>> inputs(2);
    Subgraph::BlockedShape output;
    Shape expectedOutput;
    std::tie(inputs[0], inputs[1], output, expectedOutput) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < inputs.size(); i++) {
        const auto &blockedshape = std::get<1>(inputs[i]);
        // input shape
        result << "IS[" << i << "]=" << CommonTestUtils::vec2str(std::get<0>(inputs[i])) << "_";
        // input blocked shape
        result << "IBS[" << i << "]=" << CommonTestUtils::vec2str(std::get<0>(blockedshape)) << "_";
        // input blocked order
        result << "IBO[" << i << "]=" << CommonTestUtils::vec2str(std::get<1>(blockedshape)) << "_";
    }
    // output blocked shape
    result << "OBS[0]=" << CommonTestUtils::vec2str(std::get<0>(output)) << "_";
    // output blocked order
    result << "OBO[0]=" << CommonTestUtils::vec2str(std::get<1>(output)) << "_";
    result << "ExpOS[0]=" << CommonTestUtils::vec2str(expectedOutput) << "_";
    return result.str();
}

void SnippetsCanonicalizationTests::SetUp() {
    TransformationTestsF::SetUp();
    std::vector<std::tuple<Shape, Subgraph::BlockedShape>> inputs(2);
    output_blocked_shapes.resize(1);
    std::tie(inputs[0], inputs[1], output_blocked_shapes[0], expected_output_shape) = this->GetParam();

    input_shapes = {std::get<0>(inputs[0]), std::get<0>(inputs[1])};
    input_blocked_shapes = {std::get<1>(inputs[0]), std::get<1>(inputs[1])};
}

TEST_P(SnippetsCanonicalizationTests, CompareWithRefImpl) {
    const auto &f = AddFunction(input_shapes);
    function = f.getOriginal();
    function_ref = f.getReference();
    prepare();
    Shape canonical_output_shape = canonicalize(input_blocked_shapes, output_blocked_shapes);
    ASSERT_DIMS_EQ(canonical_output_shape, expected_output_shape);
}

namespace {
std::vector<Shape> input_shapes;
Shape expected_output_shape;
Subgraph::BlockedShapeVector input_blocked_shapes;
Subgraph::BlockedShapeVector output_blocked_shapes;

using ov::Shape;
ov::element::Type_t prec = ov::element::f32;
std::tuple<Shape, Subgraph::BlockedShape> blockedInput0{{1, 64, 2, 5},
                                                        {{1, 4, 2, 5, 16}, {0, 1, 2, 3, 1}, prec}};
Subgraph::BlockedShape output{{1, 4, 2, 5, 16}, {0, 1, 2, 3, 1}, prec};
Shape canonical_shape{1, 4, 2, 5, 16};

std::vector<std::tuple<Shape, Subgraph::BlockedShape>> blockedInput1{{{1, 1,  2, 5}, {{1, 1, 2, 5, 1},  {0, 1, 2, 3, 1}, prec}},
                                                                     {{1, 1,  2, 1}, {{1, 1, 2, 1, 1},  {0, 1, 2, 3, 1}, prec}},
                                                                     {{1, 64, 1, 1}, {{1, 4, 1, 1, 16}, {0, 1, 2, 3, 1}, prec}}};

INSTANTIATE_TEST_SUITE_P(BroadcastBlockedBlocked, SnippetsCanonicalizationTests,
                         ::testing::Combine(
                                 ::testing::Values(blockedInput0),
                                 ::testing::ValuesIn(blockedInput1),
                                 ::testing::Values(output),
                                 ::testing::Values(canonical_shape)),
                         SnippetsCanonicalizationTests::getTestCaseName);

std::vector<std::tuple<Shape, Subgraph::BlockedShape>> planarInput1{{{1, 1, 2, 5}, {{1, 2, 5}, {0, 1, 2}, prec}},
                                                                    {{1, 1, 2, 5}, {{2, 5},    {0, 1},    prec}},
                                                                    {{1, 2, 5},    {{2, 5},    {0, 1},    prec}},
                                                                    {{2, 5},       {{2, 5},    {0, 1},    prec}},
                                                                    {{5},          {{5},       {0},       prec}}};

INSTANTIATE_TEST_SUITE_P(BroadcastBlockedPlanar, SnippetsCanonicalizationTests,
                         ::testing::Combine(
                                 ::testing::Values(blockedInput0),
                                 ::testing::ValuesIn(planarInput1),
                                 ::testing::Values(output),
                                 ::testing::Values(canonical_shape)),
                         SnippetsCanonicalizationTests::getTestCaseName);
} // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
