// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <subgraph_lowered.hpp>
#include <snippets_helpers.hpp>
#include "snippets/op/subgraph.hpp"
#include "lowering.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_assertions.hpp"

/* This file contains definitions of relatively simple functions (models) that will be used
 * to test snippets-specific behavior. All the functions are expected to be direct descendants of
 * SnippetsFunctionBase, so their constructors take only one (inputShapes) argument.
 */

namespace ngraph {
namespace builder {
namespace subgraph {

// todo: implement tests with 3 inputs and two outputs (aka SnippetsCanonicalizationParams3Inputs)
// Note that the expected output shape isn't necessary equal to one of the output blocked_shapes.
// For example, consider the following graph: (1, 2, 2, 1, 8) + (1, 2, 1, 1, 8) + (1, 2, 1, 5, 8) => (1, 2, 2, 1, 8) + (1, 2, 1, 5, 8).
typedef std::tuple<
        std::tuple<Shape, Subgraph::BlockedShape>, // Shape & BlockedShape for input 0
        std::tuple<Shape, Subgraph::BlockedShape>, // Shape & BlockedShape for input 0
        Subgraph::BlockedShape, // BlockedShape output shape passed to canonicalize()
        Shape // expected output Shape
> SnippetsCanonicalizationParams2Inputs;

using ngraph::snippets::op::Subgraph;
class SnippetsCanonicalizationTests2 : public SnippetsLoweringTests, public testing::WithParamInterface<SnippetsCanonicalizationParams2Inputs> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SnippetsCanonicalizationParams2Inputs> obj) {
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

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        std::vector<std::tuple<Shape, Subgraph::BlockedShape>> inputs(2);
        output_blocked_shapes.resize(1);
        std::tie(inputs[0], inputs[1], output_blocked_shapes[0], expected_output_shape) = this->GetParam();

        input_shapes = {std::get<0>(inputs[0]), std::get<0>(inputs[1])};
        input_blocked_shapes  = {std::get<1>(inputs[0]), std::get<1>(inputs[1])};
    }
    std::vector<Shape> input_shapes;
    Shape expected_output_shape;
    Subgraph::BlockedShapeVector input_blocked_shapes;
    Subgraph::BlockedShapeVector output_blocked_shapes;
};

TEST_P(SnippetsCanonicalizationTests2, CompareWithRefImpl) {
    const auto &f = AddFunction(input_shapes);
    function = f.getOriginal();
    function_ref = f.getReference();
    prepare();
    Shape canonical_output_shape = canonicalize(input_blocked_shapes, output_blocked_shapes);
    ASSERT_DIMS_EQ(canonical_output_shape, expected_output_shape);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
