// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering.hpp"

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
> SnippetsCanonicalizationParamsInputs;

using ngraph::snippets::op::Subgraph;
class SnippetsCanonicalizationTests : public SnippetsLoweringTests, public testing::WithParamInterface<SnippetsCanonicalizationParamsInputs> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SnippetsCanonicalizationParamsInputs> obj);

protected:
    void SetUp() override;
    std::vector<Shape> input_shapes;
    Shape expected_output_shape;
    Subgraph::BlockedShapeVector input_blocked_shapes;
    Subgraph::BlockedShapeVector output_blocked_shapes;
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph