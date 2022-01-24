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

/* This file contains definitions of relatively simple functions (models) that will be used
 * to test snippets-specific behavior. All the functions are expected to be direct descendants of
 * SnippetsFunctionBase, so their constructors take only one (inputShapes) argument.
 */

namespace ngraph {
namespace builder {
namespace subgraph {

typedef std::tuple<
        Shape, // Input shape 0
        Shape, // Input shape 1
        Shape, // Broadcast shape 0
        Shape // Broadcast shape 1
> multiInputParams;

using ngraph::snippets::op::Subgraph;
class SnippetsMoveBroadcastTests : public SnippetsLoweringTests, public testing::WithParamInterface<multiInputParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<multiInputParams> obj) {
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
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        input_shapes.resize(2);
        broadcast_shapes.resize(2);
        std::tie(input_shapes[0], input_shapes[1], broadcast_shapes[0], broadcast_shapes[1]) = this->GetParam();
    }
    std::vector<Shape> input_shapes;
    std::vector<Shape> broadcast_shapes;
};

TEST_P(SnippetsMoveBroadcastTests, CompareWithRefImpl) {
    const auto &f = AddFunctionLoweredBroadcast(input_shapes, broadcast_shapes);
    function = f.getOriginal();
    function_ref = f.getLowered();
    prepare();
    lower();
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
