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

/* The main purpose is to test that:
 * - Load/Store ops are inserted
 * - Load + BroadcastMove fuses to BroadcastLoad (not the main focus, but still had to cover; overlays with insert_movebroadcast.cpp)
 * - Proper Load/Stores are converted to scalar form to avoid invalid memory access by vector tile
 *      (temporary disabled, since corresponding PR is not merged yet)
 */

namespace ngraph {
namespace builder {
namespace subgraph {

typedef std::tuple<
        Shape, // Input shape 0
        Shape, // Input shape 1
        Shape, // Input shape 2
        Shape, // Broadcast shape 0
        Shape, // Broadcast shape 1
        Shape // Broadcast shape 2
> multiInputParams;

using ngraph::snippets::op::Subgraph;
class SnippetsLoadStoreTests : public SnippetsLoweringTests, public testing::WithParamInterface<multiInputParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<multiInputParams> obj) {
        std::vector<Shape> inputShapes(3);
        std::vector<Shape> broadcastShapes(3);
        std::tie(inputShapes[0], inputShapes[1], inputShapes[2],
                 broadcastShapes[0], broadcastShapes[1], broadcastShapes[2]) = obj.param;
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
        input_shapes.resize(3);
        broadcast_shapes.resize(3);
        std::tie(input_shapes[0], input_shapes[1], input_shapes[2],
                 broadcast_shapes[0], broadcast_shapes[1], broadcast_shapes[2]) = this->GetParam();
    }
    std::vector<Shape> input_shapes;
    std::vector<Shape> broadcast_shapes;
};

TEST_P(SnippetsLoadStoreTests, CompareWithRefEltwise) {
    const auto &f = EltwiseFunctionThreeInputsLowered(input_shapes, broadcast_shapes);
    function = f.getOriginal();
    function_ref = f.getLowered();

    prepare();
    lower();
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
