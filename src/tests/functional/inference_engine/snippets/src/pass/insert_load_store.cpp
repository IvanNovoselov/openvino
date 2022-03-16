// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/insert_load_store.hpp"
#include "common_test_utils/common_utils.hpp"
#include <subgraph_lowered.hpp>

namespace ov {
namespace test {
namespace snippets {

std::string SnippetsLoadStoreTests::getTestCaseName(testing::TestParamInfo<multiInputParams> obj) {
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

void SnippetsLoadStoreTests::SetUp() {
    TransformationTestsF::SetUp();
    input_shapes.resize(3);
    broadcast_shapes.resize(3);
    std::tie(input_shapes[0], input_shapes[1], input_shapes[2],
        broadcast_shapes[0], broadcast_shapes[1], broadcast_shapes[2]) = this->GetParam();
}

TEST_P(SnippetsLoadStoreTests, EltwiseThreeInputs) {
    const auto &f = EltwiseFunctionThreeInputsLowered(input_shapes, broadcast_shapes);
    function = f.getOriginal();
    function_ref = f.getLowered();

    prepare();
    lower();
}

namespace {
using ov::Shape;
std::vector<Shape> inputShapes1{{1, 1, 2, 5, 1}, {1, 4, 1, 5, 1}};
std::vector<Shape> inputShapes2{{1, 1, 2, 5, 1}, {1, 4, 1, 5, 1}, {1, 4, 1, 5, 16}};
Shape exec_domain{1, 4, 2, 5, 16};
Shape emptyShape{};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastLoad, SnippetsLoadStoreTests,
                         ::testing::Combine(
                                 ::testing::Values(exec_domain),
                                 ::testing::ValuesIn(inputShapes1),
                                 ::testing::ValuesIn(inputShapes1),
                                 ::testing::Values(emptyShape),
                                 ::testing::Values(exec_domain),
                                 ::testing::Values(exec_domain)),
                         SnippetsLoadStoreTests::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastMove, SnippetsLoadStoreTests,
                         ::testing::Combine(
                                 ::testing::Values(exec_domain),
                                 ::testing::Values(Shape {1, 4, 1, 5, 16}),
                                 ::testing::ValuesIn(inputShapes2),
                                 ::testing::Values(emptyShape),
                                 ::testing::Values(exec_domain),
                                 ::testing::Values(exec_domain)),
                         SnippetsLoadStoreTests::getTestCaseName);
} // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov