// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering.hpp"

/* The main purpose is to test whether BroadcastMove ops are inserted.
 * Conversion of Load + BroadcastMove to LoadBroadcastLoad is covered in insert_load_store.cpp
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
    static std::string getTestCaseName(testing::TestParamInfo<multiInputParams> obj);
protected:
    void SetUp() override;
    std::vector<Shape> input_shapes;
    std::vector<Shape> broadcast_shapes;
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
