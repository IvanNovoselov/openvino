// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering.hpp"

/* The main purpose is to test that:
 * - Load/Store ops are inserted
 * - Load + BroadcastMove fuses to BroadcastLoad (not the main focus, but still had to cover; overlays with insert_movebroadcast.cpp)
 * - Proper Load/Stores are converted to scalar form to avoid invalid memory access by vector tile
 *      (temporary disabled, since corresponding PR is not merged yet)
 */

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        Shape, // Input shape 0
        Shape, // Input shape 1
        Shape, // Input shape 2
        Shape, // Broadcast shape 0
        Shape, // Broadcast shape 1
        Shape // Broadcast shape 2
> multiInputParams;

class SnippetsLoadStoreTests : public SnippetsLoweringTests, public testing::WithParamInterface<multiInputParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<multiInputParams> obj);
protected:
    void SetUp() override;
    std::vector<Shape> input_shapes;
    std::vector<Shape> broadcast_shapes;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
