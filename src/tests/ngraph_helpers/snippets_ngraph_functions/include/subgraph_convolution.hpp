// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "snippets_helpers.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
/// Convolution followed by a two-input Multiply, Relu and Sqrt
/// Tokenized by attaching eltwises, but becomes non-tokenizable if Multiply is substituted with Add (CPU-specific fusing)
//    in1          in2
// Convolution   Convert
//         Multiply
//           Relu
//           Sqrt
//          Result
class ConvMulActivation : public SnippetsFunctionBase {
public:
    explicit ConvMulActivation(std::vector<Shape> inputShapes) : SnippetsFunctionBase(inputShapes) {
            NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
            NGRAPH_CHECK(input_shapes[0].size() == 4, "Only 4D input shapes are currently supported");
    }
private:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
