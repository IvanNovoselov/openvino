// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"

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
class ConvMulActivation {
public:
    static std::shared_ptr<ov::Model> getOriginal();
    static std::shared_ptr<ov::Model> getReference();
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
