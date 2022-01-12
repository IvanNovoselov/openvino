// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
class AddSubtractMultiplyFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal();
    static std::shared_ptr<ov::Model> getReference();
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
