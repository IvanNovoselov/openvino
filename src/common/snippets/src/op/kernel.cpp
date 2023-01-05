// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/kernel.hpp"

namespace ngraph {
namespace snippets {
namespace op {

Kernel::Kernel(LoweredExprIR nested, std::shared_ptr<const ov::Model> m)
        : Op(), region(std::move(nested)), model(std::move(m)) {
}

} // namespace op
} // namespace snippets
} // namespace ngraph