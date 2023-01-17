// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered_expr.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

// todo: add description
bool buffer_propagate_offset_and_reset(LoweredExprIR& linear_ir);

} // namespace pass
} // namespace snippets
} // namespace ngraph
