// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered_expr.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

// todo: add description
bool insertLoopsLowered(LoweredExprIR& linear_ir, size_t vector_size, bool explicit_loop_insertion);

} // namespace pass
} // namespace snippets
} // namespace ngraph
