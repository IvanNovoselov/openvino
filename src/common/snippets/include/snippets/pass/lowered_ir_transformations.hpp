// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//todo: remove relative path. check cmakelists?
#include "../lowered_expr.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

// todo: add description
bool insertTailLoop(LoweredExprIR& linear_ir);
bool assignRegisters(LoweredExprIR& linear_ir);

} // namespace pass
} // namespace snippets
} // namespace ngraph
