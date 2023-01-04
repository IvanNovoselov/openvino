// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface TransposeDecomposition
 * @brief Decompose Transpose to Load + Store wrapped in several loops.
 * @ingroup snippets
 */
class SoftmaxDecomposition : public LinearIRTransformation {
public:
    OPENVINO_RTTI("SoftmaxDecomposition", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;
};

} //namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
