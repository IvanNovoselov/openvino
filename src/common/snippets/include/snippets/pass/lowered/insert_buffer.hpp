// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {
// todo: update description
/**
 * @interface PropagateOffsetAndResetBuffer
 * @brief Propagates Buffer offsets to connected Load/Store (and other MemoryAccess) operations.
 *        Also, calculates the amount of data stored to the Buffer (via Store inside one or more Loops),
 *        and resets the corresponding pointer (sets negative finalization offset to the outermost LoopEnd).
 * @ingroup snippets
 */

class InsertBuffer : public LinearIRTransformation {
public:
    OPENVINO_RTTI("InsertBuffer", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
