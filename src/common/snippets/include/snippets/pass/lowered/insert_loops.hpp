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
 * @interface InsertLoops
 * @brief Insert explicit Loop operations into the body to process multiple data entities during one kernel execution
 * @param vector_size - the number of entities processed on one iteration of vector loop
 * @param explicit_loop_insertion - true, if we can just insert LoopBegin on inputs and LoopEnd on outputs, othwerwise
 *                           the pass goes all over the body analyzing where LoopBegin and LoopEnd should be inserted:
 *                           synchronization nodes are MatMul, Buffer and other already existing Loops.
 * @ingroup snippets
 */
class InsertLoops : public LinearIRTransformation {
    size_t m_vector_size;
    bool m_explicit_loop_insertion;
public:
    OPENVINO_RTTI("InsertLoops", "LinearIRTransformation")
    InsertLoops(size_t vector_size, bool explicit_loop_insertion);
    bool run(LoweredExprIR& linear_ir) override;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
