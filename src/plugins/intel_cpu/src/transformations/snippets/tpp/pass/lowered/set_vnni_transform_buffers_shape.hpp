// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

 // todo: this transformation is very similar to SetBrgemmCopyBBuffersShape. They can be performed as a single pass
 //  when BrgemmCopyB and VnniTransform will be derived from one class
class SetVnniTransformBufferShape: public snippets::lowered::pass::RangedPass {
public:
    SetVnniTransformBufferShape() = default;
    OPENVINO_RTTI("SetVnniTransformBufferShape", "Pass");
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
