// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface DomainOptimization
 * @brief Collapse input/output dimensions to balance parallel/per-thread load
 * @ingroup snippets
 */

class DomainOptimization : public snippets::lowered::pass::Pass {
    const size_t m_min_parallel_work_amount = 0;
    const size_t m_min_jit_work_amount = 0;
public:
    OPENVINO_RTTI("DomainOptimization", "Pass")
    DomainOptimization(size_t min_parallel_work_amount, size_t min_jit_work_amount);
    bool run(snippets::lowered::LinearIR& linear_ir) override;
    static bool optimize(std::vector<snippets::IShapeInferSnippets::VectorDims>& input_shapes,
                         snippets::IShapeInferSnippets::VectorDims& master_shape,
                         size_t min_parallel_work_amount,
                         size_t min_jit_work_amount);
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov