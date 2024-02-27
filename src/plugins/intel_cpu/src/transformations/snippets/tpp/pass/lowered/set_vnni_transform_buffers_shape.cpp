// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "set_vnni_transform_buffers_shape.hpp"
#include "transformations/snippets/tpp/op/vnni_transform.hpp"
#include "snippets/snippets_isa.hpp"

namespace ov {
namespace intel_cpu {
using LinearIR = snippets::lowered::LinearIR;

bool pass::SetVnniTransformBufferShape::run(LinearIR& linear_ir,
                                            LinearIR::constExprIt begin,
                                            LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(pass::itt::domains::SnippetsTransform, "Snippets::SetVnniTransformBufferShape")

    auto get_buffer_from_output = [](const snippets::lowered::ExpressionPtr& expr, const size_t out_idx) {
        const auto& consumers = expr->get_output_port_connector(out_idx)->get_consumers();
        OPENVINO_ASSERT(consumers.size() == 1, "VnniTransform must have only 1 consumer");
        const auto buffer = ov::as_type_ptr<snippets::op::IntermediateMemoryBuffer>(consumers.begin()->get_expr()->get_node());
        OPENVINO_ASSERT(buffer, "VnniTransform consumer must be Buffer");
        return buffer;
    };

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        if (auto copy_b = ov::as_type_ptr<tpp::op::VnniTransform>(expr->get_node())) {
            const auto buffer = get_buffer_from_output(expr, 0);
            const auto& out_desc = expr->get_output_port_descriptor(0);
            buffer->set_allocation_shape(copy_b->get_data_repacking_shape(out_desc->get_shape()));
            modified = true;
        }
    }
    return modified;
}
} // namespace intel_cpu
} // namespace ov