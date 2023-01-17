// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/buffer_propagate_offset_and_reset.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
bool buffer_propagate_offset_and_reset(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::buffer_propagate_offset_and_reset")
    std::vector<LoweredExprIR::container::iterator> exprs_to_del;
    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& op = (*expr_it)->get_node();
        if (auto buffer = as_type_ptr<op::Buffer>(op)) {
            if (buffer->get_input_size() != 1 || buffer->get_output_size() != 1)
                throw ngraph_error("Buffer is expected to have exactly one input and one output");
            const auto offset = buffer->get_offset();
            const auto& parent = buffer->get_input_node_shared_ptr(0);
            const auto& child = buffer->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            if (const auto& store = ov::as_type_ptr<op::Store>(parent)) {
                store->set_offset(offset);
                auto store_expr = linear_ir.get_expr_by_node(buffer);
                auto upstream_it = expr_it;
                std::shared_ptr<op::LoopEnd> last_LoopEnd = nullptr;
                int last_LoopEnd_store_index = -1;
                int64_t store_integral_offset = 0;
                int num_loops = 0;
                while (*upstream_it != store_expr) {
                    upstream_it--;
                    if (auto loop_end = as_type_ptr<op::LoopEnd>((*upstream_it)->get_node())) {
                        auto loop_inputs = loop_end->inputs();
                        for (int i = 0; i < loop_inputs.size(); i++) {
                            if (loop_inputs[i].get_source_output().get_node_shared_ptr() == store) {
                                num_loops++;
                                if (!last_LoopEnd) {
                                    last_LoopEnd = loop_end;
                                    last_LoopEnd_store_index = i;
                                }
                                const auto& finalization_offsets = loop_end->get_finalization_offsets();
                                // todo: here we rely on assumption that pointer increments are dense, but this obviously
                                //  is not always true. To calculate actual offsets, we should change ptr_increments, so
                                //  they hold PER DATUM increments, so total_increment = ptr_increments[i] * work_amount;
                                //  currently ptr_increment can be vector_size, 1 or arbitrary value which makes it hard
                                //  to derive the actual value, at least until scalar loops are injected
                                const auto total_ptr_increment = loop_end->get_work_amount();
                                store_integral_offset += finalization_offsets[i] + static_cast<int64_t>(total_ptr_increment);
                                break;
                            }
                        }
                    }
                }
                if (last_LoopEnd && last_LoopEnd_store_index >= 0) {
                    auto last_finalization_offsets = last_LoopEnd->get_finalization_offsets();
                    last_finalization_offsets[last_LoopEnd_store_index] -= store_integral_offset;
                    last_LoopEnd->set_finalization_offsets(last_finalization_offsets);
                }
            }
            if (const auto& load = ov::as_type_ptr<op::Load>(child)) {
                load->set_offset(offset);
            }
            modified = true;
        }
    }
    return modified;
}

} // namespace pass
} // namespace snippets
} // namespace ngraph

