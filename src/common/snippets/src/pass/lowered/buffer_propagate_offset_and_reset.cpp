// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/buffer_propagate_offset_and_reset.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

bool PropagateOffsetAndResetBuffer::run(LoweredExprIR& linear_ir) {
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
                auto store_expr = linear_ir.get_expr_by_node(store);
                auto upstream_it = expr_it;
                int64_t store_integral_offset = 0;
                int num_loops = 0;
                std::vector<std::pair<std::shared_ptr<op::LoopEnd>, int>> loop_ends;
                while (*upstream_it != store_expr) {
                    upstream_it--;
                    if (auto loop_end = as_type_ptr<op::LoopEnd>((*upstream_it)->get_node())) {
                        auto loop_inputs = loop_end->inputs();
                        for (int i = 0; i < loop_inputs.size(); i++) {
                            if (loop_inputs[i].get_source_output().get_node_shared_ptr() == store) {
                                num_loops++;
                                loop_ends.emplace_back(loop_end, i);
                                break;
                            }
                        }
                    }
                }
                // Note: starting to calc integral_offset from the innermost loop
                for (auto loop_it = loop_ends.rbegin(); loop_it != loop_ends.rend(); loop_it++) {
                    const auto& loop = loop_it->first;
                    const auto index = loop_it->second;
                    const auto work_amount = static_cast<int64_t>(loop->get_work_amount());
                    // todo: here we rely on assumption that pointer increments are dense, but this obviously
                    //  is not always true. To calculate actual offsets, we should change ptr_increments, so
                    //  they hold PER DATUM increments, so total_increment = ptr_increments[i] * work_amount;
                    //  currently ptr_increment can be vector_size, 1 or arbitrary value which makes it hard
                    //  to derive the actual value, at least until scalar loops are injected
                    const auto ptr_incr = loop->get_ptr_increments()[index];
                    const auto fin_offset = loop->get_finalization_offsets()[index];
                    store_integral_offset = (store_integral_offset + ptr_incr) * work_amount + fin_offset;
                }
                if (!loop_ends.empty()) {
                    auto fin_offsets = loop_ends.front().first->get_finalization_offsets();
                    const auto index = loop_ends.front().second;
                    fin_offsets[index] -= store_integral_offset;
                    loop_ends.front().first->set_finalization_offsets(fin_offsets);
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

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
