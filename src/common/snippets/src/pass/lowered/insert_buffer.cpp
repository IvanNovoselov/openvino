// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/insert_buffer.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

bool InsertBuffer::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::InsertBuffer")
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        LoweredExprPtr loop_end_expr {nullptr};
        while (auto loop = as_type_ptr<op::LoopEnd>((*expr_it)->get_node())) {
            loop_end_expr = *expr_it;
            expr_it++;
        }
        if (is_type<opset1::Result>((*expr_it)->get_node()))
                continue;
        if (loop_end_expr) {
            const auto& loop_end_inputs = loop_end_expr->get_inputs();
            // Note: Buffer ALWAYS connects to the last output register
            std::vector<TensorDescriptorPtr> last_loop_out {loop_end_inputs[loop_end_inputs.size() - 2]};
            const auto& buffer_parent = linear_ir.get_expr_by_output(last_loop_out[0])->get_node();
            auto buffer = std::make_shared<op::Buffer>(buffer_parent->output(0));
            const std::vector<TensorDescriptorPtr> buffer_outputs{std::make_shared<TensorDescriptor>(*last_loop_out[0])};
            linear_ir.insert(expr_it, std::make_shared<LoweredExpr>(buffer, last_loop_out, buffer_outputs));
            auto consumes_td = [](const LoweredExprPtr& expr, const TensorDescriptorPtr& to_find) {
                const auto& ins {expr->get_inputs()};
                return std::find(ins.begin(), ins.end(), to_find) != ins.end();
            };
            std::cerr << expr_it->get()->get_node()->get_friendly_name() << "\n";
            auto consumer = std::find_if(expr_it, linear_ir.end(), [=](const LoweredExprPtr& expr) {
                                                return consumes_td(expr, last_loop_out[0]);
                                        });
            if (consumer == linear_ir.end())
                throw ngraph_error("InsertBuffer found unconsumed tensors. Check validity of the graph");

            if (!is_type<op::Brgemm>((*consumer)->get_node())) {
                // Load is not allowed to change shape/layout, so simply copy the input tensor descriptor
                // Note that we do need this copy for assignment registers to work, since Load input is gpr, but output is vec
                std::vector<TensorDescriptorPtr> load_outputs{std::make_shared<TensorDescriptor>(*last_loop_out[0])};
                auto load = std::make_shared<op::Load>(buffer->output(0));
                linear_ir.insert(consumer, std::make_shared<LoweredExpr>(load, buffer_outputs, load_outputs));
                linear_ir.replace_input(*consumer, last_loop_out[0], load_outputs[0]);
            }
        }
    }

/*
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
    */
return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
