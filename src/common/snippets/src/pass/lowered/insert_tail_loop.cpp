// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/insert_tail_loop.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

namespace {
auto tail_transformations(LoweredExprIR::container& tail, const size_t tail_size, const LoweringConfig& config) -> void {
    auto insertFill = [tail_size](const ov::Input<ov::Node>& input) -> std::shared_ptr<ov::Node> {
        auto copyRegInfo = [](const ov::descriptor::Tensor& from, ov::descriptor::Tensor& to) -> void {
            auto rt = from.get_rt_info();
            auto reginfo = rt.find("reginfo");
            if (reginfo != rt.end()) {
                to.get_rt_info()["reginfo"] = reginfo->second;
            }
        };
        std::shared_ptr<ov::Node> fill = nullptr;
        auto& rt = input.get_rt_info();
        auto fill_rt = rt.find("set_fill");
        if (fill_rt != rt.end()) {
            const auto fill_value = fill_rt->second.as<uint32_t>();
            fill = std::make_shared<ngraph::snippets::op::Fill>(input.get_source_output(), tail_size, fill_value);
            input.get_node()->set_argument(input.get_index(), fill);
            // we should explicitly copy reg info because we insert Fill after assign register
            copyRegInfo(fill->get_input_tensor(0), fill->get_output_tensor(0));
        }
        return fill;
    };

    for (auto expr_it = tail.begin(); expr_it != tail.end(); expr_it++) {
        // We should fill vector regs by float_min and zero to have
        // correct math calculations for ReduceMax and ReduceSum in scalar case.
        // Note: We find Maximum and Add ops because HorizonMax and HorizonSum are outside Loop,
        //       so they are missed in <tail>
        auto op = (*expr_it)->get_node();
        if (config.m_need_fill_tail_register &&
            (ov::is_type<ov::op::v1::Maximum>(op) ||
             ov::is_type<ov::op::v1::Add>(op))) {
            for (auto i = 0; i < op->inputs().size(); ++i) {
                if (auto fill = insertFill(op->input(i))) {
                    auto fill_expr = std::make_shared<LoweredExpr>(fill);
                    auto reg_out = (*expr_it)->get_reg_info().first;
                    auto reg_in = (*std::prev(expr_it))->get_reg_info().second;
                    fill_expr->set_reg_info({reg_in, reg_out});
                    tail.insert(expr_it, fill_expr);
                    //updated_tile.push_back(fill);
                }
            }
        } else if (const auto memory_access = std::dynamic_pointer_cast<ngraph::snippets::op::MemoryAccess>(op)) {
            if (memory_access->get_count() != 1) {
                memory_access->set_count(tail_size);
            }
        }
    }
}
} //namespace

bool insertTailLoop(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::insertTailLoop")
    const auto& lowering_config = linear_ir.get_config();
    // *1* solo vector/tail loop + empty outer loop
    //      => skip increments (both counter & ptr) : set evaluate_once flag
    // *2* solo vector/tail loop + non-empty outer loop
    //      => skip counter increments but perform ptr increments : set evaluate_once,
    //         and perform pointer increments through finalization offsets
    // *3* vector loop(s) + one tail loop
    //      => vector as usual, tail depends on outer loop, see *1* and *2*
    auto optimize_single_evaluation = [](const std::shared_ptr<op::LoopEnd>& loop, bool force_ptr_increment = false) {
        if (loop->get_work_amount() < 2 * loop->get_increment()) {
            loop->set_evaluate_once(true);
            if (force_ptr_increment || loop->has_outer_loop) {
                std::vector<int64_t> new_finalization_offsets(loop->get_finalization_offsets());
                const auto& ptr_increments = loop->get_ptr_increments();
                for (auto i = 0; i < new_finalization_offsets.size(); i++) {
                    new_finalization_offsets[i] += ptr_increments[i];
                }
                loop->set_finalization_offsets(new_finalization_offsets);
            }
            return true;
        } else {
            return false;
        }
    };
    auto& expressions = linear_ir.get_ops();
    for (auto expr_it = expressions.begin(); expr_it != expressions.end(); expr_it++) {
        const auto& loop_begin = ov::as_type_ptr<ngraph::snippets::op::LoopBegin>((*expr_it)->get_node());
        auto loop_begin_expr_it = expr_it;
        // ignore outer loops and possible manual scalar loops
        if (loop_begin && loop_begin->get_increment() != 1) {
            LoweredExprIR vector_loop, tail_loop;
            std::shared_ptr<op::LoopEnd> vector_loop_end, tail_loop_end;
            vector_loop_end = loop_begin->get_loop_end();
            tail_loop_end = nullptr;
            auto& loop_exprs = vector_loop.get_ops();
            while ((*expr_it)->get_node() != vector_loop_end)
                loop_exprs.push_back(*expr_it++);
            // Note that exp_it points to the element AFTER loop_end
            loop_exprs.push_back(*expr_it++);
            const auto work_amount = vector_loop_end->get_work_amount();
            const auto increment = vector_loop_end->get_increment();
            const auto tail_size = work_amount % increment;
            const auto need_tail = tail_size != 0;
            const auto need_vector_loop = work_amount >= increment;
            // Note, that finalization_offsets could be modified inside optimize_single_evaluation,
            // so need to save them here to cover (evaluate_once vector with non-zero finalization_offsets + tail)
            std::vector<int64_t> tail_finalization_offsets = need_tail ? vector_loop_end->get_finalization_offsets() : std::vector<int64_t> {};
            // vector loops are required => Just copy the body, original loop is already a vector one
            if (need_vector_loop) {
                // Note that finalization offsets should be applied after the last iteration.
                // So if there is a tail, then we should apply offsets after it, but not now.
                if (need_tail)
                    vector_loop_end->set_finalization_offsets(std::vector<int64_t>(tail_finalization_offsets.size(), 0));

                if (lowering_config.m_optimize_single_evaluation) {
                    // force ptr increments if there is tail
                    optimize_single_evaluation(vector_loop_end, need_tail);
                }
            } else {
                // todo: we might want to store shared_ptr<LoweredExpr> in linear IR to allow for direct insertion/deletion
                //  Original LoweredIR => detect and copy vector tile (add/remove nodes)
                //  If LoweredIR stores pointers to LoweredExpr, then the modifications will be visible in Original LoweredIR
                // scalar tile can have different number of ops, so ve have to remove vector tile first,
                // and then insert scalar one
                expressions.erase(loop_begin_expr_it, expr_it);
            }

            // tail is required => transform the body into a tail representation
            // tail loop is fake loop because for tail we should calculate only
            // finalization offsets which are supported by LoopEnd.
            if (need_tail) {
                if (need_vector_loop) {
                    NodeMap vector_to_tail_node_map;
                    // todo: we have to clone nodes here since tail transformations can change the same nodes
                    //  (e.g. reset Load&Store count). this is a bit costy.
                    //  an alternative is no pass target machine and create emitters for vector loop here
                    //  (then we don't care if the nodes are updated)
                    tail_loop = vector_loop.deep_copy();

                } else {
                    tail_loop = std::move(vector_loop);
                }

                tail_transformations(tail_loop.get_ops(), tail_size, lowering_config);
                tail_loop_end = ov::as_type_ptr<op::LoopEnd>(tail_loop.get_ops().back()->get_node());
                tail_loop_end->set_finalization_offsets(tail_finalization_offsets);
                tail_loop_end->set_increment(tail_size);
                // ptr increments were set to the old increment, need to update them in accordance with the new one
                tail_loop_end->update_ptr_increments(static_cast<int64_t>(tail_size));
                tail_loop_end->set_work_amount(tail_size);
                tail_loop_end->has_outer_loop = vector_loop_end->has_outer_loop;

                if (lowering_config.m_optimize_single_evaluation) {
                    // tail loop is always executed once
                    optimize_single_evaluation(tail_loop_end);
                }
                linear_ir.get_ops().insert(expr_it, tail_loop.get_ops().begin(), tail_loop.get_ops().end());
            }
        }
    }
    return true;
}

} // namespace pass
} // namespace snippets
} // namespace ngraph

