// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_blocking.hpp"

#include "cpu_iter_handlers.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using LoopPort = LinearIR::LoopManager::LoopPort;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;
using LoopInfo = LinearIR::LoopManager::LoopInfo;
using namespace ov::snippets::lowered::pass;

BrgemmBlocking::BrgemmBlocking() : RangedPass() {}

LinearIR::constExprIt BrgemmBlocking::move_new_memory_buffer(LinearIR& linear_ir, const LinearIR::constExprIt& brgemm_it) {
    const auto& brgemm_expr = brgemm_it->get();
    const auto wsp_expr = brgemm_expr->get_input_port_connector(2)->get_source().get_expr();
    const auto wsp_buffer = ov::as_type_ptr<ov::snippets::op::NewMemoryBuffer>(wsp_expr->get_node());
    OPENVINO_ASSERT(wsp_buffer, "Incorrect Scratchpad buffer for Brgemm AMX");
    // If scratchpad with temp memory is not explicitly before Brgemm, need to move to there.
    if (wsp_expr != *std::prev(brgemm_it)) {
        const auto wsp_it = linear_ir.find(wsp_expr);
        linear_ir.move(wsp_it, brgemm_it);
    }
    return std::prev(brgemm_it);
}

LinearIR::constExprIt BrgemmBlocking::get_loop_begin_pos(LinearIR& linear_ir,
                                                         const LinearIR::constExprIt& brgemm_it,
                                                         bool shared_loop_with_repacking) {
    auto loop_begin_it = brgemm_it;
    const auto& brgemm_expr = *brgemm_it;
    const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
    if (brgemm->is_amx()) {
        loop_begin_it = move_new_memory_buffer(linear_ir, brgemm_it);
    }
    if (shared_loop_with_repacking && brgemm->is_with_data_repacking()) {
        const auto& copy_b = brgemm->get_brgemm_copy();
        const auto& copy_b_expr = linear_ir.get_expr_by_node(copy_b);
        loop_begin_it = linear_ir.find(copy_b_expr);
    }
    return loop_begin_it;
}

bool BrgemmBlocking::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmBlocking")
    const auto& loop_manager = linear_ir.get_loop_manager();
    auto blocking_loop_exists = [&](const ExpressionPtr& brgemm_expr, const std::shared_ptr<ov::intel_cpu::BrgemmCPU>& brgemm) {
        auto check_port = [&](const LoopPort& p) {
            return p.expr_port->get_expr() == brgemm_expr && ov::snippets::utils::one_of(p.dim_idx, 0ul, 1ul);
        };

        const auto& loop_ids = brgemm_expr->get_loop_ids();
        for (const auto& id : loop_ids) {
            const auto loop = loop_manager->get_loop_info(id);
            if (std::any_of(loop->get_entry_points().begin(), loop->get_entry_points().end(), check_port) ||
                std::any_of(loop->get_exit_points().begin(), loop->get_exit_points().end(), check_port)) {
                return true;
            }
        }
        return false;
    };

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& brgemm_expr = *expr_it;
        const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node());
        if (!brgemm || blocking_loop_exists(brgemm_expr, brgemm))
            continue;

        const auto& in_0_desc = brgemm_expr->get_input_port_descriptor(0);
        const auto& in_1_desc = brgemm_expr->get_input_port_descriptor(1);
        const auto& out_desc = brgemm_expr->get_output_port_descriptor(0);

        const auto& in_0_planar_dims = ov::snippets::utils::get_planar_vdims(in_0_desc->get_shape(), in_0_desc->get_layout());
        const auto& in_1_planar_dims = ov::snippets::utils::get_planar_vdims(in_1_desc->get_shape(), in_1_desc->get_layout());
        const auto& out_preordered_dims = ov::snippets::utils::get_preordered_vdims(out_desc->get_shape(), out_desc->get_layout());

        auto in_0_subtensor = in_0_desc->get_subtensor();
        auto in_1_subtensor = in_1_desc->get_subtensor();
        auto out_subtensor = out_desc->get_subtensor();

        const std::shared_ptr<ov::intel_cpu::BrgemmCopyB> copy_b = brgemm->is_with_data_repacking() ? brgemm->get_brgemm_copy() : nullptr;
        const ov::snippets::lowered::ExpressionPtr copy_b_expr = copy_b ? linear_ir.get_expr_by_node(copy_b) : nullptr;
        ov::snippets::VectorDims copy_b_subtensor = copy_b_expr ? copy_b_expr->get_input_port_descriptor(0)->get_subtensor() : ov::snippets::VectorDims();

        auto apply_m_blocking = [&]() {
            const auto& m = *(out_preordered_dims.rbegin() + 1);
            const auto block_size_m = brgemm->get_m_block_size() < m ? brgemm->get_m_block_size() : m;
            *(in_0_subtensor.rbegin() + 1) = block_size_m;
            *(out_subtensor.rbegin() + 1) = block_size_m;
            if (block_size_m == m)
                return;

            const auto loop_begin_it = get_loop_begin_pos(linear_ir, expr_it, true);
            const auto loop_end_it = std::next(expr_it);

            std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true)};
            if (brgemm->is_with_data_repacking()) {
                 entries.emplace_back(copy_b_expr->get_input_port(0), false);
            } else {
                 entries.emplace_back(brgemm_expr->get_input_port(1), false);
                if (brgemm->is_with_compensations())
                    entries.emplace_back(brgemm_expr->get_input_port(2), false);
            }
            std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
            loop_manager->mark_loop(loop_begin_it, loop_end_it, m, block_size_m, 1, entries, exits);
        };

        auto apply_n_blocking = [&]() {
            const auto& n = *out_preordered_dims.rbegin();
            const auto block_size_n = brgemm->get_n_block_size() < n ? brgemm->get_n_block_size() : n;
            *in_1_subtensor.rbegin() = block_size_n;
            *out_subtensor.rbegin() = block_size_n;
            if (brgemm->is_with_data_repacking())
                *copy_b_subtensor.rbegin() = block_size_n;
            if (block_size_n == n)
                return;

            const auto loop_begin_it = get_loop_begin_pos(linear_ir, expr_it, true);
            const auto loop_end_it = std::next(expr_it);

            std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), false)};
            if (brgemm->is_with_data_repacking()) {
                 entries.emplace_back(copy_b_expr->get_input_port(0), true);
            } else {
                 entries.emplace_back(brgemm_expr->get_input_port(1), true);
                if (brgemm->is_with_compensations())
                    entries.emplace_back(brgemm_expr->get_input_port(2), true);
            }

            std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), true)};
            loop_manager->mark_loop(loop_begin_it, loop_end_it, n, block_size_n, 0, entries, exits);
        };

        auto apply_k_blocking = [&]() {
            const auto& k = *in_0_planar_dims.rbegin();
            OPENVINO_ASSERT(k == *(in_1_planar_dims.rbegin() + 1), "Brgemm input descriptors have different K dimension value.");
            const auto block_size_k = brgemm->get_k_block_size() < k ? brgemm->get_k_block_size() : k;
            *in_0_subtensor.rbegin() = block_size_k;
            *(in_1_subtensor.rbegin() + 1) = block_size_k;
            if (brgemm->is_with_data_repacking()) {
                const auto& copy_b_desc = copy_b_expr->get_input_port_descriptor(0);
                const auto& shape = copy_b_desc->get_shape();
                const auto& layout = copy_b_desc->get_layout();
                const auto copy_b_planar_dims = ov::snippets::utils::get_planar_vdims(shape, layout);
                *++copy_b_subtensor.rbegin() = *++copy_b_planar_dims.rbegin();
            }
            if (block_size_k == k) {
                brgemm->set_beta(0.f);
                return;
            }

            const auto loop_begin_it = get_loop_begin_pos(linear_ir, expr_it, true);
            const auto loop_end_it = std::next(expr_it);

            std::vector<LoopPort> entries{LoopPort(brgemm_expr->get_input_port(0), true, 0),
                                          brgemm->is_with_data_repacking()
                                              ? LoopPort(copy_b_expr->get_input_port(0), true, 1)
                                              : LoopPort(brgemm_expr->get_input_port(1), true, 1)};
            // if (brgemm->is_with_compensations())
            //     entries.emplace_back(brgemm_expr->get_input_port(2), false, 1);
            std::vector<LoopPort> exits{LoopPort(brgemm_expr->get_output_port(0), false)};
            const auto id = loop_manager->mark_loop(loop_begin_it, loop_end_it, k, block_size_k, entries, exits);
            const auto loop_info = loop_manager->get_loop_info(id);
            loop_info->register_handler<LoopInfo::SpecificIterationHandlers::HandlerType::FIRST_ITER, SetBrgemmBeta>(0.f);
        };

        apply_k_blocking();
        apply_n_blocking();
        apply_m_blocking();

        brgemm_expr->get_input_port_descriptor(0)->set_subtensor(in_0_subtensor);
        brgemm_expr->get_input_port_descriptor(1)->set_subtensor(in_1_subtensor);
        brgemm_expr->get_output_port_descriptor(0)->set_subtensor(out_subtensor);
        if (copy_b) {
            copy_b_expr->get_input_port_descriptor(0)->set_subtensor(copy_b_subtensor);
            copy_b_expr->get_output_port_descriptor(0)->set_subtensor(copy_b_subtensor);
            if (copy_b->is_with_compensations())
                copy_b_expr->get_output_port_descriptor(1)->set_subtensor(copy_b_subtensor);
        }
        modified = true;
    }

    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov