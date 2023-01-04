// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/insert_loops.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils.hpp"
#include "snippets/pass/loop_helpers.hpp"
#include "snippets/tensor_descriptor.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

namespace {
std::vector<bool> calculate_inner_apply_increments(const ov::PartialShape& master,
                                                   const std::vector<ov::PartialShape>& shapes) {
    // Inner Loop applies increments if a dimension is not broadcasted
    std::vector<bool> apply_increments;
    apply_increments.reserve(shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(apply_increments),
                   [=](const ov::PartialShape& ps) {
                       return utils::get_inner_dim(ps) != 1 && utils::get_inner_dim(master) != 1;
                   });
    return apply_increments;
}

std::vector<bool> calculate_outer_apply_increments(const std::vector<ov::PartialShape>& shapes) {
    // Outer Loop applies increments only if a corresponding lower dim was broadcasted (or all lower dims == 1)
    std::vector<bool> apply_increments;
    apply_increments.reserve(shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(apply_increments),
                   [=](const ov::PartialShape& ps) {
                       return utils::get_outer_dim(ps) != 1 && utils::get_inner_dim(ps) == 1;
                   });
    return apply_increments;
}

std::vector<int64_t> calculate_finalization_offsets(const ov::PartialShape& master,
                                                    const std::vector<ov::PartialShape>& shapes) {
    const auto inner_work_amount = utils::get_inner_dim(master).get_length();
    std::vector<int64_t> inner_finalization_offsets;//(shapes.size(), 0);
//    std::transform(shapes.begin(), shapes.end(), inner_finalization_offsets.begin(),
//                   [=](const ov::PartialShape& ps) {
//                       return utils::get_outer_dim(ps) == 1 && utils::get_inner_dim(ps) != 1 ? -inner_work_amount : 0;
//                   });
    for (const auto& ps : shapes) {
        int64_t offset = 0;
        if (utils::get_outer_dim(ps) == 1 && utils::get_inner_dim(ps) != 1)
            offset = -inner_work_amount;
        inner_finalization_offsets.push_back(offset);
    }
    return inner_finalization_offsets;
}

void insert_loops_explicitly(LoweredExprIR& linear_ir, const size_t vector_size) {
    ov::NodeVector body;
    ov::NodeVector body_remainder;
    ov::OutputVector body_parameters;
    std::vector<ov::Input<ov::Node>> body_results;

    // check for potential parameters for new Loop
    auto add_body_parameters = [](const std::shared_ptr<ov::Node>& op, ov::OutputVector& body_parameters) {
        for (const auto& input : op->inputs()) {
            auto parent = input.get_source_output().get_node_shared_ptr();
            if (ov::is_type<op::LoopEnd>(parent) ||
                ov::is_type<op::Buffer>(parent) ||
                ov::is_type<ov::op::v0::Parameter>(parent) ||
                ov::is_type<op::Brgemm>(parent)) {
                body_parameters.push_back(input.get_source_output());
            }
        }
    };

    // check for potential results for new Loop
    auto add_body_results = [](const std::shared_ptr<ov::Node>& op, std::vector<ov::Input<ov::Node>>& body_results) {
        for (const auto& output : op->outputs()) {
            for (const auto& target_input : output.get_target_inputs()) {
                auto child = target_input.get_node();
                if (ov::is_type<op::LoopBegin>(child) ||
                    ov::is_type<op::Buffer>(child) ||
                    ov::is_type<ov::op::v0::Result>(child) ||
                    ov::is_type<op::Brgemm>(child)) {
                    body_results.push_back(target_input);
                }
            }
        }
    };

    // check for potential missing body ops for new loop
    std::function<void(const std::shared_ptr<ov::Node>& op, ov::NodeVector& body)> add_missing_body_ops;
    add_missing_body_ops = [&](const std::shared_ptr<ov::Node>& op, ov::NodeVector& body) {
        if (body_remainder.size()) {
            for (const auto& input : op->inputs()) {
                auto parent = input.get_source_output().get_node_shared_ptr();
                auto iter = std::find(body_remainder.begin(), body_remainder.end(), parent);
                if (iter != body_remainder.end()) {
                    *std::back_inserter(body) = std::move(*iter);
                    add_missing_body_ops(parent, body);
                    add_body_parameters(parent, body_parameters);
                    add_body_results(op, body_results);
                }
            }
        }
    };

    auto wrap_body_by_loop = [&](const ov::NodeVector& body, const ov::OutputVector& body_parameters,
                                 const std::vector<ov::Input<ov::Node>>& body_results) {
        NGRAPH_CHECK(!body_parameters.empty(),
                     "The count of parameters for loop should be more than zero to create loop");
        NGRAPH_CHECK(!body_results.empty(), "The count of results for loop should be more than zero to create loop");
        std::vector<ov::PartialShape> body_shapes;
        const auto count_io = body_parameters.size() + body_results.size();
        body_shapes.reserve(count_io);
        std::transform(body_parameters.begin(), body_parameters.end(), std::back_inserter(body_shapes),
                       [](const ov::Output<ov::Node>& out) { return out.get_partial_shape(); });
        std::transform(body_results.begin(), body_results.end(), std::back_inserter(body_shapes),
                       [](const ov::Input<ov::Node>& in) { return in.get_partial_shape(); });

        auto body_master_shape = body_shapes.front();
        for (const auto& shape : body_shapes) {
            NGRAPH_CHECK(PartialShape::broadcast_merge_into(body_master_shape, shape,
                                                            ::ngraph::op::AutoBroadcastType::NUMPY),
                         "Loop input and output must be numpy broadcastable");
        }
        const auto inner_work_amount = utils::get_inner_dim(body_master_shape).get_length();
        const auto outer_work_amount = utils::get_outer_dim(body_master_shape).get_length();

        auto apply_increments = calculate_inner_apply_increments(body_master_shape, body_shapes);
        std::vector<int64_t> inner_finalization_offsets(body_shapes.size(), 0);
        if (outer_work_amount > 1) {
            inner_finalization_offsets = calculate_finalization_offsets(body_master_shape, body_shapes);
        }

        const auto& inner_loop_begin = op::insertLoopBeginAfterOutputs(body_parameters);
        const auto& inner_loop_end = op::insertLoopEndBeforeInputs(
                body_results, inner_loop_begin, inner_work_amount, vector_size,
                apply_increments, inner_finalization_offsets);
        // set internal flag to enable scalar vs vector loop optimizations
        inner_loop_end->has_outer_loop = outer_work_amount > 1;
        // Due to features of topological sort, some Constants (Scalars) may appear right after Parameters in
        // sorted ops (so it's between Parameters and LoopBegin). Consequently, ScalarEmitters would be called
        // outside the Loop, and only the first Loop iteration would yield correct data (assuming the vector reg
        // assigned to scalar will get corrupted inside the loop body). To avoid such cases, we add control dependency
        // on LoopBegin to guarantee that the constants are executed inside the Loop.
        for (const auto& n : body) {
            if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(n)) {
                c->add_control_dependency(inner_loop_begin);
            }
        }

        if (outer_work_amount > 1) {
            std::vector<bool> apply_increments = calculate_outer_apply_increments(body_shapes);
            std::vector<int64_t> outer_finalization_offsets(body_shapes.size(), 0);
            const auto& outer_loop_begin = op::insertLoopBegin(body_parameters);
            op::insertLoopEnd(body_results, outer_loop_begin, outer_work_amount, 1lu,
                              apply_increments, outer_finalization_offsets);
        }
    };


    auto is_syncronization_point = [](const std::shared_ptr<LoweredExpr>& expr) -> bool {
        return is_type<op::Buffer>(expr->get_node()) || is_type<op::Brgemm>(expr->get_node());
    };
    auto expr_requires_loop = [](const std::shared_ptr<LoweredExpr>& expr) -> bool {
        const auto& n = expr->get_node();
        return !(is_type<op::Brgemm>(n) ||
                 is_type<opset1::Parameter>(n) ||
                 is_type<opset1::Result>(n) ||
                 is_type<op::Buffer>(n));
    };

    OutputVector loop_managed_outputs;
    std::vector<bool> connected_to_buffer;
    bool buffer_is_managed = false;
    // All IR are supposed to start with LoopBegin
//    auto loop_begin_pos = std::make_shared<LoweredExpr>(std::make_shared<op::LoopBegin>());
    auto loop_begin_pos = linear_ir.begin();
    bool loop_is_active = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;
        const auto& node = expr->get_node();
        // skip explicitly inserted loop
        if (auto loop_begin = as_type_ptr<op::LoopBegin>(node)) {
            auto loop_end = loop_begin->get_loop_end();
            while ((*expr_it)->get_node() != loop_end)
                expr_it++;
            // expr_it now points to loop_end, so we need to jump to the next iterations
            continue;
        }
        // insert Loops from the last syncronization point till here
        if (expr_requires_loop(expr) && !loop_is_active) {
            loop_begin_pos = expr_it;
            loop_is_active = true;
        }
        if (loop_is_active) {
            // Load or Store
            if (is_type<op::Load>(node) || is_type<op::BroadcastLoad>(node)) {
                const auto& source = node->get_input_source_output(0);
                loop_managed_outputs.push_back(source);
                connected_to_buffer.push_back(is_type<op::Buffer>(source.get_node_shared_ptr()));
            } else if (is_type<op::Store>(node)) {
                const auto& dest = node->output(0);
                loop_managed_outputs.push_back(dest);
                connected_to_buffer.push_back(
                        is_type<op::Buffer>(dest.get_target_inputs().begin()->get_node()->shared_from_this()));
            }

            if ((is_syncronization_point(expr) || expr == linear_ir.back())) {
                if (loop_managed_outputs.empty()) {
                    loop_is_active = false;
                    continue;
                }
                auto loop_end_pos = expr_it;
                std::vector<ov::PartialShape> body_shapes;
                std::transform(loop_managed_outputs.begin(), loop_managed_outputs.end(),
                               std::back_inserter(body_shapes),
                               [](const ov::Output<ov::Node>& out) { return out.get_partial_shape(); });
                auto body_master_shape = body_shapes.front();
                for (const auto& shape : body_shapes) {
                    NGRAPH_CHECK(PartialShape::broadcast_merge_into(body_master_shape, shape,
                                                                    ::ngraph::op::AutoBroadcastType::NUMPY),
                                 "Loop managed shapes must be numpy broadcastable");
                }
                const auto inner_work_amount = utils::get_inner_dim(body_master_shape).get_length();
                const auto outer_work_amount = linear_ir.get_config().m_loop_depth == 2 ?
                                               utils::get_outer_dim(body_master_shape).get_length() :
                                               1;
                //
                std::vector<int64_t> finalization_offsets(body_shapes.size(), 0);
                if (outer_work_amount > 1) {
                    // Return pointer in case of outer dim broadcasting.
                    finalization_offsets = calculate_finalization_offsets(body_master_shape, body_shapes);
                }
                auto apply_increments = calculate_inner_apply_increments(body_master_shape, body_shapes);
                bool last_connected = false;
                for (int i = static_cast<int>(loop_managed_outputs.size()) - 1; i >= 0; i--) {
                    if (connected_to_buffer[i]) {
                        if (!last_connected) {
                            last_connected = true;
                        } else {
                            apply_increments[i] = false;
                            finalization_offsets[i] = 0;
                        }
                    }
                }
                const auto& inner_loop_begin = std::make_shared<op::LoopBegin>();
                OutputVector managed_outputs = loop_managed_outputs;
                managed_outputs.push_back(inner_loop_begin->output(0));
                const auto& inner_loop_end = std::make_shared<op::LoopEnd>(managed_outputs,
                                                                           inner_work_amount,
                                                                           vector_size,
                                                                           apply_increments,
                                                                           finalization_offsets);
                // set internal flag to enable scalar vs vector loop optimizations
                inner_loop_end->has_outer_loop = outer_work_amount > 1;
                loop_begin_pos = linear_ir.insert(loop_begin_pos, std::make_shared<LoweredExpr>(inner_loop_begin));
                linear_ir.insert(loop_end_pos, std::make_shared<LoweredExpr>(inner_loop_end));

                if (outer_work_amount > 1) {
                    std::vector<bool> apply_increments = calculate_outer_apply_increments(body_shapes);
                    std::vector<int64_t> finalization_offsets(apply_increments.size(), 0);
                    const auto& outer_loop_begin = std::make_shared<op::LoopBegin>();
                    OutputVector managed_outputs = loop_managed_outputs;
                    managed_outputs.push_back(outer_loop_begin->output(0));
                    const auto& outer_loop_end = std::make_shared<op::LoopEnd>(managed_outputs,
                                                                               outer_work_amount,
                                                                               1lu,
                                                                               apply_increments,
                                                                               finalization_offsets);
                    linear_ir.insert(loop_begin_pos, std::make_shared<LoweredExpr>(outer_loop_begin));
                    linear_ir.insert(loop_end_pos, std::make_shared<LoweredExpr>(outer_loop_end));
                }
                loop_is_active = false;
                loop_managed_outputs.clear();
                connected_to_buffer.clear();
                buffer_is_managed = false;
            }
        }
    }

    if (!body.empty()) {
        wrap_body_by_loop(body, body_parameters, body_results);
    }
}
} // namespace
InsertLoops::InsertLoops(size_t vector_size, bool explicit_loop_insertion)
    : LinearIRTransformation(), m_vector_size(vector_size), m_explicit_loop_insertion(explicit_loop_insertion) {
}

bool InsertLoops::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::insertLoops")
    const auto& lowering_config = linear_ir.get_config();
    auto master_shape = lowering_config.m_master_shape;
    auto loop_depth = lowering_config.m_loop_depth;

    if (master_shape.is_dynamic())
        throw ngraph_error("InsertLoops doesn't support dynamic shapes yet");

    const auto inner_work_amount = utils::get_inner_dim(master_shape).get_length();
    const auto outer_work_amount = loop_depth == 2 ? utils::get_outer_dim(master_shape).get_length() : 1;


    std::vector<PartialShape> ioShapes {}; //= linear_ir.get_forced_shapes();
    const auto& io_exprs = linear_ir.get_IO_ops();
    OutputVector io_outputs;
    // Here we employ the fact that Result has one output that duplicates input
    std::transform(io_exprs.begin(), io_exprs.end(), std::back_inserter(io_outputs),
                   [](const std::shared_ptr<IOLoweredExpr>& expr) { return expr->get_node()->output(0); });
    if (inner_work_amount > 0) {
        if (!m_explicit_loop_insertion) {
            const auto apply_increments = calculate_inner_apply_increments(master_shape, ioShapes);
            std::vector<int64_t> finalization_offsets(ioShapes.size(), 0);
            if (outer_work_amount > 1) {
                // Return pointer in case of outer dim broadcasting.
                finalization_offsets = calculate_finalization_offsets(master_shape, ioShapes);
            }
            const auto& inner_loop_begin = std::make_shared<op::LoopBegin>();
            OutputVector managed_outputs = io_outputs;
            managed_outputs.push_back(inner_loop_begin->output(0));
            const auto& inner_loop_end = std::make_shared<op::LoopEnd>(managed_outputs,
                                                                       inner_work_amount,
                                                                       m_vector_size,
                                                                       apply_increments,
                                                                       finalization_offsets);
            // set internal flag to enable scalar vs vector loop optimizations
            inner_loop_end->has_outer_loop = outer_work_amount > 1;
            linear_ir.insert(linear_ir.begin(), std::make_shared<LoweredExpr>(inner_loop_begin));
            linear_ir.insert(linear_ir.end(), std::make_shared<LoweredExpr>(inner_loop_end));

            if (outer_work_amount > 1) {
                std::vector<bool> apply_increments = calculate_outer_apply_increments(ioShapes);
                std::vector<int64_t> finalization_offsets(apply_increments.size(), 0);
                const auto& outer_loop_begin = std::make_shared<op::LoopBegin>();
                OutputVector managed_outputs = io_outputs;
                managed_outputs.push_back(outer_loop_begin->output(0));
                const auto& outer_loop_end = std::make_shared<op::LoopEnd>(managed_outputs,
                                                                           outer_work_amount,
                                                                           1lu,
                                                                           apply_increments,
                                                                           finalization_offsets);
                linear_ir.insert(linear_ir.begin(), std::make_shared<LoweredExpr>(outer_loop_begin));
                linear_ir.insert(linear_ir.end(), std::make_shared<LoweredExpr>(outer_loop_end));
            }
        } else {
//            std::cerr << "Explicit loop insertion is not yet supported\n";
            insert_loops_explicitly(linear_ir, m_vector_size);
        }
    }

    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

