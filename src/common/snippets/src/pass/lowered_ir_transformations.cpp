// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered_ir_transformations.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

namespace {
constexpr size_t reg_count = 16lu;
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

    for (auto expr = tail.begin(); expr != tail.end(); expr++) {
        // We should fill vector regs by float_min and zero to have
        // correct math calculations for ReduceMax and ReduceSum in scalar case.
        // Note: We find Maximum and Add ops because HorizonMax and HorizonSum are outside Loop,
        //       so they are missed in <tail>
        auto op = expr->get_node();
        if (config.m_need_fill_tail_register &&
            (ov::is_type<ov::op::v1::Maximum>(op) ||
             ov::is_type<ov::op::v1::Add>(op))) {
            for (auto i = 0; i < op->inputs().size(); ++i) {
                if (auto fill = insertFill(op->input(i))) {
                    tail.insert(expr, LoweredExpr(fill));
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
    for (auto expr = linear_ir.get_ops().begin(); expr != linear_ir.get_ops().end(); expr++) {
        auto op{expr->get_node()};

        const auto& loop_begin = ov::as_type_ptr<ngraph::snippets::op::LoopBegin>(op);
        auto loop_begin_expr = expr;

        // ignore outer loops and possible manual scalar loops
        if (loop_begin && loop_begin->get_increment() != 1) {
            LoweredExprIR vector_loop, tail_loop;
            NodeVector loop_ops;
            std::shared_ptr<op::LoopEnd> vector_loop_end, tail_loop_end;
            vector_loop_end = loop_begin->get_loop_end();
            tail_loop_end = nullptr;
            while (expr->get_node() != vector_loop_end) {
                loop_ops.push_back(expr->get_node());
                vector_loop.get_ops().push_back(*expr++);
            }
            // Note that lowered_exp points to the element AFTER loop_end
            loop_ops.push_back(expr->get_node());
            vector_loop.get_ops().push_back(*expr++);
//            const auto& vector_loop_end_expr = *lowered_expr;
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

                if (lowering_config.m_optimize_single_evaluation && false) {
                    // force ptr increments if there is tail
                    optimize_single_evaluation(vector_loop_end, need_tail);
                }
            } else {
                // todo: we might want to store shared_ptr<LoweredExpr> in linear IR to allow for direct insertion/deletion
                //  Original LoweredIR => detect and copy vector tile (add/remove nodes)
                //  If LoweredIR stores pointers to LoweredExpr, then the modifications will be visible in Original LoweredIR
                // scalar tile can have different number of ops, so ve have to remove vector tile first,
                // and then insert scalar one
                linear_ir.get_ops().erase(loop_begin_expr, expr);
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
                tail_loop_end = ov::as_type_ptr<op::LoopEnd>(tail_loop.get_ops().rbegin()->get_node());
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
                linear_ir.get_ops().insert(expr, tail_loop.get_ops().begin(), tail_loop.get_ops().end());
            }
        }
    }
    return true;
}

bool assignRegisters(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::AssignRegisters")
    using Reg = size_t;
    using tensor = std::shared_ptr<descriptor::Tensor>;
    auto& expressions = linear_ir.get_ops();
//    auto ops = linear_ir.get_ops();
    // Note that currently there are 3 types of ops:
    //  * gpr->gpr: (Parameter, Result, LoopBegin, LoopEnd) will also be Buffer?
    //  * gpr->vec: or vec->gpr Load/LoadConvert, Store/StoreConvert, BroadcastLoad etc.
    //  * vec->vec: all other "normal" operations that perform calculations on vector registers: Add, BroadcastMove, Power, etc.
    enum op_reg_type {gpr2gpr, gpr2vec, vec2gpr, vec2vec};

    auto get_op_reg_type = [](const std::shared_ptr<Node>& op) {
        if (std::dynamic_pointer_cast<opset1::Parameter>(op) ||
            std::dynamic_pointer_cast<opset1::Result>(op) ||
            std::dynamic_pointer_cast<op::LoopBegin>(op) ||
            std::dynamic_pointer_cast<op::LoopEnd>(op) ||
            std::dynamic_pointer_cast<op::Brgemm>(op) ||
            std::dynamic_pointer_cast<op::Buffer>(op))
            return gpr2gpr;
        else if (std::dynamic_pointer_cast<snippets::op::Load>(op) ||
                 std::dynamic_pointer_cast<snippets::op::BroadcastLoad>(op))
            return gpr2vec;
        else if (std::dynamic_pointer_cast<snippets::op::Store>(op))
            return vec2gpr;
        else
            return vec2vec;
    };
    std::vector<std::pair<op_reg_type, LoweredExpr&>> typed_ops;
    NodeVector ops;
    Reg num_parameters = 0;
    Reg num_results = 0;
    Reg num_expressions = 0;
    for (auto& expr : expressions) {
        auto op = expr.get_node();
        auto reg_type = get_op_reg_type(op);
        typed_ops.emplace_back(reg_type, expr);
        num_parameters += is_type<opset1::Parameter>(op);
        num_results += is_type<opset1::Result>(op);
        ops.push_back(op);
        num_expressions++;
    }
    size_t counter_vec = 0;
    size_t counter_gpr = 0;
    std::map<tensor, Reg> regs_vec, regs_gpr;
    // Define a set of immune tensors that will be ignored by auto reg allocation => their reg allocation is done manually
    std::map<tensor, Reg> manually_assigned_gprs, manually_assigned_vecs;
    const auto IS_MANUALLY_ALLOCATED_REG = SIZE_MAX;
    auto accumulator_reg = 0lu;
    Reg param_index = 0;
    Reg result_index = 0;
    for (auto& expr : expressions) {
        auto op = expr.get_node();
        if (const auto& param = ov::as_type_ptr<ov::op::v0::Parameter>(op)) {
            manually_assigned_gprs[op->output(0).get_tensor_ptr()] = param_index++;
        } else if (const auto& result = ov::as_type_ptr<opset1::Result>(op)) {
            // here we use the fact that Result input & output tensors are identical by construction
            manually_assigned_gprs[op->output(0).get_tensor_ptr()] = num_parameters + result_index++;
        } else if (const auto& buffer = ov::as_type_ptr<op::Buffer>(op)) {
            // All buffers have one common data pointer
            manually_assigned_gprs[op->input(0).get_tensor_ptr()] =
                    static_cast<Reg>(num_results + num_parameters);
            manually_assigned_gprs[op->output(0).get_tensor_ptr()] =
                    static_cast<Reg>(num_results + num_parameters);
        } else if (ov::is_type<op::HorizonMax>(op) || ov::is_type<op::HorizonSum>(op)) {
            // Only in SoftmaxDecomposition ReduceMax and ReduceSum use HorizonMax/HorizonSum and VectorBuffer.
            // We should manually set the one vector register for VectorBuffer and Max/Sum output to simulate a accumulator
            // TODO [96351]: We should rewrite accumulator pattern using another way
            const auto input = op->get_input_node_shared_ptr(0); // input - it's accumulator math op: Add or Max
            for (size_t i = 0; i < input->get_input_size(); ++i) {
                if (ov::is_type<op::VectorBuffer>(input->get_input_node_shared_ptr(i))) {
                    manually_assigned_vecs[input->input(i).get_tensor_ptr()] =
                            static_cast<Reg>(accumulator_reg);
                }
            }

            manually_assigned_vecs[input->output(0).get_tensor_ptr()] =
                    static_cast<Reg>(accumulator_reg);
            manually_assigned_vecs[op->output(0).get_tensor_ptr()] =
                    static_cast<Reg>(accumulator_reg);

            // If there is Broadcast, it should have the same register as Horizon op
            // because it's a result of the accumulator as well
            for (auto& out : op->output(0).get_target_inputs()) {
                const auto child = out.get_node()->shared_from_this();
                if (ov::is_type<op::BroadcastMove>(child)) {
                    manually_assigned_vecs[child->output(0).get_tensor_ptr()] =
                            static_cast<Reg>(accumulator_reg);
                }
            }
            accumulator_reg++;
        }
    }
    auto enumerate_out_tensors = [IS_MANUALLY_ALLOCATED_REG] (const std::shared_ptr<const ov::Node>& op,
                                                              decltype(regs_vec)& reg_map,
                                                              const std::map<tensor, Reg>& manually_assigned_regs,
                                                              size_t& counter) {
        for (const auto& output : op->outputs()) {
            const auto& t = output.get_tensor_ptr();
            // Note that some ops might have identical input&output tensors (Result and Tile* for ex.)
            // so we have to check that the tensor has not been enumerated already
            if (reg_map.count(t) == 0) {
                reg_map[t] = manually_assigned_regs.count(t) == 0 ? counter++ : IS_MANUALLY_ALLOCATED_REG;
            }
        }
    };
    for (const auto& t_op : typed_ops) {
        switch (t_op.first) {
            case vec2vec:
            case gpr2vec:
                enumerate_out_tensors(t_op.second.get_node(), regs_vec, manually_assigned_vecs, counter_vec);
                break;
            case gpr2gpr:
            case vec2gpr:
                enumerate_out_tensors(t_op.second.get_node(), regs_gpr, manually_assigned_gprs, counter_gpr);
                break;
        }
    }
    // todo: make one for gpr and one for vector
    std::vector<std::set<Reg>> used_gpr(num_expressions, std::set<Reg>()); // used = used as an input
    std::vector<std::set<Reg>> defined_gpr(num_expressions, std::set<Reg>()); // defined = used as output
    std::vector<std::set<Reg>> used_vec(num_expressions, std::set<Reg>());
    std::vector<std::set<Reg>> defined_vec(num_expressions, std::set<Reg>());

    auto tensor2reg = [IS_MANUALLY_ALLOCATED_REG] (const std::vector<tensor>& tensors, const std::map<tensor, Reg>& reg_map) {
        std::set<Reg> result;
        for (const auto& t : tensors) {
            if (reg_map.count(t) == 0)
                throw ngraph::ngraph_error("Assign registers: attempt to access not enumerated tensor");
            Reg reg_id = reg_map.at(t);
            if (reg_id != IS_MANUALLY_ALLOCATED_REG)
                result.insert(reg_id);
        }
        return result;
    };
    for (size_t i = 0; i < typed_ops.size(); i++) {
        const auto& t_op = typed_ops[i];
        std::vector<tensor> used_tensors, defined_tensors;
        for (const auto& in : t_op.second.get_node()->inputs())
            used_tensors.push_back(in.get_tensor_ptr());
        for (const auto& out : t_op.second.get_node()->outputs())
            defined_tensors.push_back(out.get_tensor_ptr());
        switch (t_op.first) {
            case vec2vec:
                used_vec[i] = tensor2reg(used_tensors, regs_vec);
                defined_vec[i] = tensor2reg(defined_tensors, regs_vec);
                break;
            case gpr2gpr:
                used_gpr[i] = tensor2reg(used_tensors, regs_gpr);
                defined_gpr[i] = tensor2reg(defined_tensors, regs_gpr);
                break;
            case gpr2vec:
                used_gpr[i] = tensor2reg(used_tensors, regs_gpr);
                defined_vec[i] = tensor2reg(defined_tensors, regs_vec);
                break;
            case vec2gpr:
                used_vec[i] = tensor2reg(used_tensors, regs_vec);
                defined_gpr[i] = tensor2reg(defined_tensors, regs_gpr);
                break;
        }
    }

    // define life intervals
    // liveOut[i] - regs that are live on exit from i-th (topologically ordered) operation
    // liveIn[i] - regs that are live on entering the i-th (topologically ordered) operation
    std::vector<std::set<Reg>> life_in_vec(std::move(used_vec));
    std::vector<std::set<Reg>> life_out_vec(typed_ops.size(), std::set<Reg>());
    std::vector<std::set<Reg>> life_in_gpr(std::move(used_gpr));
    std::vector<std::set<Reg>> life_out_gpr(typed_ops.size(), std::set<Reg>());

    // todo: this part if O(N*N), so it's slow for large subgraphs. Can we simplify it? At least add an early stopping criteria
    for (size_t i = 0; i < typed_ops.size(); i++) {
        for (size_t n = 0; n < typed_ops.size(); n++) {
            // Regs that are live on entering the operation = regs used by the op + (all other regs alive - regs defined by the op)
            // copy regs from lifeOut to lifeIn while ignoring regs in def
            std::set_difference(life_out_gpr[n].begin(), life_out_gpr[n].end(),
                                defined_gpr[n].begin(), defined_gpr[n].end(),
                                std::inserter(life_in_gpr[n], life_in_gpr[n].begin()));
            std::set_difference(life_out_vec[n].begin(), life_out_vec[n].end(),
                                defined_vec[n].begin(), defined_vec[n].end(),
                                std::inserter(life_in_vec[n], life_in_vec[n].begin()));
        }
        for (size_t n = 0; n < typed_ops.size(); n++) {
            auto op = typed_ops[n].second.get_node();
            for (const auto& out : op->outputs()) {
                for (const auto& port : out.get_target_inputs()) {
                    auto k = std::find(ops.begin(), ops.end(), port.get_node()->shared_from_this()) - ops.begin();
                    if (k == ops.size())
                        throw ngraph_error("assign registers can't find target op in the body");
                    switch (typed_ops[k].first) {
                        case vec2vec:
                        case vec2gpr:
                            life_out_vec[n].insert(life_in_vec[k].begin(), life_in_vec[k].end());
                            break;
                        case gpr2gpr:
                        case gpr2vec:
                            life_out_gpr[n].insert(life_in_gpr[k].begin(), life_in_gpr[k].end());
                            break;
                    }
                }
            }
        }
    }
    struct by_starting {
        auto operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const -> bool {
            return lhs.first < rhs.first|| (lhs.first == rhs.first && lhs.second < rhs.second);
        }
    };

    struct by_ending {
        auto operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const -> bool {
            return lhs.second < rhs.second || (lhs.second == rhs.second && lhs.first < rhs.first);
        }
    };
    // A variable live interval - is a range (start, stop) of op indexes, such that
    // the variable is alive within this range (defined but not used by the last user)
    std::map<std::pair<int, int>, Reg, by_starting> live_intervals_vec, live_intervals_gpr;

    std::reverse(life_in_vec.begin(), life_in_vec.end());
    std::reverse(life_in_gpr.begin(), life_in_gpr.end());
    auto find_last_use = [](decltype(life_in_gpr) life_in, int i) -> int {
        int ln = static_cast<int>(life_in.size()) - 1;
        for (auto& x : life_in) {
            if (x.find(i) != x.end()) {
                return ln;
            }
            ln--;
        }
        return i;
    };
    for (int i = 0; i < static_cast<int>(typed_ops.size()); i++) {
        for (const auto& def : defined_vec[i])
            live_intervals_vec[std::make_pair(i, find_last_use(life_in_vec, static_cast<int>(def)))] = def;
        for (const auto& def : defined_gpr[i])
            live_intervals_gpr[std::make_pair(i, find_last_use(life_in_gpr, static_cast<int>(def)))] = def;
    }

    auto linescan_assign_registers = [](const decltype(live_intervals_vec)& live_intervals,
                                        const std::set<Reg>& reg_pool) {
        // http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf
        // todo: do we need multimap? <=> can an op have two inputs from the same op?
        std::map<std::pair<int, int>, Reg, by_ending> active;
        // uniquely defined register => reused reg (reduced subset enabled by reg by reusage)
        std::map<Reg, Reg> register_map;
        std::stack<Reg> bank;
        // regs are stored in ascending order in reg_pool, so walk in reverse to assign them the same way
        for (auto rit = reg_pool.crbegin(); rit != reg_pool.crend(); rit++)
            bank.push(*rit);

        std::pair<int, int> interval, active_interval;
        Reg unique_reg, active_unique_reg;
        for (const auto& interval_reg : live_intervals) {
            std::tie(interval, unique_reg) = interval_reg;
            // check expired
            while (!active.empty()) {
                std::tie(active_interval, active_unique_reg) = *active.begin();
                // if end of active interval has not passed yet => stop removing actives since they are sorted by end
                if (active_interval.second >= interval.first) {
                    break;
                }
                active.erase(active_interval);
                bank.push(register_map[active_unique_reg]);
            }
            // allocate
            if (active.size() == reg_pool.size()) {
                // todo: if it is LoopBegin or LoopEnd that requires gpr, and we don't have any in the pool,
                //  then assign SIZE_MAX-1 as a flag to spill a reg inside emitter
                throw ngraph::ngraph_error("can't allocate registers for a snippet ");
            } else {
                register_map[unique_reg] = bank.top();
                bank.pop();
                active.insert(interval_reg);
            }
        }
        return register_map;
    };
    // todo: vec_/gpr_pool are hardware-specific and should be provided by a backend, e.g. overloaded generator
    std::set<Reg> vec_pool;
    for (Reg i = 0; i < reg_count; i++)
        vec_pool.insert(i);
    std::set<Reg> gpr_pool(vec_pool);
    for (const auto& t_reg : manually_assigned_vecs)
        vec_pool.erase(t_reg.second);
    for (const auto& t_reg : manually_assigned_gprs)
        gpr_pool.erase(t_reg.second);
    auto unique2reused_map_vec = linescan_assign_registers(live_intervals_vec, vec_pool);
    auto unique2reused_map_gpr = linescan_assign_registers(live_intervals_gpr, gpr_pool);

    std::map<tensor, Reg> assigned_regs(std::move(manually_assigned_gprs));
    assigned_regs.insert(manually_assigned_vecs.begin(), manually_assigned_vecs.end());
    auto register_assigned_regs = [IS_MANUALLY_ALLOCATED_REG, &assigned_regs](const std::map<tensor, Reg>& unique_regs,
                                                                              const std::map<Reg, Reg>& unique2reused) {
        for (const auto& reg : unique_regs) {
            if (reg.second == IS_MANUALLY_ALLOCATED_REG)
                continue;
            if (unique2reused.count(reg.second) == 0)
                throw ngraph::ngraph_error("Assign registers failed to allocate register for a tensor");
            assigned_regs[reg.first] = unique2reused.at(reg.second);
        }
    };
    register_assigned_regs(regs_vec, unique2reused_map_vec);
    register_assigned_regs(regs_gpr, unique2reused_map_gpr);

    for (auto& t_op : typed_ops) {
        RegInfo rinfo;
        for (const auto& in : t_op.second.get_node()->inputs()) {
            rinfo.first.push_back(assigned_regs[in.get_tensor_ptr()]);
        }
        for (const auto& out : t_op.second.get_node()->outputs()) {
            rinfo.second.push_back(assigned_regs[out.get_tensor_ptr()]);
        }
        t_op.second.set_reg_info(rinfo);
    }
    return false;
}

} // namespace pass
} // namespace snippets
} // namespace ngraph

