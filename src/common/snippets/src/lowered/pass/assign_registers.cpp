// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/assign_registers.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"

//todo: remove debug headers
#include "snippets/lowered/pass/serialize_control_flow.hpp"


// This header is needed to avoid MSVC warning "C2039: 'inserter': is not a member of 'std'"
#include <iterator>

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void AssignRegisters::set_reg_types(LinearIR& linear_ir) {
    std::map<RegType, size_t> reg_counter;
    for (const auto& expr : linear_ir) {
        const auto op = expr->get_node();
        if (ov::is_type<op::LoopEnd>(op) ||
            ov::is_type<ov::op::v0::Result>(op)
#ifdef SNIPPETS_DEBUG_CAPS
        || ov::is_type<op::PerfCountBeginBase>(op)
        || ov::is_type<op::PerfCountEndBase>(op)
#endif
        )
            continue;
        OPENVINO_ASSERT(expr->get_output_count() == op->get_output_size(), "Incorrect count of output port descriptors!");
        for (size_t i = 0; i < expr->get_output_count(); ++i) {
            const auto reg_type = m_reg_type_mapper(op->output(i));
            const auto& reg = Reg(reg_type, reg_counter[reg_type]++);
            expr->get_output_port_descriptor(i)->set_reg(reg);
            // propogate to consumers
            for (const auto& consumer : expr->get_output_port_connector(i)->get_consumers()) {
                consumer.get_descriptor_ptr()->set_reg(reg);
            }
        }
    }
}

AssignRegisters::RegMap AssignRegisters::assign_regs_manually(LinearIR& linear_ir) const {
    RegMap manually_assigned;
    size_t io_index = 0;
    for (const auto& param : linear_ir.get_parameters()) {
        manually_assigned[param->get_output_port_connector(0)] = Reg(RegType::gpr, io_index);
        // TODO [96434]: Support shape infer ops in arbitrary place in pipeline, not just after inputs
        // shape infer ops sequence after input
        const auto& shape_infer_consumers = utils::get_first_child_shape_infer_expr_seq(param);
        for (const auto& child_shape_infer_expr : shape_infer_consumers) {
            manually_assigned[child_shape_infer_expr->get_output_port_connector(0)] = Reg(RegType::gpr, io_index);
        }
        io_index++;
    }
    for (const auto& result : linear_ir.get_results()) {
        manually_assigned[result->get_input_port_connector(0)] = Reg(RegType::gpr, io_index);
        // shape infer ops sequence before result
        const auto& shape_infer_sources = utils::get_first_parent_shape_infer_expr_seq(result);
        for (const auto& parent_shape_infer_expr : shape_infer_sources) {
            manually_assigned[parent_shape_infer_expr->get_input_port_connector(0)] = Reg(RegType::gpr, io_index);
        }
        io_index++;
    }

    auto accumulator_reg = 0lu;
    const auto reg_buffer_idx_offset = linear_ir.get_parameters().size() + linear_ir.get_results().size();
    for (const auto& expr : linear_ir.get_ops()) {
        auto op = expr->get_node();
        if (const auto& buffer = ov::as_type_ptr<op::Buffer>(op)) {
            const auto reg_idx = reg_buffer_idx_offset + buffer->get_reg_group();
            // All buffers have one common data pointer
            if (ov::is_type<op::IntermediateMemoryBuffer>(buffer)) {
                for (const auto& input : expr->get_input_port_connectors()) {
                    manually_assigned[input] = Reg(RegType::gpr, reg_idx);
                    // shape infer ops in the middle of subgraph. IntermediateMemoryBuffer is inserted before reshape as new loop should start.
                    // child shape info ops share the same memory as IntermediateMemoryBuffer.
                    const auto& shape_infer_consumers = utils::get_first_child_shape_infer_expr_seq(expr);
                    for (const auto& child_shape_infer_expr : shape_infer_consumers) {
                        manually_assigned[child_shape_infer_expr->get_input_port_connector(0)] =
                        manually_assigned[child_shape_infer_expr->get_output_port_connector(0)] =
                                Reg(RegType::gpr, reg_idx);
                    }
                }
            }
            manually_assigned[expr->get_output_port_connector(0)] = Reg(RegType::gpr, reg_idx);
        } else if (ov::is_type<op::HorizonMax>(op) || ov::is_type<op::HorizonSum>(op)) {
            // Only in ReduceDecomposition Reduce ops use HorizonMax/HorizonSum and VectorBuffer.
            // We should manually set the one vector register for VectorBuffer and Max/Sum output to simulate a accumulator
            // TODO [96351]: We should rewrite accumulator pattern using another way
            const auto& input_tensor = expr->get_input_port_connector(0);
            const auto& input_expr = input_tensor->get_source().get_expr();
            const auto& input_expr_input_tensors = input_expr->get_input_port_connectors();
            for (const auto& tensor : input_expr_input_tensors) {
                const auto parent_expr = tensor->get_source().get_expr();
                if (ov::is_type<op::Fill>(parent_expr->get_node())) {
                    if (ov::is_type<op::VectorBuffer>(parent_expr->get_input_port_connector(0)->get_source().get_expr()->get_node())) {
                        manually_assigned[tensor] = Reg(RegType::vec, accumulator_reg);
                        manually_assigned[parent_expr->get_input_port_connector(0)] = Reg(RegType::vec, accumulator_reg);
                    }
                }
            }
            manually_assigned[input_tensor] = Reg(RegType::vec, accumulator_reg);
            accumulator_reg++;
        }
    }
    return manually_assigned;
}

bool AssignRegisters::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AssignRegisters")

    //todo: remove debug prtin helper
    auto print_reg = [](const Reg& r) {
        std::string res = regTypeToStr(r.type) + " " + std::to_string(r.idx);
        return res;
    };

    const auto& exprs = linear_ir.get_ops();
//    size_t num_expressions = exprs.size();

    set_reg_types(linear_ir);
    const auto& manually_assigned_regs = assign_regs_manually(linear_ir);

    ov::snippets::lowered::pass::SerializeControlFlow("snsdebug_assign_control.xml").run(linear_ir);

    struct life_info_t {
        std::set<Reg> in;
        std::set<Reg> out;
        std::set<Reg> used;
        std::set<Reg> defined;
        int id = -1;
    };
    std::unordered_map<ExpressionPtr, life_info_t> life_info;
    auto get_reg = [](const PortDescriptorPtr& pd) { return pd->get_reg(); };
    int expr_id = 0;
    for (const auto& expr : exprs) {
        auto& life = life_info[expr];
        for (const auto& in : expr->get_input_port_descriptors())
            life.used.insert(get_reg(in));
        for (const auto& out : expr->get_output_port_descriptors())
            life.defined.insert(get_reg(out));
        life.id = expr_id++;
    }

    // todo: remove
    auto print_life_info = [&]() {
        for (const auto& expr : exprs) {
            std::cerr << expr->get_node()->get_friendly_name() << " :";
            const auto& life = life_info[expr];
            std::cerr << "\n    IO: ";
            for (auto r : life.in)
                std::cerr << print_reg(r) << ", ";
            std::cerr << " | ";
            for (auto r : life.out)
                std::cerr << print_reg(r) << ", ";
            std::cerr << "\n    UD: ";
            for (auto r : life.used)
                std::cerr << print_reg(r) << ", ";
            std::cerr << " | ";
            for (auto r : life.defined)
                std::cerr << print_reg(r) << ", ";
            std::cerr << "\n";
        }
        std::cerr << "========================================\n";
    };

    bool converged = false;
    while (!converged) {
        converged = true;
        // Note: we are traversing backwards, so next_expr is the one that was visited on the prev iteration
        ExpressionPtr next_expr = nullptr;
        for (auto expr_it = linear_ir.rbegin(); expr_it != linear_ir.rend(); expr_it++) {
            const auto& expr =  *expr_it;
            // Regs that are live on entering the operation = regs used by the op + (all other regs alive - regs defined by the op)
            // copy regs from lifeOut to lifeIn while ignoring regs in def
            // life_in = used + life_out - defined
            auto& life = life_info[expr];
            std::set<Reg> in, out;
            if (next_expr) {
                auto& next_life = life_info[next_expr];
                out.insert(next_life.in.begin(), next_life.in.end());
            }

            in = life.used;
            std::set_difference(out.begin(), out.end(),
                                life.defined.begin(), life.defined.end(),
                                std::inserter(in, in.begin()));
            converged = converged && in == life.in;
            life.in = std::move(in);
            // toodo: we don't really have to store out
            life.out = std::move(out);
            next_expr = expr;
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
    // todo: why do we need to reverse when building life intervals?
//    std::reverse(life_in_vec.begin(), life_in_vec.end());
//    std::reverse(life_in_gpr.begin(), life_in_gpr.end());
//    for (const auto& expr : exprs) {
//        auto node = expr->get_node();
//        std::cerr << node->get_friendly_name() << " : ";
//        for (auto in : expr->get_input_port_descriptors())
//            std::cerr << print_reg(in->get_reg()) << ", ";
//        std::cerr << " | ";
//        for (auto out : expr->get_output_port_descriptors())
//            std::cerr << print_reg(out->get_reg()) << ", ";
//        std::cerr << "\n";
//    }


    print_life_info();
    std::map<Reg, std::pair<int, int>> reg_life_range;
    for (const auto& expr : exprs) {
        const auto& life = life_info[expr];
        // Init start time for regs defined by this expr
        for (auto d : life.defined) {
            OPENVINO_ASSERT(reg_life_range.count(d) == 0, "Register can't be alive until it's defined");
            reg_life_range[d].first = life.id;
        }
        // Update end time for regs that are alive
        for (auto r : life.in) {
            OPENVINO_ASSERT(reg_life_range.count(r), "Register must be defined when it's live");
            auto& d = reg_life_range.at(r).second;
            d = std::max(d, life.id);
        }
    }
    // A variable live interval - is a range (start, stop) of op indexes, such that
    // the variable is alive within this range (defined but not used by the last user)
    std::map<std::pair<int, int>, Reg, by_starting> live_intervals_vec, live_intervals_gpr;
    for (const auto& regint : reg_life_range) {
        const auto& reg = regint.first;
        switch (reg.type) {
            case (RegType::gpr):
                live_intervals_gpr[regint.second] = reg;
                break;
            case (RegType::vec):
                live_intervals_vec[regint.second] = reg;
                break;
            case (RegType::undefined):
            default:
                OPENVINO_THROW("Unhabdler register type");
        }
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
                OPENVINO_THROW("Can't allocate registers for a snippet: not enough registers");
            } else {
                register_map[unique_reg] = bank.top();
                bank.pop();
                active.insert(interval_reg);
            }
        }
        return register_map;
    };
    // todo: vec_/gpr_pool are hardware-specific and should be provided by a backend, e.g. overloaded generator
    std::set<Reg> vec_pool, gpr_pool;
    std::set<Reg> assigned;
    for (const auto& r : manually_assigned_regs)
        assigned.insert(r.second);
    for (size_t i = 0; i < reg_count; i++) {
        const auto vacant_vec = Reg(RegType::vec, i);
        if (assigned.count(vacant_vec) == 0)
            vec_pool.insert(vacant_vec);
        const auto vacant_gpr = Reg(RegType::gpr, i);
        if (assigned.count(vacant_gpr) == 0)
            gpr_pool.insert(vacant_gpr);
    }

    auto unique2reused_map_vec = linescan_assign_registers(live_intervals_vec, vec_pool);
    auto unique2reused_map_gpr = linescan_assign_registers(live_intervals_gpr, gpr_pool);

    for (const auto& expr : exprs) {
        for (size_t i = 0; i < expr->get_input_count(); ++i) {
            const auto& desc = expr->get_input_port_descriptor(i);
            auto con = expr->get_input_port_connector(i);
            const auto& manual =  manually_assigned_regs.find(con);
            if (manual != manually_assigned_regs.end()) {
                desc->set_reg(manual->second);
            } else {
                const auto reg = desc->get_reg();
                if (reg.type == RegType::gpr)
                    desc->set_reg(unique2reused_map_gpr.at(reg));
                else if (reg.type == RegType::vec)
                    desc->set_reg(unique2reused_map_vec.at(reg));
            }
        }
    }
    return false;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

