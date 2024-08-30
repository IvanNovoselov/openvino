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

void AssignRegisters::set_reg_types(LinearIR& linear_ir, std::map<Reg, std::pair<double, double>>& reg_life_range) {
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
            const double start = expr->get_exec_num();
            double stop = start;
            // propogate to consumers
            for (const auto& consumer : expr->get_output_port_connector(i)->get_consumers()) {
                consumer.get_descriptor_ptr()->set_reg(reg);
                stop = std::max(stop, consumer.get_expr()->get_exec_num());
            }
            reg_life_range[reg] = std::make_pair(start, stop);
        }
    }
}

AssignRegisters::RegMap AssignRegisters::assign_regs_manually(LinearIR& linear_ir) const {
    RegMap manually_assigned;
    size_t io_index = 0;
    for (const auto& param : linear_ir.get_parameters()) {
        manually_assigned[param->get_output_port_descriptor(0)->get_reg()] = Reg(RegType::gpr, io_index);
        // TODO [96434]: Support shape infer ops in arbitrary place in pipeline, not just after inputs
        // shape infer ops sequence after input
        const auto& shape_infer_consumers = utils::get_first_child_shape_infer_expr_seq(param);
        for (const auto& child_shape_infer_expr : shape_infer_consumers) {
            manually_assigned[child_shape_infer_expr->get_output_port_descriptor(0)->get_reg()] = Reg(RegType::gpr, io_index);
        }
        io_index++;
    }
    for (const auto& result : linear_ir.get_results()) {
        manually_assigned[result->get_input_port_descriptor(0)->get_reg()] = Reg(RegType::gpr, io_index);
        // shape infer ops sequence before result
        const auto& shape_infer_sources = utils::get_first_parent_shape_infer_expr_seq(result);
        for (const auto& parent_shape_infer_expr : shape_infer_sources) {
            manually_assigned[parent_shape_infer_expr->get_input_port_descriptor(0)->get_reg()] = Reg(RegType::gpr, io_index);
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
                for (const auto& input : expr->get_input_port_descriptors()) {
                    manually_assigned[input->get_reg()] = Reg(RegType::gpr, reg_idx);
                    // shape infer ops in the middle of subgraph. IntermediateMemoryBuffer is inserted before reshape as new loop should start.
                    // child shape info ops share the same memory as IntermediateMemoryBuffer.
                    const auto& shape_infer_consumers = utils::get_first_child_shape_infer_expr_seq(expr);
                    for (const auto& child_shape_infer_expr : shape_infer_consumers) {
                        manually_assigned[child_shape_infer_expr->get_input_port_descriptor(0)->get_reg()] =
                        manually_assigned[child_shape_infer_expr->get_output_port_descriptor(0)->get_reg()] =
                                Reg(RegType::gpr, reg_idx);
                    }
                }
            }
            manually_assigned[expr->get_output_port_descriptor(0)->get_reg()] = Reg(RegType::gpr, reg_idx);
        } else if (ov::is_type<op::HorizonMax>(op) || ov::is_type<op::HorizonSum>(op)) {
            // Only in ReduceDecomposition Reduce ops use HorizonMax/HorizonSum and VectorBuffer.
            // We should manually set the one vector register for VectorBuffer and Max/Sum output to simulate a accumulator
            // TODO [96351]: We should rewrite accumulator pattern using another way
            const auto& input_tensor = expr->get_input_port_connector(0);
            const auto& input = input_tensor->get_source();
            for (const auto& tensor : input.get_expr()->get_input_port_connectors()) {
                const auto parent = tensor->get_source();
                const auto parent_expr = parent.get_expr();
                if (ov::is_type<op::Fill>(parent_expr->get_node())) {
                    if (ov::is_type<op::VectorBuffer>(parent_expr->get_input_port_connector(0)->get_source().get_expr()->get_node())) {
                        manually_assigned[parent.get_descriptor_ptr()->get_reg()] = Reg(RegType::vec, accumulator_reg);
                        manually_assigned[parent_expr->get_input_port_descriptor(0)->get_reg()] = Reg(RegType::vec, accumulator_reg);
                    }
                }
            }
            manually_assigned[input.get_descriptor_ptr()->get_reg()] = Reg(RegType::vec, accumulator_reg);
            accumulator_reg++;
        }
    }
    return manually_assigned;
}

bool AssignRegisters::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AssignRegisters")
    OV_ITT_SCOPE_CHAIN(FIRST_INFERENCE, taskChain, ov::pass::itt::domains::SnippetsTransform, "Snippets::AssignRegisters", "RegTypesAssign");

    //todo: remove debug prtin helper
//    auto print_reg = [](const Reg& r) {
//        std::string res = regTypeToStr(r.type) + " " + std::to_string(r.idx);
//        return res;
//    };

    const auto& exprs = linear_ir.get_ops();
//    size_t num_expressions = exprs.size();
    std::map<Reg, Interval> reg_life_range;
    set_reg_types(linear_ir, reg_life_range);
    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "Manual");
    auto assigned_reg_map = assign_regs_manually(linear_ir);

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "LifeRange");

//    ov::snippets::lowered::pass::SerializeControlFlow("snsdebug_assign_control.xml").run(linear_ir);


    // todo: remove
//    auto print_life_info = [&]() {
//        for (const auto& expr : exprs) {
//            std::cerr << expr->get_node()->get_friendly_name() << " :";
//            const auto& life = life_info[expr];
//            std::cerr << "\n    IO: ";
//            for (auto r : life.in)
//                std::cerr << print_reg(r) << ", ";
//            std::cerr << " | ";
//            for (auto r : life.out)
//                std::cerr << print_reg(r) << ", ";
//            std::cerr << "\n    UD: ";
//            for (auto r : life.used)
//                std::cerr << print_reg(r) << ", ";
//            std::cerr << " | ";
//            for (auto r : life.defined)
//                std::cerr << print_reg(r) << ", ";
//            std::cerr << "\n";
//        }
//        std::cerr << "========================================\n";
//    };

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "LifeIntervals");
    struct by_starting {
        auto operator()(const Interval& lhs, const Interval& rhs) const -> bool {
            return lhs.first < rhs.first|| (lhs.first == rhs.first && lhs.second < rhs.second);
        }
    };

    struct by_ending {
        auto operator()(const Interval& lhs, const Interval& rhs) const -> bool {
            return lhs.second < rhs.second || (lhs.second == rhs.second && lhs.first < rhs.first);
        }
    };


//    print_life_info();

    // override live range for manually assigned registers
    const auto first = linear_ir.begin()->get()->get_exec_num();
    const auto last = linear_ir.rbegin()->get()->get_exec_num();
    for (auto reg_manreg : assigned_reg_map)
        reg_life_range[reg_manreg.first] = {first, last};

    // A variable live interval - is a range (start, stop) of op indexes, such that
    // the variable is alive within this range (defined but not used by the last user)
    std::map<Interval, Reg, by_starting> live_intervals_vec, live_intervals_gpr;
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

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "Pools");
    // todo: vec_/gpr_pool are hardware-specific and should be provided by a backend, e.g. overloaded generator
    std::set<Reg> vec_pool, gpr_pool;
    std::set<Reg> assigned;
    for (const auto& reg_manreg : assigned_reg_map)
        assigned.insert(reg_manreg.second);
    for (size_t i = 0; i < reg_count; i++) {
        const auto vacant_vec = Reg(RegType::vec, i);
        if (assigned.count(vacant_vec) == 0)
            vec_pool.insert(vacant_vec);
        const auto vacant_gpr = Reg(RegType::gpr, i);
        if (assigned.count(vacant_gpr) == 0)
            gpr_pool.insert(vacant_gpr);
    }

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "Assignment");
    auto linescan_assign_registers = [](const decltype(live_intervals_vec)& live_intervals,
                                        const std::set<Reg>& reg_pool) {
        // http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf
        // todo: do we need multimap? <=> can an op have two inputs from the same op?
        std::map<Interval, Reg, by_ending> active;
        // uniquely defined register => reused reg (reduced subset enabled by reg by reusage)
        std::map<Reg, Reg> register_map;
        std::stack<Reg> bank;
        // regs are stored in ascending order in reg_pool, so walk in reverse to assign them the same way
        for (auto rit = reg_pool.crbegin(); rit != reg_pool.crend(); rit++)
            bank.push(*rit);

        Interval interval, active_interval;
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
            OPENVINO_ASSERT(active.size() != reg_pool.size(), "Can't allocate registers for a snippet: not enough registers");
            register_map[unique_reg] = bank.top();
            bank.pop();
            active.insert(interval_reg);
        }
        return register_map;
    };

    const auto& map_vec = linescan_assign_registers(live_intervals_vec, vec_pool);
    assigned_reg_map.insert(map_vec.begin(), map_vec.end());
    const auto& map_gpr = linescan_assign_registers(live_intervals_gpr, gpr_pool);
    assigned_reg_map.insert(map_gpr.begin(), map_gpr.end());

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "Postprocessing");

    for (const auto& expr : exprs) {
        for (const auto& in : expr->get_input_port_descriptors())
            in->set_reg(assigned_reg_map[in->get_reg()]);
        for (const auto& out : expr->get_output_port_descriptors())
            out->set_reg(assigned_reg_map[out->get_reg()]);
    }
    return false;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

