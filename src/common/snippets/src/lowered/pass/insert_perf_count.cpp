// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_perf_count.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"
#include "../../../plugins/intel_cpu/src/transformations/snippets/x64/op/perf_count_rdtsc.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

namespace {
bool insert_around_brgemms(LinearIR& linear_ir) {
    size_t seq_number = 0;
    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); expr_it++) {
        // if (ov::is_type<snippets::op::LoopBegin>(expr_it->get()->get_node()) &&
        //     ov::is_type<snippets::op::Brgemm>(std::next(expr_it)->get()->get_node()) &&
        //     ov::is_type<snippets::op::LoopEnd>(std::next(expr_it, 2)->get()->get_node())) {
        if (ov::is_type<snippets::op::Brgemm>(expr_it->get()->get_node())) {
            const auto& perf_count_begin_pos = expr_it;
            // const auto& perf_count_end_pos = std::next(expr_it, 3);
            const auto& perf_count_end_pos = std::next(expr_it, 1);
            // insert perf_count_begin after last parameter
            // linear_ir.insert has insert before behavior, need move to next.
            const auto& perf_count_begin = std::make_shared<intel_cpu::PerfCountRdtscBegin>();
            perf_count_begin->set_friendly_name(std::string("PerfCount_Begin_") + std::to_string(seq_number));
            const auto empty_connectors = std::vector<PortConnectorPtr>{};
            const auto& perf_count_begin_expr = linear_ir.create_expression(perf_count_begin, empty_connectors);
            linear_ir.insert(perf_count_begin_pos, perf_count_begin_expr);

            // insert perf_count_end before first result
            const auto& perf_count_end = std::make_shared<intel_cpu::PerfCountRdtscEnd>(perf_count_begin->output(0));
            perf_count_end->set_friendly_name(std::string("PerfCount_End_") + std::to_string(seq_number));
            const auto& perf_count_end_expr = linear_ir.create_expression(perf_count_end, empty_connectors);
            expr_it = linear_ir.insert(perf_count_end_pos, perf_count_end_expr);
            seq_number++;
        }
    }

    return true;
}

bool wrap_whole_body(LinearIR& linear_ir) {
    // mark perf_count_begin and perf_count_end position
    auto perf_count_begin_pos = linear_ir.cbegin();
    auto perf_count_end_pos = linear_ir.cend();
    const auto empty_connectors = std::vector<PortConnectorPtr>{};
    perf_count_begin_pos = std::next(perf_count_begin_pos);
    const auto& perf_count_begin = std::make_shared<intel_cpu::PerfCountRdtscBegin>();
    const auto& perf_count_begin_expr = linear_ir.create_expression(perf_count_begin, empty_connectors);
    linear_ir.insert(perf_count_begin_pos, perf_count_begin_expr);
    perf_count_begin->set_friendly_name("PerfCount_Begin_0");

    // insert perf_count_end before first result
    const auto& perf_count_end = std::make_shared<intel_cpu::PerfCountRdtscEnd>(perf_count_begin->output(0));
    perf_count_end->set_friendly_name("PerfCount_End_0");
    const auto& perf_count_end_expr = linear_ir.create_expression(perf_count_end, empty_connectors);
    linear_ir.insert(perf_count_end_pos, perf_count_end_expr);

    return true;
}

bool wrap_selected(LinearIR& linear_ir, const std::map<std::string, std::string>& wrap_op_names) {
    size_t seq_number = 0;
    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); expr_it++) {
        const auto& op_name = expr_it->get()->get_node()->get_friendly_name();
        const auto& found = wrap_op_names.find(op_name);
        if (found != wrap_op_names.end()) {
            const auto perf_count_begin_pos = expr_it;
            auto perf_count_end_pos = expr_it;
            while (perf_count_end_pos->get()->get_node()->get_friendly_name() != found->second &&
                   perf_count_end_pos != linear_ir.cend()) {
                   perf_count_end_pos++;
            }
            OPENVINO_ASSERT(perf_count_end_pos != linear_ir.cend(), "Failed to find requested op name to insert PerfCountEnd");
            perf_count_end_pos++;
            // insert perf_count_begin after last parameter
            // linear_ir.insert has insert before behavior, need move to next.
            const auto& perf_count_begin = std::make_shared<intel_cpu::PerfCountRdtscBegin>();
            perf_count_begin->set_friendly_name(std::string("PerfCount_Begin_") + std::to_string(seq_number));
            const auto empty_connectors = std::vector<PortConnectorPtr>{};
            const auto& perf_count_begin_expr = linear_ir.create_expression(perf_count_begin, empty_connectors);
            linear_ir.insert(perf_count_begin_pos, perf_count_begin_expr);

            // insert perf_count_end before first result
            const auto& perf_count_end = std::make_shared<intel_cpu::PerfCountRdtscEnd>(perf_count_begin->output(0));
            perf_count_end->set_friendly_name(std::string("PerfCount_End_") + std::to_string(seq_number));
            const auto& perf_count_end_expr = linear_ir.create_expression(perf_count_end, empty_connectors);
            linear_ir.insert(perf_count_end_pos, perf_count_end_expr);
            seq_number++;
        }
    }
    return true;
}

} // namespace

bool InsertPerfCount::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertPerfCount")
    if (linear_ir.empty())
        return false;
    ///// Transpose loop
    // TPP ///
    // const std::map<std::string, std::string>& wrap_op_names{{"LoopBegin_2860", "LoopEnd_2861"}};
    // CPU ///
    // const std::map<std::string, std::string>& wrap_op_names{{"LoopBegin_2865", "LoopEnd_2866"}};
    // TPP ///
    ///// Additivity test Single run
    // TPP ///
    // const std::map<std::string, std::string>& wrap_op_names{{"Transpose_18", "Result_2625"},
    //                                                         {"LoopBegin_2860", "LoopEnd_2861"},
    //                                                         {"LoopBegin_2864", "LoopEnd_2865"}};
    // CPU ///
    // const std::map<std::string, std::string>& wrap_op_names{{"Transpose_18", "Result_2625"},
    //                                                         {"LoopBegin_2865", "LoopEnd_2866"},
    //                                                         {"LoopBegin_2869", "LoopEnd_2870"}};
    ///// Additivity test Single run inside cycle
    // TPP ///
    const std::map<std::string, std::string>& wrap_op_names{{"LoopBegin_2866", "LoopEnd_2867"},
                                                            {"IntermediateMemoryBuffer_2856", "IntermediateMemoryBuffer_2855"},
                                                            {"LoopBegin_2872", "LoopEnd_2873"}};
    //CPU ///
    // const std::map<std::string, std::string>& wrap_op_names{{"LoopBegin_2871", "LoopEnd_2872"},
    //                                                         {"IntermediateMemoryBuffer_2852", "IntermediateMemoryBuffer_2853"},
    //                                                         {"LoopBegin_2881", "LoopEnd_2882"}};
    return wrap_selected(linear_ir, wrap_op_names);
    // return wrap_whole_body(linear_ir);
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
