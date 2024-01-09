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

bool InsertPerfCount::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertPerfCount")
    if (linear_ir.empty())
        return false;

    auto is_parameter = [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<ov::op::v0::Parameter>(node);
    };
    auto is_result = [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<ov::op::v0::Result>(node);
    };

    // mark perf_count_begin and perf_count_end position
    // auto perf_count_begin_pos = linear_ir.cbegin();
    // auto perf_count_end_pos = perf_count_begin_pos;
    bool first_result_marked = false;
    // std::vector<std::pair<decltype(linear_ir.cbegin()), decltype(linear_ir.cbegin())>> pair_positions;
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
            perf_count_begin->set_friendly_name(std::string("PerfCount_End_") + std::to_string(seq_number));
            const auto& perf_count_end_expr = linear_ir.create_expression(perf_count_end, empty_connectors);
            expr_it = linear_ir.insert(perf_count_end_pos, perf_count_end_expr);
            seq_number++;
        }
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
