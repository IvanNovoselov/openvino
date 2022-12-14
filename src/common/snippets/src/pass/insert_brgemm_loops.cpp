// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/pass/insert_brgemm_loops.hpp"
#include "snippets/pass/loop_helpers.hpp"
#include "snippets/op/brgemm.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"

#include <ngraph/rt_info.hpp>
namespace ngraph {
namespace snippets {
namespace pass {


InsertBrgemmLoops::InsertBrgemmLoops() {
    MATCHER_SCOPE(InsertBrgemmLoops);
    auto brgemm_pattern = pattern::wrap_type<snippets::op::Brgemm>();

    auto callback = [=](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::InsertBrgemmLoops")
        const auto& brgemm = as_type_ptr<snippets::op::Brgemm>(m.get_match_root());
        const auto M_block_size = brgemm->get_M_block_size();
        // Brgemm checks that input shapes are static and with ranks > 2
        const auto& A_shape =  brgemm->get_input_shape(0);
        const auto num_M_rows = A_shape[A_shape.size() - 2];
        if (num_M_rows > M_block_size) {
            const auto& loop_begin = op::insertLoopBegin(brgemm->input_values());
//            OutputVector child_inputs;
//            for (const auto& in : brgemm->output(0).get_target_inputs())
//                child_inputs.push_back(in.get_source_output());

            const auto LDA = brgemm->get_layout_and_leading_dimension(0).second;
            const auto LDC = brgemm->get_layout_and_leading_dimension(2).second;

            const std::vector<int64_t> ptr_increments {static_cast<int64_t>(M_block_size * LDA * brgemm->get_input_element_type(0).size()),
                                                       0,
                                                       static_cast<int64_t>(M_block_size * LDC * brgemm->get_output_element_type(0).size())};
            const std::vector<int64_t> finalization_offsets(ptr_increments.size(), 0);

//            OutputVector loop_end_inputs(brgemm->outputs());
//            loop_end_inputs.push_back(loop_begin->output(loop_begin->get_output_size() - 1));
//            auto loop_end = std::make_shared<op::LoopEnd>(loop_end_inputs, num_M_rows, M_block_size,
//                                                                              ptr_increments, finalization_offsets);


            std::vector<Input<Node>> child_inputs;
            for (const auto& in : brgemm->output(0).get_target_inputs())
                child_inputs.push_back(in);
            const auto& inner_loop_end = insertLoopEnd(child_inputs, loop_begin, num_M_rows, M_block_size,
                                                       ptr_increments,  finalization_offsets);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(brgemm_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph