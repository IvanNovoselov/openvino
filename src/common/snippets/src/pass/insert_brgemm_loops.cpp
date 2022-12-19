// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/pass/insert_brgemm_loops.hpp"
#include "snippets/pass/loop_helpers.hpp"
#include "snippets/op/brgemm.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"
#include "snippets/utils.hpp"

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
        std::vector<size_t> layout_A;
        size_t leading_dim_A;
        std::tie(layout_A, leading_dim_A) = brgemm->get_layout_and_leading_dimension(0);
        const auto& shape_A =  utils::get_reordered_planar_shape(brgemm->get_input_shape(0), layout_A);
        const auto M_rows = shape_A[shape_A.size() - 2].get_length();
        if (M_rows > M_block_size) {
            const auto& loop_begin = op::insertLoopBegin(brgemm->input_values());
            const auto leading_dim_C = brgemm->get_layout_and_leading_dimension(2).second;
            const std::vector<int64_t> ptr_increments {static_cast<int64_t>(M_block_size * leading_dim_A),
                                                       0,
                                                       static_cast<int64_t>(M_block_size * leading_dim_C)};
            const std::vector<int64_t> finalization_offsets(ptr_increments.size(), 0);

            std::vector<Input<Node>> child_inputs;
            for (const auto& in : brgemm->output(0).get_target_inputs())
                child_inputs.push_back(in);
            insertLoopEnd(child_inputs, loop_begin, M_rows, M_block_size,
                          ptr_increments,  finalization_offsets);
            return true;
        }
        brgemm->set_count(M_rows);
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(brgemm_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph