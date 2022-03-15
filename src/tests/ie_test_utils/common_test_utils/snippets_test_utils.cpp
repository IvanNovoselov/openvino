// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets_test_utils.hpp"
#include "snippets/op/subgraph.hpp"
#include "ngraph/pattern/op/label.hpp"

namespace ov {
namespace test {
namespace snippets {

SnippetsRestoreResultInputName::SnippetsRestoreResultInputName() {
    auto label =
            std::make_shared<ngraph::pattern::op::Label>(ngraph::pattern::any_input(),
             [](const std::shared_ptr<const Node> &n) {
                 return is_type<op::v0::Result>(n) &&
                        is_type<ngraph::snippets::op::Subgraph>(n->get_input_source_output(0).get_node_shared_ptr());
             });
    ov::graph_rewrite_callback callback = [](ngraph::pattern::Matcher &m) -> bool {
        const auto& node = m.get_match_root();
        const auto& subgraph = as_type_ptr<ngraph::snippets::op::Subgraph>(node->get_input_source_output(0).get_node_shared_ptr());
        bool not_set = true;
        for (unsigned int i = 0; i < subgraph->get_output_size() && not_set; i++) {
            for (const auto &in : subgraph->get_output_target_inputs(i)) {
                if (ov::is_type<op::v0::Result>(in.get_node())) {
                    const auto& body_result = subgraph->get_body()->get_output_op(i);
                    const auto& body_result_input = body_result->get_input_source_output(0);
                    subgraph->set_friendly_name(body_result_input.get_node_shared_ptr()->get_friendly_name());
                    not_set = false;
                    break;
                }
            }
        }
        return true;
    };
    auto matcher = std::make_shared<ngraph::pattern::Matcher>(label);
    register_matcher(matcher, callback);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov