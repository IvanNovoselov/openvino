// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets_helpers.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using ngraph::snippets::op::Subgraph;

SnippetsRestoreResultInputName::SnippetsRestoreResultInputName() {
    auto label =
            std::make_shared<ngraph::pattern::op::Label>(ngraph::pattern::any_input(),
                                                         [](const std::shared_ptr<const Node> &n) {
                                                             return is_type<op::v0::Result>(n) &&
                                                                    is_type<Subgraph>(n->get_input_source_output(0).get_node_shared_ptr());
                                                         });
    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher &m) -> bool {
        const auto& node = m.get_match_root();
        const auto& subgraph = as_type_ptr<Subgraph>(node->get_input_source_output(0).get_node_shared_ptr());
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

void SnippetsFunctionBase::validate_function(const std::shared_ptr<Model> &f) const {
    NGRAPH_CHECK(f != nullptr, "The test requires Model to be defined");
    const auto &params = f->get_parameters();
    NGRAPH_CHECK(params.size() == input_shapes.size(),
                 "Passed input shapes and produced function are inconsistent.");
    for (size_t i = 0; i < input_shapes.size(); i++)
        NGRAPH_CHECK(std::equal(input_shapes[i].begin(), input_shapes[i].end(), params[i]->get_shape().begin()),
                     "Passed input shapes and produced function are inconsistent.");
}

SnippetsFunctionCustomizable::SnippetsFunctionCustomizable(std::vector<Shape>& inputShapes,
                                                           std::vector<std::shared_ptr<Node>>& customOps,
                                                           std::vector<size_t>&& customOpsNumInputs)
        : SnippetsFunctionBase(inputShapes), custom_ops{customOps} {
    custom_ops_num_inputs = std::move(customOpsNumInputs);
    NGRAPH_CHECK(custom_ops_num_inputs.size() == custom_ops.size(), "Got inconsistent numbers of custom ops and custom ops inputs");
    // We need to set dummy inputs to increase input arguments count,
    // so clone_with_new_inputs() could pass without errors inside initOriginal() and initReference().
    ResetCustomOpsInputs();
}

void SnippetsFunctionCustomizable::ResetCustomOpsInputs() {
    auto dummy_input = std::make_shared<ov::op::v0::Parameter>(precision, Shape{});
    for (size_t i = 0; i < custom_ops.size(); i++) {
        const NodeVector inputs(custom_ops_num_inputs[i], dummy_input);
        custom_ops[i]->set_arguments(inputs);
    }
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph