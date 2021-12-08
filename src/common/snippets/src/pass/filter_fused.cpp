// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/filter_fused.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

namespace {
bool hasSkippedParent(std::shared_ptr<Node> node) {
    for (const auto& input : node->inputs()) {
        const auto parent = input.get_source_output().get_node_shared_ptr();
        if (GetSnippetsNodeType(parent) == SnippetsNodeType::SkippedBySnippets)
            return true;
    }
    return false;
}
bool hasParameterParent(std::shared_ptr<Node> node) {
    for (const auto& input : node->inputs()) {
        const auto parent = input.get_source_output().get_node_shared_ptr();
        if (ov::is_type<ngraph::op::Parameter>(parent))
            return true;
    }
    return false;
}
bool hasParentInStartedSubgraph(std::shared_ptr<Node> node) {
    auto inputs = node->inputs();
    for (const auto& input : inputs) {
        const auto parent = input.get_source_output().get_node_shared_ptr();
        // True for SubgraphStart and SubgraphBody by convention
        if (GetSnippetsNodeType(parent) < SnippetsNodeType::NotSet)
            return true;
    }
    return false;
}

} // namespace

bool SkipInputChainsAndSetOrder::run_on_function(std::shared_ptr<Function> f) {
    RUN_ON_FUNCTION_SCOPE(FulterFused);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::SkipInputChainsAndSetOrder")
    auto ordered_ops = f->get_ordered_ops();
    // todo: It is probably better to move this check to a separate MatcherPass
    // Check that plugin set only allowed flags. See collapse_subgraph.hpp for details.
    const bool pluginSetAllowedFlags = std::all_of(ordered_ops.begin(), ordered_ops.end(),
                [](std::shared_ptr<Node> &node) {
                        return GetSnippetsNodeType(node) == SnippetsNodeType::NotSet ||
                                GetSnippetsNodeType(node) == SnippetsNodeType::SkippedByPlugin;
                });
    if (!pluginSetAllowedFlags)
        ngraph_error("Plugin has set invalid SnippetsNodeType, please check the corresponding markup transformation.");
    for (size_t order = 0; order < ordered_ops.size(); order++) {
        auto &node = ordered_ops[order];
        if (ngraph::op::is_constant(node) || ngraph::op::is_parameter(node))
            continue;
        SetTopologicalOrder(node, order);
        if ((hasParameterParent(node) || hasSkippedParent(node)) && AppropriateForSubgraph(node)) {
            SetSnippetsNodeType(node, SnippetsNodeType::SkippedBySnippets);
        }
    }
    std::cerr << "FilterFused passed" << std::endl;
    return true;
}
} // namespace pass
} // namespace snippets
} // namespace ngraph
