// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/enumerate_nodes.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
void SetTopologicalOrder(std::shared_ptr<Node> node, int64_t order) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::SetTopologicalOrder")
    auto &rt = node->get_rt_info();
    rt["TopologicalOrder"] = order;
}

int64_t GetTopologicalOrder(std::shared_ptr<Node> node) {
    auto &rt = node->get_rt_info();
    const auto rinfo = rt.find("TopologicalOrder");
    if (rinfo == rt.end())
        throw ngraph_error("Topological order is required, but not set.");
    return rinfo->second.as<int64_t>();
}
bool EnumerateNodes::run_on_function(std::shared_ptr<Function> f) {
    RUN_ON_FUNCTION_SCOPE(FulterFused);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::EnumerateNodes")
    int64_t order = 0;
    // Todo: We don't really have to set order for every node, just for subgraph parents and children would be enough
    for (auto &node : f->get_ordered_ops()) {
        SetTopologicalOrder(node, order++);
    }
    return true;
}
} // namespace pass
} // namespace snippets
} // namespace ngraph
