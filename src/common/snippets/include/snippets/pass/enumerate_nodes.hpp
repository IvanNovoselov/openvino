// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/pass.hpp>
#include <snippets/itt.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface EnumerateNodes
 * @brief  Snippets rely on topological order to avoid creating cyclic dependencies, so this transformation enumerates nodes in topological order.
 * @ingroup snippets
 */
class TRANSFORMATIONS_API EnumerateNodes : public ngraph::pass::FunctionPass {
public:
    EnumerateNodes() : FunctionPass() {}
    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};

void SetTopologicalOrder(std::shared_ptr<Node>, int64_t);
int64_t GetTopologicalOrder(std::shared_ptr<Node>);

} // namespace pass
} // namespace snippets
} // namespace ngraph
