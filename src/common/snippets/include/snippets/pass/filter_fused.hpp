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
 * @interface SkipInputChainsAndSetOrder
 * @brief  Snippets currently support only FP32 precision, but input precisions could potentially be converted to
 * U8 after tokenization. So this transformation mark eltwise chains starting at input Parameters as SnippetsNodeType::SkippedBySnippets.
 * Snippets rely on topological order to avoid creating cyclic dependencies. This transformation also sets topological order.
 * @ingroup snippets
 */
// Todo: We don't really have to set order for every node, just for subgraph parents and children would be enough
// Todo: Note that thera
class TRANSFORMATIONS_API SkipInputChainsAndSetOrder : public ngraph::pass::FunctionPass {
public:
    SkipInputChainsAndSetOrder() : FunctionPass() {}
    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
