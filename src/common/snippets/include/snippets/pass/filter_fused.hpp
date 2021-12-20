// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/pass.hpp>
#include <snippets/itt.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface FilterFused
 * @brief Mark operations that will be fused on plugin side (but not yet in snippets) so they'll be ignored by snippets.
 * @ingroup snippets
 */
class FilterFused : public ngraph::pass::FunctionPass {
public:
    FilterFused() : FunctionPass() {}
    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};
/*
 FusedWithConvolution, FusedWithConvolutionSumActivation, FusedWithMisc - fusing chain is active and may be continued
 FusedTerminator - the node is fused, but the chain must be interrupted
 Ignored - must be skipped, since can't be handled properly at this time
 Order of SnippetsNodeType is important!:
    * SnippetsNodeType < NotSet is a part of subgraph
    * SnippetsNodeType >= FusedTerminator is a Fused chain
    * SnippetsNodeType > FusedTerminator is a Fused chain that may be continued
 */
enum class SnippetsNodeType : int64_t {SubgraphStart, SubgraphBody,
                                        NotSet, Ignored,
                                        FusedTerminator,
                                        FusedWithConvolution,  FusedWithBinaryConvolution, FusedWithConvolutionSumActivation,
                                        FusedWithMatMul, FusedWithMisc};
void SetSnippetsNodeType(std::shared_ptr<Node> node, SnippetsNodeType);
SnippetsNodeType GetSnippetsNodeType(std::shared_ptr<Node> node);

} // namespace pass
} // namespace snippets
} // namespace ngraph
