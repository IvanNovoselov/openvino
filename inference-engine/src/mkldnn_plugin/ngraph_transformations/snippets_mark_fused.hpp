// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <snippets/itt.hpp>

namespace MKLDNNPlugin {
/**
 * @interface SnippetsMarkFused
 * @brief Mark operations that will be fused on plugin side (but not yet in snippets) so they'll be ignored by snippets.
 */
class SnippetsMarkFused : public ngraph::pass::FunctionPass {

public:
    NGRAPH_RTTI_DECLARATION;
    SnippetsMarkFused() : FunctionPass() {}
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
enum class NodeFusingType : int64_t {SubgraphStart, SubgraphBody,
    NotSet, Ignored,
    FusedTerminator,
    FusedWithConvolution,  FusedWithBinaryConvolution, FusedWithConvolutionSumActivation,
    FusedWithMatMul, FusedWithMisc};
void SetNodeFusingType(std::shared_ptr<ov::Node> node, NodeFusingType);
NodeFusingType GetNodeFusingType(std::shared_ptr<ov::Node> node);
}  // namespace MKLDNNPlugin