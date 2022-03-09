// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <ngraph/pass/pass.hpp>
#include <openvino/pass/graph_rewrite.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

// todo: This transformation is required to pass ngraph::pass::CheckUniqueNames executed from TransformationTestsF
//  Note that other solutions are also possible:
//  - Modify TransformationTestsF to make it possible to disable CheckUniqueNames on demand.
//      Not favorable because several other checks inside CheckUniqueNames are quite useful (solution: modify CheckUniqueNames also?).
//  - Write our own class similar to TransformationTestsF, but a bit more flexible.
//      Not favorable because of code duplication.
//  - Slightly modify TokenizeSnippets subgraph naming policy. It currently tries to mimic the one used by MKLDNNPlugin
//      (node name before result may be changed, but tensor name must be preserved), but it's easy to update Subgraph name
//      (on demand) if it goes before the Result.
//      Not favorable because it seems that this alternative-naming-behavior is needed only to pass these tests.
//  - An alternative solution is to insert dummy non-tokenizable node before the Result (Such as Convert)
//      Not favorable because we won't be able to test that the tensor names are preserved for the Result inputs.
class SnippetsRestoreResultInputName : public ov::pass::MatcherPass {
public:
    SnippetsRestoreResultInputName();
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph