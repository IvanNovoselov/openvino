// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface InsertBrgemmLoops
 * @brief Insert explicit Loop operations around Brgemm op, so it processes only a part of first input tensor in one call.
 * @ingroup snippets
 */
class InsertBrgemmLoops: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertBrgemmLoops", "0");
    InsertBrgemmLoops();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
