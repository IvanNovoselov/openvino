// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

//todo: update the description
/**
 * @interface InsertBrgemmLoops
 * @brief Insert explicit Loop operations around Brgemm op to account for the fa into the body to process multiple data entities during one kernel execution
 * @param master_shape - shape used to determine loop work amounts
 * @param loop_depth - the number of last master_shape dimensions processed by loops (aka tileRank - obsolete), could be 1 or 2
 * @param vector_size - the number of entities processed on one iteration of vector loop
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
