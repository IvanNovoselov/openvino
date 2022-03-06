// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ngraph_test_utils.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

class SnippetsCollapseSubgraphTests : public TransformationTestsF {
public:
    virtual void run();
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
