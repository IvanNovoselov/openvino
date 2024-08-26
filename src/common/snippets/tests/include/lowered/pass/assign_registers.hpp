// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

struct AssignRegistersTest : public LoweringTests  {
    void SetUp() override;
    std::shared_ptr<ov::snippets::op::Subgraph> m_subgraph;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
