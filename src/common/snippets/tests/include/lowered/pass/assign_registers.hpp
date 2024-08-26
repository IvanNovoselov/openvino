// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lir_test_utils.hpp"
#include "snippets_helpers.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace test {
namespace snippets {

class AssignRegistersTest : public LoweredPassTestsF  {
public:
    AssignRegistersTest();
    void SetUp() override;
    void set_reg_types(ov::snippets::lowered::LinearIR& linear_ir) const;
    std::shared_ptr<ov::snippets::op::Subgraph> subgraph;
    std::shared_ptr<ov::snippets::Generator> generator;
};
/***
 * Reference implementation of AssignRegisters pass
 */
class AssignRegistersRef : public ov::snippets::lowered::pass::Pass {
public:
    OPENVINO_RTTI("AssignRegistersRef", "Pass")

    explicit AssignRegistersRef(const std::function<ov::snippets::RegType(const ov::Output<Node>& out)>& mapper,
                                const size_t reg_cnt)
            : m_reg_type_mapper(mapper), reg_count(reg_cnt) {}

    bool run(ov::snippets::lowered::LinearIR& linear_ir) override;

private:
    void set_reg_types(ov::snippets::lowered::LinearIR& linear_ir);

    std::function<ov::snippets::RegType(const ov::Output<Node>& out)> m_reg_type_mapper;
    size_t reg_count;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
