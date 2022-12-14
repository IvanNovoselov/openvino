// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/matmul.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Brgemm
 * @brief Brgemm is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 * @ingroup snippets
 */
class Brgemm : public ngraph::op::v0::MatMul {
public:
    OPENVINO_OP("Brgemm", "SnippetsOpset", ngraph::op::v0::MatMul);
    Brgemm(const Output<Node>& A, const Output<Node>& B, size_t M_block_size = 32, size_t count = 32);
    Brgemm() = default;

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    size_t get_M_block_size() const;
    bool visit_attributes(AttributeVisitor& visitor) override;

    bool has_evaluate() const override { return false; }
    // Brgemm incorporates memory i/o semantics, so we introduce MemoryAccess-like methods
    size_t get_count() const;
    void set_count(size_t count);
    /**
    * @interface get_layout_and_leading_dimension
    * @brief Returns a <layout, leading_dimension> tuple.
     * @param index - i/o por number: 0 - 0th input, 1 - 1st input, 2 - 0th output.
    * @ingroup snippets
    */
    std::pair<std::vector<size_t>, size_t> get_layout_and_leading_dimension(int index);

private:
    size_t m_optimal_M_block_size = 0;
    size_t m_count = 0;
};

} // namespace op
} // namespace snippets
} // namespace ngraph