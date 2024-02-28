// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/add.hpp"
#include "eltwise.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {
using AutoBroadcastSpec = ov::op::AutoBroadcastSpec;
using AutoBroadcastType = ov::op::AutoBroadcastType;

class VnniTransform : public UnaryEltwiseTPP, public ov::op::Op {
public:
    OPENVINO_OP("VnniTransform", "TppOpset");
    VnniTransform(const Output<Node>& arg, element::Type src_type,
                  size_t offset_in = 0lu, size_t offset_out = 0lu,
                  const std::vector<size_t>& layout_in = {}, size_t blk_size_k = 0, size_t blk_size_n = 0);

    // todo: some of these methods are very similar to BrgemmCopyB.
    //  Should we consider moving shared functionality to a base class in the common part?
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    ov::Shape get_data_repacking_shape(const ov::snippets::VectorDims& planar_dims) const;
    static size_t get_vnni_factor(const ov::element::Type& type) {return 4 / type.size();}

    class ShapeInfer : public snippets::IShapeInferSnippets {
        std::vector<size_t> m_layout{};
    public:
        explicit ShapeInfer(const std::shared_ptr<ov::Node>& n);
        Result infer(const std::vector<snippets::VectorDimsRef>& input_shapes) override;
    };

private:
    void compute_block_size_values(size_t blk_size_k, size_t blk_size_n);
    void validate_element_type(const ov::element::Type& element_type);
    void custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_input);
    libxsmm_meltw_unary_type get_libxsmm_op_type(element::Type src_type, const snippets::VectorDims& planar_shape) const;
    element::Type m_src_type = ov::element::undefined;
    size_t m_K_blk = 0;
    size_t m_N_blk = 0;
    size_t m_VnniFactor = 1;
};


} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
