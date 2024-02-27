// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vnni_transform.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

VnniTransform::VnniTransform(const Output<Node>& arg, const element::Type src_type,
                             const size_t offset_in, const size_t offset_out,
                             std::vector<size_t> layout_in, const size_t blk_size_k, const size_t blk_size_n) :
                             UnaryEltwiseTPP(get_libxsmm_op_type(src_type)),
                             Op({arg}), m_src_type(src_type) {
    m_input_ports[0].offset = offset_in;
    m_output_ports[0].offset = offset_out;
    compute_block_size_values(blk_size_k, blk_size_n);
    custom_constructor_validate_and_infer_types(std::move(layout_in));
}

libxsmm_meltw_unary_type VnniTransform::get_libxsmm_op_type(element::Type src_type) {
    switch (src_type) {
        case element::bf16:
            return LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2;
        case element::i8:
        case element::u8:
            return LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4;
        default:
            OPENVINO_THROW("Unsupported src precision for VnniTransform: " + src_type.to_string());
    }
}

void VnniTransform::compute_block_size_values(const size_t blk_size_k, const size_t blk_size_n) {
    const auto& input_shape = snippets::utils::get_planar_pshape(input(0)).get_shape();
    OPENVINO_ASSERT(input_shape.size() > 1, "1D inputs are not supported by VnniTransform");
    m_K_blk = blk_size_k != 0 ? blk_size_k : *(input_shape.rbegin() + 1);
    m_N_blk = blk_size_n != 0 ? blk_size_n : *input_shape.rbegin();
}

ov::Shape VnniTransform::get_data_repacking_shape(const ov::snippets::VectorDims& planar_dims) const {
    const auto& N = *planar_dims.rbegin();
    const auto& K = *(planar_dims.rbegin() + 1);
    return ov::Shape{rnd_up(K, m_brgemmVNNIFactor), rnd_up(N, m_N_blk)};
}

std::shared_ptr<Node> VnniTransform::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    const auto& clone = std::make_shared<VnniTransform>(*this);
    clone->set_input_port_descriptor(get_input_port_descriptor(0), 0);
    clone->set_output_port_descriptor(get_output_port_descriptor(0), 0);
    clone->custom_constructor_validate_and_infer_types(snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout());
    return clone;
}

void VnniTransform::validate_and_infer_types() {
    const auto& element_type = get_input_element_type(0);
    validate_element_type(element_type);

    const auto port = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0));
    const auto shape = ov::Shape(port->get_shape());
    const auto& planar_pshape = snippets::utils::get_planar_pshape(shape, port->get_layout());
    set_output_type(0, element_type, planar_pshape);
}

void VnniTransform::validate_element_type(const ov::element::Type& element_type) {
    OPENVINO_ASSERT(one_of(element_type, element::bf16, element::i8, element::u8),
                    "VnniTransform doesn't support element type" + element_type.get_type_name());
}

void VnniTransform::custom_constructor_validate_and_infer_types(std::vector<size_t> layout_input) {
    // During ctor call, BrgemmCopyB doesn't know his port descriptors.
    // So we use port descs from source inputs
    const auto element_type = get_input_element_type(0);
    validate_element_type(element_type);
    const auto planar_pshape = snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout_input);
    set_output_type(0, element_type, planar_pshape);
}

bool VnniTransform::visit_attributes(AttributeVisitor& visitor) {
    std::string modifier{"TPP"};
    visitor.on_attribute("modifier", modifier);
    return true;
}

VnniTransform::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    OPENVINO_ASSERT(ov::is_type<VnniTransform>(n), "Got invalid node in VnniTransform::ShapeInfer");
    m_layout = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(n->input(0))->get_layout();
}

snippets::IShapeInferSnippets::Result VnniTransform::ShapeInfer::infer(const std::vector<snippets::VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Got unexpected number of input shapes");
    const auto& planar_shape = snippets::utils::get_planar_vdims(input_shapes[0].get(), m_layout);
    return {std::vector<VectorDims>{planar_shape}, snippets::ShapeInferStatus::success};
}


} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
