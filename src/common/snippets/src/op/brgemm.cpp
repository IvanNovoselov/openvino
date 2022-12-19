// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/op/brgemm.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/core/rt_info.hpp"
#include "snippets/utils.hpp"
#include "matmul_shape_inference.hpp"

namespace ngraph {
namespace snippets {
namespace op {

namespace {
std::pair<std::vector<size_t>, size_t> get_node_layout_and_leading_dimension(const Output<Node>& in) {
    auto in_node = in.get_node_shared_ptr();
    // If input is LoopBegin then it has multiple outputs and doesn't store output layout,
    // so we have to check the original input node rt_info
    if (ov::is_type<snippets::op::LoopBegin>(in_node)) {
        in_node = in_node->get_input_node_shared_ptr(in.get_index());;
    }
    auto layout = ngraph::snippets::utils::get_node_output_layout(in_node);
    size_t leading_dimension;
    const auto& io_shape = in.get_shape();
    if (layout.empty()) {
        // empty value indicates a planar layout
        leading_dimension = io_shape.back();
        layout.resize(io_shape.size());
        std::iota(layout.begin(), layout.end(), 0);
    } else {
        // The idea here is to find "2" (for 4D shapes) in the layout and multiply dimensions that are to the right
        // This implies that "3" is the last layout value, otherwise this layout is not supported.
        // counting from the end since shape could be prepended with ones
        const int64_t num_last_dims = layout.end() - std::find(layout.begin(), layout.end(), layout.size() - 2) - 1;
        if (layout.back() != layout.size() - 1 || num_last_dims < 1)
            throw ngraph_error("Brgemm detected unschedulable shape + layout combination");
        leading_dimension = std::accumulate(io_shape.end() - num_last_dims, io_shape.end(), 1, std::multiplies<size_t>());
    }
    return {layout, leading_dimension};
}
} //namespace

Brgemm::Brgemm(const Output<Node>& A, const Output<Node>& B, const size_t M_block_size, const size_t count)
    : MatMul(), m_optimal_M_block_size(M_block_size), m_count(count) {
    set_arguments({A, B});
    set_output_size(1);
    constructor_validate_and_infer_types();
}

Brgemm::Brgemm(const Output<Node>& A, const Output<Node>& B, const size_t M_block_size, const size_t count, const std::vector<size_t>& output_layout)
    : MatMul(), m_optimal_M_block_size(M_block_size), m_count(count) {
    set_arguments({A, B});
    set_output_size(1);
    get_rt_info()["Layout"] = output_layout;
    constructor_validate_and_infer_types();
}

size_t Brgemm::get_M_block_size() const {
    return m_optimal_M_block_size;
}

size_t Brgemm::get_count() const {
        return m_count;
}

void Brgemm::set_count(const size_t count) {
        m_count = count;
}

void Brgemm::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Brgemm_validate_and_infer_types);
    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)),
                          "Arguments do not have the same element type (arg0 element type: ",
                          get_input_element_type(0),
                          ", arg1 element type: ",
                          get_input_element_type(1),
                          ").");
    // If no leading dimensions are provided, assume dense row-major inputs-outputs
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static(),
                          "Brgemm currently supports only static shapes.");
    // Two rightmost dimensions processed by jit kernel, and what's left is used for scheduling
    NODE_VALIDATION_CHECK(this, get_input_shape(0).size() > 2 && get_input_shape(1).size() > 2,
                          "Brgemm supports only inputs with rank 2 or higher.");

    NODE_VALIDATION_CHECK(this, m_count <= m_optimal_M_block_size,
                          "Brgemm count must be <= than optimal_M_block_size. Insert Loops if you need to process more elements.");
    std::vector<ov::PartialShape> planar_input_shapes;
    for (const auto& in : input_values()) {
        ov::PartialShape planar_shape;
        auto in_node = in.get_node_shared_ptr();
        // If input is LoopBegin then it has multiple outputs and doesn't store output layout,
        // so we have to check the original input node rt_info
        if (ov::is_type<snippets::op::LoopBegin>(in_node)) {
            in_node = in_node->get_input_node_shared_ptr(in.get_index());;
        }
        const auto& layout = utils::get_node_output_layout(in_node);
        planar_input_shapes.push_back(utils::get_reordered_planar_shape(in.get_partial_shape(), layout));
    }

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    ov::op::v0::shape_infer(this, planar_input_shapes, output_shapes);
    const auto& output_layout = utils::get_node_output_layout(this);
    output_shapes[0] = utils::get_reordered_planar_shape(output_shapes[0], output_layout);
    set_output_type(0, result_et, output_shapes[0]);
}

bool Brgemm::visit_attributes(AttributeVisitor& visitor) {
    MatMul::visit_attributes(visitor);
    visitor.on_attribute("count", m_count);
    return true;
}

std::shared_ptr<Node> Brgemm::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Brgemm_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto layout = ngraph::snippets::utils::get_node_output_layout(this);
    const auto& A = new_args.at(0);
    const auto& B = new_args.at(1);
    return layout.empty() ?
           std::make_shared<Brgemm>(A, B, m_optimal_M_block_size, m_count) :
           std::shared_ptr<Brgemm>(new Brgemm(A, B, m_optimal_M_block_size, m_count, layout));
}

std::pair<std::vector<size_t>, size_t> Brgemm::get_layout_and_leading_dimension(const int index) {
    switch (index) {
        case 0 : return get_node_layout_and_leading_dimension(input_value(0)); break;
        case 1 : return get_node_layout_and_leading_dimension(input_value(1)); break;
        case 2 : return get_node_layout_and_leading_dimension(output(0)); break;
        default : throw ngraph_error("Unsupported index in get_layout_and_leading_dimension");
    }
}

} // namespace op
} // namespace snippets
} // namespace ngraph
