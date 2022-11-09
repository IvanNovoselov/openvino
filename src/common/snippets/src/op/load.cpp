// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/load.hpp"

#include <ngraph/runtime/host_tensor.hpp>

using namespace std;
using namespace ngraph;

snippets::op::Load::Load(const Output<Node>& x, const size_t count) : Op({x}), m_count(count) {
    constructor_validate_and_infer_types();
}

bool snippets::op::Load::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("count", m_count);
    return true;
}

std::shared_ptr<Node> snippets::op::Load::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Load);
    check_new_args_count(this, new_args);
    return std::make_shared<Load>(new_args.at(0), m_count);
}

void snippets::op::Load::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool snippets::op::Load::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    INTERNAL_OP_SCOPE(Load);
    NGRAPH_CHECK(input_values.size() == this->inputs().size(), "wrong input config");
    NGRAPH_CHECK(output_values.size() == this->outputs().size(), "wrong output config");
    NGRAPH_CHECK(input_values.size() == output_values.size() && input_values.size() == 1, "must be 1->1 operation");
    NGRAPH_CHECK(this->output(0).get_shape() == output_values[0]->get_shape(), "output vector must have the same shape as output port");
    NGRAPH_CHECK(this->input(0).get_shape() == input_values[0]->get_shape(), "input and output must have same shape");
    NGRAPH_CHECK(this->input(0).get_shape() == input_values[0]->get_shape(), "input and output must have same shape");

    std::copy(input_values[0]->get_data_ptr<uint8_t>(),
        input_values[0]->get_data_ptr<uint8_t>() + shape_size(get_output_shape(0))*output_values[0]->get_element_type().size(),
        output_values[0]->get_data_ptr<uint8_t>());

    return true;
}

snippets::op::LoadReshape::LoadReshape(const Output<ov::Node>& x, const size_t count, std::vector<size_t> order)
                            : Load(x, count), m_order(std::move(order)) {
    const auto& in_shape = x.get_partial_shape();
    NGRAPH_CHECK(in_shape.is_static(), "LoadReshape supports only static input shapes");
    const auto in_shape_size = in_shape.size();
    NGRAPH_CHECK(m_order.size() == in_shape_size, "LoadReshape got new_order of invalid size");
    NGRAPH_CHECK(*std::max_element(m_order.begin(), m_order.end()) == in_shape_size - 1 &&
                 *std::min_element(m_order.begin(), m_order.end()) == 0, "LoadReshape detected invalid values in new_order");
    const std::set<size_t> unique_dims(order.begin(), order.end());
    NGRAPH_CHECK(unique_dims.size() == order.size(), "LoadReshape order must not contain repeated elements");
    constructor_validate_and_infer_types();
}

void snippets::op::LoadReshape::validate_and_infer_types() {
    const auto& old_shape = get_input_partial_shape(0);
    ov::PartialShape new_shape;
    for (const auto idx : m_order)
        new_shape.push_back(old_shape[idx]);
    set_output_type(0, get_input_element_type(0), new_shape);
}

bool snippets::op::LoadReshape::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("count", m_count);
    visitor.on_attribute("order", m_order);
    return true;
}

std::shared_ptr<Node> snippets::op::LoadReshape::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(LoadReshape);
    check_new_args_count(this, new_args);
    return std::make_shared<LoadReshape>(new_args.at(0), m_count, m_order);
}