// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>
#include <ngraph/op/power.hpp>
#include <snippets/snippets_isa.hpp>

namespace ngraph {
namespace snippets {
namespace op {

// todo: add description
class SerializationNode : public ngraph::op::Op {
public:
    OPENVINO_OP("SerializationNode", "SnippetsOpset");

    SerializationNode() = default;
    SerializationNode(const Output <Node> &arg, const std::shared_ptr<Node>& node)
    : Op({arg}), m_node(node) {
        if (!node)
            throw ngraph_error("SerializationNode requires non-null node pointer");
        std::string type = node->get_type_name();
        std::string name = node->get_friendly_name();
        // If node is a parameter, show another type name, so the node will be displayed correctly
        get_rt_info()["layerType"] = type == "Parameter" ? "ParameterLowered" : type;
        set_friendly_name(name);
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        set_output_type(0, element::f32, {});
    }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector &new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<SerializationNode>(new_args.at(0), m_node);
    }
    bool visit_attributes(AttributeVisitor &visitor) override {
        std::vector<std::pair<std::string, ov::PartialShape>> shapes;
        for (int i = 0; i < m_node->get_input_size(); i++) {
            const auto& pshape =  m_node->get_input_partial_shape(i);
            if (pshape.begin() != pshape.end())
                shapes.emplace_back("in_shape_" + std::to_string(i), m_node->get_input_partial_shape(i));
        }
        for (int i = 0; i < m_node->get_output_size(); i++) {
            const auto& pshape =  m_node->get_output_partial_shape(i);
            if (pshape.begin() != pshape.end())
            shapes.emplace_back("out_shape_" + std::to_string(i), pshape);
        }
        for (auto& s : shapes )
            visitor.on_attribute(s.first, s.second);
        m_node->visit_attributes(visitor);
        return true;
    }

private:
    std::shared_ptr<Node> m_node;
};

} // namespace op
} // namespace snippets
} // namespace ngraph