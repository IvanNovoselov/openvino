// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph.hpp"

namespace ov {
namespace intel_cpu {
namespace node {
constexpr size_t SnippetShapeInfer::NO_BLOCKING;

SnippetShapeInfer::SnippetShapeInfer(const std::shared_ptr<snippets::op::Subgraph>& s)
    : m_subgraph(s), m_input_is_blocked{}, m_output_is_blocked{} {
    m_status_map[snippets::ShapeInferStatus::success] = ov::intel_cpu::ShapeInferStatus::success;
    m_status_map[snippets::ShapeInferStatus::skip] = ov::intel_cpu::ShapeInferStatus::skip;
}
IShapeInfer::Result
SnippetShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                         const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    OPENVINO_ASSERT(m_input_is_blocked.size() == input_shapes.size() && !m_input_is_blocked.empty(),
                    "Shape infer can't be performed with uninitialized blocked memory descriptors");
    std::vector<std::reference_wrapper<const VectorDims>> unblocked_input_shapes;
    // temporary container to keep references to unblocked shapes alive
    std::vector<VectorDims> tmp_input_shapes;
    if (m_any_is_blocked) {
        tmp_input_shapes.resize(input_shapes.size());
        unblocked_input_shapes = input_shapes;
        for (size_t i = 0; i < m_input_is_blocked.size(); i++) {
            if (m_input_is_blocked[i]) {
                VectorDims shape = input_shapes[i];
                shape[m_blocked_dim_idx] = div_up(shape[m_blocked_dim_idx], m_block_size);
                shape.push_back(m_block_size);
                tmp_input_shapes[i] = shape;
                unblocked_input_shapes[i] = tmp_input_shapes[i];
            }
        }
    }
    auto snippets_result = m_subgraph->shape_infer(m_any_is_blocked ? unblocked_input_shapes : input_shapes);
    OPENVINO_ASSERT(m_status_map.count(snippets_result.status) != 0, "Failed to map snippets shapeInfer status to the plugin one");

    if (m_any_is_blocked) {
        auto& output_shapes = snippets_result.dims;
        for (size_t i = 0; i < m_output_is_blocked.size(); i++) {
            if (m_output_is_blocked[i]) {
                auto& shape = output_shapes[i];
                shape[m_blocked_dim_idx] = m_output_blocked_dim_size[i];
                shape.pop_back();
            }
        }
    }

    return {snippets_result.dims, m_status_map.at(snippets_result.status)};
}

IShapeInfer::port_mask_t SnippetShapeInfer::get_port_mask() const {
    return EMPTY_PORT_MASK;
}

void SnippetShapeInfer::update_node_config(NodeDesc* nd) {
    OPENVINO_ASSERT(nd, "Valid node descriptor required to initialize i/o memory descriptors");
    const auto config = nd->getConfig();
    OPENVINO_ASSERT(m_subgraph->get_input_size() == config.inConfs.size(), "Incompatible subgraph's input number and config");
    OPENVINO_ASSERT(m_subgraph->get_output_size() == config.outConfs.size(), "Incompatible subgraph's output number and config");
    init_blocked_params(m_input_is_blocked, config.inConfs);
    m_output_blocked_dim_size = init_blocked_params(m_output_is_blocked, config.outConfs);
}

VectorDims SnippetShapeInfer::init_blocked_params(std::vector<bool>& is_blocked, const std::vector<PortConfig>& configs) {
    is_blocked.resize(configs.size(), false);
    VectorDims blocked_dim_size(configs.size(), NO_BLOCKING);
    for (size_t i = 0; i < configs.size(); i++) {
        const auto& blocked_desc = configs[i].getMemDesc()->as<BlockedMemoryDesc>();
        const auto& order = blocked_desc->getOrder();
        const auto& dims = blocked_desc->getShape().getDims();
        const auto& blocked_dims = blocked_desc->getBlockDims();
        if (order.size() > dims.size()) {
            // If shape is blocked than there must be only one extra dim in order
            OPENVINO_ASSERT(dims.size() + 1 == order.size(), "Invalid dims and order combination in Snippets shape infer");
            // Subgraph presently supports only NCHW8c or NCHW16c, so it's always channels that must be blocked,
            // but this algorithm is a little more generic, so we allow blocking over any dim
            OPENVINO_ASSERT(std::count(order.begin(), order.end() - 1, order.back()) == 1,
                            "Blocking is supported with respect to the last dimension only");

            if (m_blocked_dim_idx == NO_BLOCKING)
                m_blocked_dim_idx = order.back();
            else if (m_blocked_dim_idx != order.back())
                OPENVINO_THROW("All blocked dims must have the same blocked dim idx");

            if (m_block_size == NO_BLOCKING)
                m_block_size = blocked_dims.back();
            else if (m_block_size != blocked_dims.back())
                OPENVINO_THROW("All blocked dims must have the same block size");
            is_blocked[i] = true;
            m_any_is_blocked = true;
            blocked_dim_size[i] = dims[m_blocked_dim_idx];
        }
    }
    return blocked_dim_size;
}

SnippetShapeInferFactory::SnippetShapeInferFactory(const std::shared_ptr<ov::Node>& op) {
    m_subgraph = ov::as_type_ptr<snippets::op::Subgraph>(op);
    OPENVINO_ASSERT(m_subgraph, "Invalid node type detected in SnippetShapeInferFactory");
}

ShapeInferPtr SnippetShapeInferFactory::makeShapeInfer() const {
    return std::make_shared<SnippetShapeInfer>(m_subgraph);
}

} // namespace node
} // namespace intel_cpu
} // namespace ov

