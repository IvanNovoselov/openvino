// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>
#include "shape_inference/shape_inference_cpu.hpp"

#pragma once
namespace ov {
namespace intel_cpu {
namespace node {
using Result = IShapeInfer::Result;

class SnippetShapeInfer : public ShapeInferEmptyPads {
public:
    explicit SnippetShapeInfer(const std::shared_ptr<snippets::op::Subgraph>& s)
        : m_subgraph(s), m_input_blocked_descs{}, m_output_blocked_descs{} {
        m_status_map[snippets::ShapeInferStatus::success] = ov::intel_cpu::ShapeInferStatus::success;
        m_status_map[snippets::ShapeInferStatus::skip] = ov::intel_cpu::ShapeInferStatus::skip;
    }
    Result infer(
            const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
            const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        OPENVINO_ASSERT(!m_input_blocked_descs.empty() && !m_output_blocked_descs.empty(),
                        "Shape infer can't be performed with uninitialized blocked memory descriptors");

        std::vector<VectorDims> new_input_shapes;
        std::vector<std::reference_wrapper<const VectorDims>> new_input_shapes_ref;
        for (auto i = 0; i < input_shapes.size(); i++) {
            const auto& new_desc = m_input_blocked_descs[i]->cloneWithNewDims(input_shapes[i], true);
            new_input_shapes.emplace_back(new_desc->as<BlockedMemoryDesc>()->getBlockDims());
        }
        for (auto i =0; i < input_shapes.size(); i++)
            new_input_shapes_ref.emplace_back(new_input_shapes[i]);
        auto snippets_result = m_subgraph->shape_infer(new_input_shapes_ref);
        OPENVINO_ASSERT(m_status_map.count(snippets_result.status) != 0, "Failed to map snippets shapeInfer status to the plugin one");


        auto& output_shapes = snippets_result.dims;
        for (auto i = 0; i < output_shapes.size(); i++) {
            const auto& order = m_output_blocked_descs[i]->as<BlockedMemoryDesc>()->getOrder();
            const auto& dims =  m_output_blocked_descs[i]->getShape().getDims();
            if (dims.size() < order.size()) {
                const auto block_idx = order.back();
                auto& out_shape = output_shapes[i];
                out_shape[block_idx] = dims[block_idx];
                out_shape.pop_back();
            }
        }

        return {snippets_result.dims, m_status_map.at(snippets_result.status)};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
    void update_node_config(NodeDesc* nd) {
        OPENVINO_ASSERT(nd, "Valid node descriptor required to initialize i/o memory descriptors");
        const auto config = nd->getConfig();
        OPENVINO_ASSERT(m_subgraph->get_input_size() == config.inConfs.size(), "Incompatible subgraph's input number and config");
        OPENVINO_ASSERT(m_subgraph->get_output_size() == config.outConfs.size(), "Incompatible subgraph's output number and config");
        for (const auto& port_config : config.inConfs)
            m_input_blocked_descs.push_back(port_config.getMemDesc()->clone());
        for (const auto& port_config : config.outConfs)
            m_output_blocked_descs.push_back(port_config.getMemDesc()->clone());
    }

private:
    std::shared_ptr<snippets::op::Subgraph> m_subgraph;
    std::vector<MemoryDescPtr> m_input_blocked_descs;
    std::vector<MemoryDescPtr> m_output_blocked_descs;
    std::map<snippets::ShapeInferStatus, ov::intel_cpu::ShapeInferStatus> m_status_map;

    VectorDims reshape_blocked(const VectorDims& planar, const VectorDims& order, const VectorDims& blockedDims) {
        VectorDims newBlockedDims(order.size());

        for (size_t i = 0; i < planar.size(); ++i) {
            newBlockedDims[i] = planar[order[i]];
        }

        for (size_t i = planar.size(); i < order.size(); ++i) {
            if (newBlockedDims[order[i]] != Shape::UNDEFINED_DIM) {
                newBlockedDims[order[i]] = div_up(newBlockedDims[order[i]], blockedDims[i]);
                newBlockedDims[i] = blockedDims[i];
            }
        }
    }

    VectorDims blocked_to_planar(const VectorDims& planar, const VectorDims& order, const VectorDims& blockedDims) {
        VectorDims newBlockedDims(order.size());

        for (size_t i = 0; i < planar.size(); ++i) {
            newBlockedDims[i] = planar[order[i]];
        }

        for (size_t i = planar.size(); i < order.size(); ++i) {
            if (newBlockedDims[order[i]] != Shape::UNDEFINED_DIM) {
                newBlockedDims[order[i]] = div_up(newBlockedDims[order[i]], blockedDims[i]);
                newBlockedDims[i] = blockedDims[i];
            }
        }
    }
};

class SnippetShapeInferFactory : public ShapeInferFactory {
public:
    explicit SnippetShapeInferFactory(const std::shared_ptr<ov::Node>& op) {
        m_subgraph = ov::as_type_ptr<snippets::op::Subgraph>(op);
        OPENVINO_ASSERT(m_subgraph, "Invalid node type detected in SnippetShapeInferFactory");
    }
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<SnippetShapeInfer>(m_subgraph);
    }

private:
    std::shared_ptr<snippets::op::Subgraph> m_subgraph = nullptr;
};
} // namespace node
} // namespace intel_cpu
} // namespace ov

