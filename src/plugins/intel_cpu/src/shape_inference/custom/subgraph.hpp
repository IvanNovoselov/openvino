// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>
#include "shape_inference/shape_inference_cpu.hpp"
#include "snippets/op/subgraph.hpp"

#pragma once
namespace ov {
namespace intel_cpu {
namespace node {

class SnippetShapeInfer : public ShapeInferEmptyPads {
public:
    explicit SnippetShapeInfer(const std::shared_ptr<snippets::op::Subgraph>& s);
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    port_mask_t get_port_mask() const override;
    void update_node_config(NodeDesc* nd);

private:
    static constexpr size_t NO_BLOCKING = SIZE_MAX;
    std::shared_ptr<snippets::op::Subgraph> m_subgraph;
    bool m_any_is_reordered = false;
    std::vector<VectorDims> m_input_order;
    std::vector<VectorDims> m_output_order;

    size_t m_block_size = NO_BLOCKING;
    size_t m_blocked_dim_idx = NO_BLOCKING;
    VectorDims m_input_blocked_dim_size;
    VectorDims m_output_blocked_dim_size;
    std::map<snippets::ShapeInferStatus, ov::intel_cpu::ShapeInferStatus> m_status_map;

    VectorDims init_blocked_params(std::vector<VectorDims>& is_blocked, const std::vector<PortConfig>& configs);
};

class SnippetShapeInferFactory : public ShapeInferFactory {
public:
    explicit SnippetShapeInferFactory(const std::shared_ptr<ov::Node>& op);
    ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<snippets::op::Subgraph> m_subgraph = nullptr;
};
} // namespace node
} // namespace intel_cpu
} // namespace ov

