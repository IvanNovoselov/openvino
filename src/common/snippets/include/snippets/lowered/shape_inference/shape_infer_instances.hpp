// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shape_inference.hpp"

namespace ov {
namespace snippets {

bool broadcast_merge_into(IShapeInferSnippets::VectorDims& dst,
                          const IShapeInferSnippets::VectorDims& src,
                          const ov::op::AutoBroadcastSpec& autob = ov::op::AutoBroadcastType::NUMPY);

bool merge_into(IShapeInferSnippets::VectorDims& dst, const IShapeInferSnippets::VectorDims& src);

class entryNumpyBroadcasting : public IShapeInferSnippets {
public:
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes) override;
};


template<class BroadcastOP>
class  BroadcastShapeInfer : public IShapeInferSnippets {
    VectorDims::value_type m_broadcasted_dim;
public:
    explicit BroadcastShapeInfer(const std::shared_ptr<Node>& n);
    IShapeInferSnippets::Result
    infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes) override;
};

class entryFirstPassthrough : public IShapeInferSnippets {
public:
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes) override {
        std::vector<VectorDims> output_shapes {input_shapes[0].get()};
        return {std::move(output_shapes), ShapeInferStatus::success};
    }
};

class entryEmpty : public IShapeInferSnippets {
public:
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes) override {
        return {{}, ShapeInferStatus::success};
    }
};

class entrySingleElement : public IShapeInferSnippets {
public:
    IShapeInferSnippets::Result
    infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes) override {
        return {{{1}}, ShapeInferStatus::success};
    }
};

class SelectShapeInfer : public IShapeInferSnippets {
    ov::op::AutoBroadcastSpec m_broadcast_spec;
public:
    explicit SelectShapeInfer(const std::shared_ptr<Node>& n);
    IShapeInferSnippets::Result
    infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes) override;
};

} // namespace snippets
} // namespace ov
