// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "snippets/lowered/shape_inference/shape_infer_instances.hpp"
#include "snippets/snippets_isa.hpp"
#include "openvino/op/select.hpp"
namespace ov {
namespace snippets {
namespace {
// broadcast_merge_into for VectorDimp
bool broadcast_merge_into(IShapeInferSnippets::VectorDims& dst,
                          const IShapeInferSnippets::VectorDims& src,
                          const ov::op::AutoBroadcastSpec& autob) {
    auto broadcast_merge_dim = [](size_t& dst, const size_t& d1, const size_t& d2) {
        if (d1 == d2 || d1 == 1 || d1 == IShapeInferSnippets::DYNAMIC_DIMENSION) {
            dst = d2;
        } else if (d2 == 1 || d2 == IShapeInferSnippets::DYNAMIC_DIMENSION) {
            dst = d1;
        } else {
           return false;
        }
        return true;
    };
    // Ranks are both static.
    const auto dst_rank = dst.size();
    const auto src_rank = src.size();
    switch (autob.m_type) {
        case ov::op::AutoBroadcastType::NONE:
            return true;
        case ov::op::AutoBroadcastType::NUMPY: {
            const auto new_rank = std::max(dst_rank, src_rank);
            IShapeInferSnippets::VectorDims dims(new_rank);
            bool success = true;
            for (int64_t i = 0; i < new_rank; i++) {
                auto dsti = i < (new_rank - dst_rank) ? 1 : dst[i - (new_rank - dst_rank)];
                auto srci = i < (new_rank - src_rank) ? 1 : src[i - (new_rank - src_rank)];
                success &= broadcast_merge_dim(dims[i], dsti, srci);
            }
            dst = std::move(dims);
            return success;
        }
        case ov::op::AutoBroadcastType::PDPD: {
            int64_t axis = autob.m_axis;
            if (src_rank > dst_rank || axis < -1)
                return false;

            axis = (axis == -1) ? (dst_rank - src_rank) : axis;
            if (src_rank + axis > dst_rank)
                return false;

            bool success = true;
            for (int64_t i = 0; i < src_rank; ++i) {
                if (dst[axis + i] != IShapeInferSnippets::DYNAMIC_DIMENSION &&
                    src[i] != IShapeInferSnippets::DYNAMIC_DIMENSION) {
                    if (src[i] > dst[axis + i])
                        return false;
                }
                success &= broadcast_merge_dim(dst[axis + i], dst[axis + i], src[i]);
            }
            return success;
        }
        default:
            OPENVINO_THROW("Unsupported auto broadcast type: ", autob.m_type);
    }
    return false;
}
bool merge_into(IShapeInferSnippets::VectorDims& dst, const IShapeInferSnippets::VectorDims& src) {
    auto merge_dim = [](size_t& dst, const size_t& d1, const size_t& d2) {
        if (d1 == d2 || d1 == IShapeInferSnippets::DYNAMIC_DIMENSION) {
            dst = d2;
        } else if (d2 == IShapeInferSnippets::DYNAMIC_DIMENSION) {
            dst = d1;
        } else {
            return false;
        }
        return true;
    };
    if (dst.size() != src.size())
        return false;

    bool success = true;
    for (int64_t i = 0; i < dst.size(); i++)
        success &= merge_dim(dst[i], dst[i], src[i]);
    return success;
}
} // namespace


IShapeInferSnippets::Result
entryNumpyBroadcasting::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes) {
        size_t max_rank = 0;
        size_t max_rank_idx = 0;
        for (size_t i = 0; i < input_shapes.size(); ++i) {
            auto item_rank = input_shapes[i].get().size();
            if (item_rank > max_rank) {
                max_rank = item_rank;
                max_rank_idx = i;
            }
        }
        auto output_shape = input_shapes[max_rank_idx].get();
        // use NUMPY broadcast rule
        for (size_t i = 0; i < input_shapes.size(); i++) {
            if (i == max_rank_idx)
                continue;

            auto& input_shape = input_shapes[i].get();
            const size_t offset = max_rank - input_shape.size();
            for (size_t j = 0; j < input_shape.size(); ++j) {
                if (input_shape[j] != output_shape[offset + j]) {
                    if (output_shape[offset + j] == 1) {
                        output_shape[offset + j] = input_shape[j];
                    } else if (input_shape[j] != 1) {
                        OPENVINO_THROW(input_shapes.size() == 1, "Got not broadcastable input shapes");
                    }
                }
            }
        }
        return { { std::move(output_shape) }, ShapeInferStatus::success };
}

template<class BroadcastOP>
BroadcastShapeInfer<BroadcastOP>::BroadcastShapeInfer(const std::shared_ptr<Node>& n) {
        static_assert(std::is_base_of<snippets::op::BroadcastMove, BroadcastOP>() ||
                      std::is_base_of<snippets::op::BroadcastLoad, BroadcastOP>(),
                      "This ShapeInfer class could be used only for BroadcastMove and BroadcastLoad operations.");
        const auto& broadcast = as_type_ptr<BroadcastOP>(n);
        OPENVINO_ASSERT(ov::is_type<BroadcastOP>(n), "Invalid node passed to BroadcastShapeInfer.",
                        "Expected ", typeid(BroadcastOP).name(), "got ", n->get_type_name());
        const auto last_dim = *broadcast->get_output_shape().rbegin();
        m_broadcasted_dim = last_dim.is_dynamic() ? IShapeInferSnippets::DYNAMIC_DIMENSION : last_dim.get_length();
}
template<class BroadcastOP>
IShapeInferSnippets::Result
BroadcastShapeInfer<BroadcastOP>::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes) {
    auto out_shape = input_shapes[0].get();
    out_shape.back() = m_broadcasted_dim;
    return {{out_shape}, ShapeInferStatus::success};
}

//// Note: we need to manually create template instances here, so they can be reused in Broadcast* headers.
template class BroadcastShapeInfer<op::BroadcastMove>;
template class BroadcastShapeInfer<op::BroadcastLoad>;

SelectShapeInfer::SelectShapeInfer(const std::shared_ptr<Node>& n) {
    const auto& select = as_type_ptr<ov::op::v1::Select>(n);
    OPENVINO_ASSERT(select, "Invalid node passed to SelectShapeInfer.");
    m_broadcast_spec = select->get_auto_broadcast();
}

IShapeInferSnippets::Result
SelectShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 3, "Invalid number of shapes passed SelectShapeInfer");
    VectorDims result_shape;
    if (m_broadcast_spec == ov::op::AutoBroadcastType::PDPD) {
        result_shape = input_shapes[1];  // 'then' tensor
        // in PDPD type, Broadcast-merging 'else' into 'then' one way not each other.
        OPENVINO_ASSERT(broadcast_merge_into(result_shape, input_shapes[2], m_broadcast_spec),
                        "'Else' tensor shape is not broadcastable.");
        OPENVINO_ASSERT(broadcast_merge_into(result_shape, input_shapes[0], m_broadcast_spec),
                        "'Cond' tensor shape is not broadcastable.");
    } else {
        result_shape = input_shapes[2];
        for (int input_port = 1; input_port >= 0; input_port--) {
            if (m_broadcast_spec.m_type == ov::op::AutoBroadcastType::NONE) {
                OPENVINO_ASSERT(merge_into(result_shape, input_shapes[input_port]),
                                "Argument shapes are inconsistent.");
            } else if (m_broadcast_spec.m_type == ov::op::AutoBroadcastType::NUMPY) {
                OPENVINO_ASSERT(broadcast_merge_into(result_shape, input_shapes[input_port], m_broadcast_spec),
                                "Argument shapes are inconsistent.");
            } else {
                OPENVINO_THROW("Unsupported auto broadcast specification");
            }
        }
    }
    return {{result_shape}, ShapeInferStatus::success};
}

} // namespace snippets
} // namespace ov
