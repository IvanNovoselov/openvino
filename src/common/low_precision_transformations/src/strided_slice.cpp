// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/strided_slice.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

#include "ngraph/pass/serialize.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

namespace {

std::shared_ptr<opset1::Constant> stridedSliceDeqConstant(
    const std::shared_ptr<ngraph::Node> strSlice,
    const std::shared_ptr<ngraph::Node> dequantizaitonConstant) {
    // step #1: prerequisistes
    const auto constant = ov::as_type_ptr<ngraph::opset1::Constant>(dequantizaitonConstant);
    const auto& original_constant_shape = constant->get_shape();
    if (shape_size(original_constant_shape) == 1ul) {
        return NetworkHelper::toScalar(constant);
    }

    const auto strided_slice = ov::as_type_ptr<ngraph::opset1::StridedSlice>(strSlice);

    // step #2: aligh shapes
    std::shared_ptr<ngraph::opset1::Constant> new_constant = constant;
    const size_t rank = strSlice->get_input_partial_shape(0).rank().get_length();
    ngraph::Shape newConstantShape = original_constant_shape;
    if (rank != newConstantShape.size()) {
        if (ngraph::shape_size(original_constant_shape) == 1) {
            newConstantShape = ngraph::Shape(rank, 1);
        } else {
            newConstantShape = original_constant_shape;

            // case when constShape without batch
            if ((original_constant_shape.size() > 1) &&
                (original_constant_shape.size() < rank)) {
                newConstantShape.insert(newConstantShape.begin(), 1);
            }
        }

        if (original_constant_shape != newConstantShape) {
            const auto newConstant = fold<ngraph::opset1::Broadcast>(
                constant,
                ngraph::opset1::Constant::create(ngraph::element::i32, { newConstantShape.size() }, newConstantShape));
            new_constant = ov::as_type_ptr<ngraph::opset1::Constant>(newConstant);
        }
    }

    // step #3: update original begin & end & strides
    auto begin = ov::as_type_ptr<opset1::Constant>(strided_slice->get_input_node_shared_ptr(1))->cast_vector<int64_t>();
    auto end = ov::as_type_ptr<opset1::Constant>(strided_slice->get_input_node_shared_ptr(2))->cast_vector<int64_t>();
    auto strides = ov::as_type_ptr<opset1::Constant>(strided_slice->get_input_node_shared_ptr(3))->cast_vector<int64_t>();
    for (auto i = 0ull; i < newConstantShape.size(); ++i) {
        if (newConstantShape[i] == 1ull) {
            if (i < begin.size()) {
                begin[i] = 0;
            }
            if (i < end.size()) {
                end[i] = 1;
            }

            if (i < strides.size()) {
                strides[i] = 1;
            }
        }
    }

    const auto result = fold<ngraph::opset1::StridedSlice>(
        new_constant,
        std::make_shared<opset1::Constant>(element::i64, Shape{ begin.size() }, begin),
        std::make_shared<opset1::Constant>(element::i64, Shape{ end.size() }, end),
        std::make_shared<opset1::Constant>(element::i64, Shape{ strides.size() }, strides),
        strided_slice->get_begin_mask(),
        strided_slice->get_end_mask(),
        strided_slice->get_new_axis_mask(),
        strided_slice->get_shrink_axis_mask(),
        strided_slice->get_ellipsis_mask());

    new_constant = ov::as_type_ptr<opset1::Constant>(NetworkHelper::toScalarIfPossible(result));

    if (shape_size(new_constant->get_shape()) == 1ul) {
        return NetworkHelper::toScalar(new_constant);
    }

    return new_constant;
}

} // namespace

StridedSliceTransformation::StridedSliceTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(StridedSliceTransformation);
    auto matcher = ngraph::pattern::wrap_type<opset1::StridedSlice>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool StridedSliceTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    if (!StridedSliceTransformation::canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto stridedSlice = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    auto dequantization = NetworkHelper::getDequantization(stridedSlice, defaultPrecisions);

    if (dequantization.subtract) {
        ngraph::pass::VisualizeTree("svg/lpt.strided_slice.1.svg").run_on_model(ov::Model::global_model);
        ngraph::pass::Serialize("svg/lpt.strided_slice.1.xml", "svg/cpu.strided_slice.1.bin").run_on_model(ov::Model::global_model);

        if (stridedSlice->get_friendly_name() == "Gs/_Run/Gs/G_synthesis/4x4/Conv/strided_slice") {
            std::cout << "StridedSliceTransformation::transform: " << stridedSlice->get_friendly_name() << std::endl;
        }
        if (stridedSlice->get_friendly_name() == "Gs/_Run/Gs/G_synthesis/4x4/ToRGB/strided_slice") {
            std::cout << "StridedSliceTransformation::transform: " << stridedSlice->get_friendly_name() << std::endl;
        }

        const auto newSubConst = stridedSliceDeqConstant(stridedSlice, dequantization.subtractConstant);
        replace_node(dequantization.subtractConstant, newSubConst);
        dequantization.subtractConstant = newSubConst;
    }

    const auto newMulConst = stridedSliceDeqConstant(stridedSlice, dequantization.multiplyConstant);
    replace_node(dequantization.multiplyConstant, newMulConst);
    dequantization.multiplyConstant = newMulConst;

    moveDequantizationAfter(context, stridedSlice, NetworkHelper::getDequantization(stridedSlice, defaultPrecisions), false);
    return true;
}

bool StridedSliceTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!ov::is_type<ngraph::opset1::StridedSlice>(operation)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(operation);
    if (dequantization.empty()) {
        return false;
    }

    const auto input_shape = operation->get_input_partial_shape(0);
    if (input_shape.rank().is_dynamic() &&
        ((dequantization.subtract && ngraph::shape_size(dequantization.subtractConstant->get_shape()) > 1) ||
         (dequantization.multiply && ngraph::shape_size(dequantization.multiplyConstant->get_shape()) > 1))) {
        return false;
    }

    if (input_shape.is_dynamic()) {
        //for (auto i = 0ull; i < input_shape.rank().get_length(); ++i) {
        //    // TODO: check operation parameters
        //    if (input_shape[i].is_dynamic() &&
        //        ((dequantization.subtract && (ngraph::shape_size(dequantization.subtractConstant->get_shape()) > 1) && (dequantization.subtractConstant->get_shape()[i] > 1)) ||
        //        (dequantization.multiply && (ngraph::shape_size(dequantization.multiplyConstant->get_shape()) > 1) && (dequantization.multiplyConstant->get_shape()[i] > 1)))) {
        //        return false;
        //    }
        //}

        //const auto stridedSlice = ov::as_type_ptr<ngraph::opset1::StridedSlice>(operation);
        //const auto& shrink_axis_mask = stridedSlice->get_shrink_axis_mask();
        //if (std::any_of(shrink_axis_mask.begin(), shrink_axis_mask.end(), [](const int64_t value) { return value != 0; })) {
        //    // TODO: dynamic shape
        //    const ngraph::Shape data_shape = stridedSlicePShapeIn.to_shape();
        //    newConstantShape = data_shape;
        //}
    }

    return true;
}

bool StridedSliceTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
