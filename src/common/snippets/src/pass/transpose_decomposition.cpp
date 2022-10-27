// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/transpose_decomposition.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/load.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/partial_shape.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/manager.hpp>
#include <numeric>

ngraph::snippets::pass::TransposeDecomposition::TransposeDecomposition() {
    MATCHER_SCOPE(TransposeDecomposition);

    auto match_data = pattern::any_input();
    auto match_order = ngraph::pattern::wrap_type<opset1::Constant>();
    auto match_transpose = ngraph::pattern::wrap_type<ngraph::opset1::Transpose>({match_data, match_order});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::TransposeDecomposition")
        auto& pattern_to_output = m.get_pattern_value_map();
        const auto transpose = ov::as_type_ptr<ngraph::opset1::Transpose>(
                                                            pattern_to_output.at(match_transpose).get_node_shared_ptr());

        const auto order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(match_order).get_node_shared_ptr());
        if (transformation_callback(transpose) || order->get_output_element_type(0) != ov::element::i32)
            return false;

        auto order_value = order->get_vector<int>();
        std::set<std::vector<int>> supported_cases{{0,2,3,1}};
        if (supported_cases.count(order_value) == 0)
            throw ngraph::ngraph_error("TransposeDecomposition: unsupported order");


        // NCHW -> NHWC
        /*
        for (size_t n = 0; n < N; n++) {            // should be handled by external Tile (outer)
            for (size_t h = 0; h < H; h++) {        // should be handled by external Tile (inner)
                size_t src_offset = n*H*W*C +  h*W;
                size_t dst_offset = n*H*W*C + h*W*C;
                for (size_t w = 0; w < W; w++) {
                    for (size_t c = 0; c < C; c++) {
                        (dst_ptr + dst_offset)[w*c + c] = (src_ptr + src_offset)[c*H*W + w];
                        src_ptr[n*H*W*C + h*W + c*H*W + w]
                    }
                }
            }
        }
         */
        uni_vmovups(vmm, ptr[reg_src]);
        uni_vmovups(ptr[reg_dst], vmm);
        size_t src_offset =
        auto load = std::make_shared<snippets::op::Load>(pattern_to_output.at(match_data), 1, src_offset);

        float* src_ptr;
        float* dst_ptr;
        size_t data_size = sizeof(float);
        for (size_t n = 0; n < N; n++) {
            for (size_t h = 0; h < H; h++) {
                float* src_inner_ptr = src_ptr + n*H*W*C +  h*W;
                for (size_t w = 0; w < W; w++) {
                    for (size_t c = 0; c < C; c++) {
                        // src_ptr[n*H*W*C + h*W + c*H*W + w]
                        *dst_ptr = *src_inner_ptr;
                        dst_ptr += data_size; // it's easy with dst, since scheduling is dst-dense => simply increment by data size
                        src_inner_ptr += H*W * data_size; // access to src is HW strided
                    }
                    src_inner_ptr += (w - H*W*C) * data_size;
                }
                // src_inner_ptr is already incremented by W at this point => do nothing
            }
            // src_inner_ptr is already incremented by H*W at this point => increment by (C-1)*H*W
            src_inner_ptr += (C-1)*H*W;
        }


        Output<Node> data{fake_quantize_node->input_value(0)};
        const Output<Node> input_low{fake_quantize_node->input_value(1)};
        const Output<Node> input_high{fake_quantize_node->input_value(2)};
        const Output<Node> output_low{fake_quantize_node->input_value(3)};
        const Output<Node> output_high{fake_quantize_node->input_value(4)};
        auto input_type = data.get_element_type();

        std::vector<float> out_scales;
        std::vector<float> cl, ch, isc, ish, osc, osh;
        const bool status = getScalesAndShifts(fake_quantize_node, cl, ch, isc, ish, osc, osh);
        if (status) {
            out_scales = calculateScales(fake_quantize_node->get_output_element_type(0), cl, ch, isc, ish, osc, osh);
        }
        const bool do_dequantize = !(status && ((std::all_of(osc.cbegin(),
                                                             osc.cend(),
                                                             [](float val) {
                                                                 return val == 1.f;
                                                             }) &&
                                                 std::all_of(osh.cbegin(),
                                                             osh.cend(),
                                                             [](float val) {
                                                                 return val == 0.f;
                                                             })) ||
                                                out_scales.size() != 0));
        const bool do_rounding = do_dequantize || fake_quantize_node->get_output_element_type(0) == ngraph::element::f32;

        ngraph::NodeVector decomp_ops;
        if (input_type != input_low.get_element_type()) {
            input_type = input_low.get_element_type();
            data = std::make_shared<ngraph::snippets::op::ConvertSaturation>(data, input_type);
            decomp_ops.push_back(data.get_node_shared_ptr());
        }

        // if we set input_low or input_high in formula we got output = output_low and output = output_high
        // respectively so we just clamp x
        const auto max = std::make_shared<ngraph::opset1::Maximum>(data, input_low);
        const auto min = std::make_shared<ngraph::opset1::Minimum>(max, input_high);
        decomp_ops.push_back(max);
        decomp_ops.push_back(min);

        std::shared_ptr<ngraph::Node> result = nullptr;
        if (out_scales.size() != 0) {
            PartialShape scale_shape = input_low.get_partial_shape();
            ngraph::PartialShape::broadcast_merge_into(scale_shape,
                                                       input_high.get_partial_shape(),
                                                       ov::op::AutoBroadcastType::NUMPY);
            const auto scales =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, scale_shape.get_shape(), out_scales);
            decomp_ops.push_back(scales);

            result = std::make_shared<ngraph::opset1::Multiply>(min, scales);
            decomp_ops.push_back(result);
        } else {
            // (levels-1)
            const auto levels_minus_one =
                std::make_shared<ngraph::opset1::Constant>(input_type, Shape{}, fake_quantize_node->get_levels() - 1);
            decomp_ops.push_back(levels_minus_one);
            // (input_high - input_low)
            const auto subInHighLow = std::make_shared<ngraph::opset1::Subtract>(input_high, input_low);
            // (levels-1) / (input_high - input_low)
            const auto isc = std::make_shared<ngraph::opset1::Divide>(levels_minus_one, subInHighLow);
            // input_low * (levels-1) / (input_high - input_low)
            const auto ish = std::make_shared<ngraph::opset1::Multiply>(input_low, isc);
            decomp_ops.push_back(subInHighLow);
            decomp_ops.push_back(isc);
            decomp_ops.push_back(ish);

            // x * (levels-1) / (input_high - input_low)
            const auto after_isc_apply = std::make_shared<ngraph::opset1::Multiply>(min, isc);
            // x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low)
            result = std::make_shared<ngraph::opset1::Subtract>(after_isc_apply, ish);
            decomp_ops.push_back(after_isc_apply);
            decomp_ops.push_back(result);
        }

        if (do_rounding) {
            // round(x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low))
            result = std::make_shared<ngraph::opset5::Round>(result, ngraph::opset5::Round::RoundMode::HALF_TO_EVEN);
            decomp_ops.push_back(result);
        }

        if (do_dequantize) {
            // (levels-1)
            const auto levels_minus_one =
                std::make_shared<ngraph::opset1::Constant>(input_type, Shape{}, fake_quantize_node->get_levels() - 1);
            // (output_high - output_low)
            const auto sub_out_high_low = std::make_shared<ngraph::opset1::Subtract>(output_high, output_low);
            // (output_high - output_low) / (levels-1)
            const auto osc = std::make_shared<ngraph::opset1::Divide>(sub_out_high_low, levels_minus_one);
            decomp_ops.push_back(sub_out_high_low);
            decomp_ops.push_back(osc);

            // round(x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low)) *
            // (output_high - output_low) / (levels-1)
            const auto after_osc_apply = std::make_shared<ngraph::opset1::Multiply>(result, osc);
            // round(x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low)) *
            // (output_high - output_low) / (levels-1) + output_low
            result = std::make_shared<ngraph::opset1::Add>(after_osc_apply, output_low);
            decomp_ops.push_back(after_osc_apply);
            decomp_ops.push_back(result);
        }

        if (result->get_output_element_type(0) != fake_quantize_node->get_output_element_type(0)) {
            result = std::make_shared<snippets::op::ConvertSaturation>(result, fake_quantize_node->get_output_element_type(0));
            decomp_ops.push_back(result);
        }

        result->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(fake_quantize_node, decomp_ops);
        ngraph::replace_node(m.get_match_root(), result);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fake_quantize, matcher_name);
    register_matcher(m, callback);
}
