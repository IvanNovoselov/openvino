// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_convolution.hpp"
#include <snippets/op/subgraph.hpp>
#include "common_test_utils/data_utils.hpp"
using namespace ov;
using ngraph::snippets::op::Subgraph;
namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ov::Model> ConvMulActivation::initOriginal() const {
    auto conv_param = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const auto batch = input_shapes[0][0];
    const auto channels = input_shapes[0][1];
    ngraph::Shape strides(2, 1);
    std::vector<ptrdiff_t> pad_begin(2, 1), pad_end(2, 1);
    const Shape const_shape {batch, channels, 3, 3};
    const std::vector<float> const_values = CommonTestUtils::generate_float_numbers(shape_size(const_shape), -10., 10.);
    auto weights = std::make_shared<op::v0::Constant>(precision, const_shape, const_values);
    auto conv = std::make_shared<op::v1::Convolution>(conv_param, weights, strides, pad_begin, pad_end, strides);

    auto mul_param = std::make_shared<op::v0::Parameter>(element::i32, input_shapes[1]);
    auto mul_convert = std::make_shared<op::v0::Convert>(mul_param, precision);

//    auto add = std::make_shared<op::v1::Add>(conv, add_convert);
    auto mul = std::make_shared<op::v1::Multiply>(conv, mul_convert);
    auto relu = std::make_shared<op::v0::Relu>(mul);
    auto sqrt = std::make_shared<ngraph::op::v0::Sqrt>(relu);

    return std::make_shared<ov::Model>(NodeVector{sqrt}, ParameterVector{conv_param, mul_param});
}
std::shared_ptr<ov::Model> ConvMulActivation::initReference() const {
    auto conv_param = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    ngraph::Shape strides(2, 1);
    std::vector<ptrdiff_t> pad_begin(2, 1), pad_end(2, 1);
    const auto batch = input_shapes[0][0];
    const auto channels = input_shapes[0][1];
    const Shape const_shape {batch, channels, 3, 3};
    const std::vector<float> const_values = CommonTestUtils::generate_float_numbers(shape_size(const_shape), -10., 10.);
    auto weights = std::make_shared<op::v0::Constant>(precision, const_shape, const_values);
    auto conv = std::make_shared<op::v1::Convolution>(conv_param, weights, strides, pad_begin, pad_end, strides);

    auto mul_param = std::make_shared<op::v0::Parameter>(element::i32, input_shapes[1]);
    auto mul_convert = std::make_shared<op::v0::Convert>(mul_param, precision);

    auto indata0 = std::make_shared<op::v0::Parameter>(precision, conv->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, mul_convert->get_shape());

    auto inmul = std::make_shared<op::v1::Multiply>(indata0, indata1);
    auto inrelu = std::make_shared<op::v0::Relu>(inmul);
    auto insqrt = std::make_shared<ngraph::op::v0::Sqrt>(inrelu);

    auto subgraph = std::make_shared<Subgraph>(NodeVector{conv, mul_convert},
                                          std::make_shared<ov::Model>(NodeVector{insqrt},
                                                                  ParameterVector{indata0, indata1}));
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{conv_param, mul_param});
}
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph