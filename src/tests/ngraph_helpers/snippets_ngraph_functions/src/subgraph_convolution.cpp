// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_convolution.hpp"
#include <snippets/op/subgraph.hpp>
using namespace ov;
using ngraph::snippets::op::Subgraph;
namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ov::Model> ConvMulActivation::getOriginal() {
    const auto precision = element::f32;
    auto conv_param = std::make_shared<op::v0::Parameter>(precision, Shape{1, 2, 16, 16});
    ngraph::Shape strides(2, 1);
    std::vector<ptrdiff_t> pad_begin(2, 1), pad_end(2, 1);
    auto weights = std::make_shared<op::v0::Constant>(precision, Shape{1, 2, 3, 3});
    auto conv = std::make_shared<op::v1::Convolution>(conv_param, weights, strides, pad_begin, pad_end, strides);

    auto mul_param = std::make_shared<op::v0::Parameter>(element::i32, Shape{1, 2, 1, 16});
    auto mul_convert = std::make_shared<op::v0::Convert>(mul_param, precision);

//    auto add = std::make_shared<op::v1::Add>(conv, add_convert);
    auto mul = std::make_shared<op::v1::Multiply>(conv, mul_convert);
    auto relu = std::make_shared<op::v0::Relu>(mul);
    auto sqrt = std::make_shared<ngraph::op::v0::Sqrt>(relu);

    return std::make_shared<ov::Model>(NodeVector{sqrt}, ParameterVector{conv_param, mul_param});
}
std::shared_ptr<ov::Model> ConvMulActivation::getReference() {
    const auto precision = element::f32;
    auto conv_param = std::make_shared<op::v0::Parameter>(precision, Shape{1, 2, 16, 16});
    ngraph::Shape strides(2, 1);
    std::vector<ptrdiff_t> pad_begin(2, 1), pad_end(2, 1);
    auto weights = std::make_shared<op::v0::Constant>(precision, Shape{1, 2, 3, 3});
    auto conv = std::make_shared<op::v1::Convolution>(conv_param, weights, strides, pad_begin, pad_end, strides);

    auto mul_param = std::make_shared<op::v0::Parameter>(element::i32, Shape{1, 2, 1, 16});
    auto mul_convert = std::make_shared<op::v0::Convert>(mul_param, precision);

    auto indata0 = std::make_shared<op::v0::Parameter>(element::f32, conv->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(element::f32, mul_convert->get_shape());

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