// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_eltwise.hpp"
#include <snippets/op/subgraph.hpp>
using namespace ov;
namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ov::Model> AddSubtractMultiplyFunction::getOriginal() {
    auto data0 = std::make_shared<op::v0::Parameter>(element::i32, Shape{2, 3});
    auto data1 = std::make_shared<op::v0::Parameter>(element::i32, Shape{1, 3});
    auto convert0 = std::make_shared<op::v0::Convert>(data0, element::f32);
    auto convert1 = std::make_shared<op::v0::Convert>(data1, element::f32);
    const std::vector<float> const_values{3, 2, 10};
    auto const_data = std::make_shared<op::v0::Constant>(element::f32, Shape{1, 3}, const_values);
    auto add = std::make_shared<op::v1::Add>(convert0, convert1);
    auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
    auto mul = std::make_shared<op::v1::Multiply>(add, sub);
    return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> AddSubtractMultiplyFunction::getReference() {
    auto data0 = std::make_shared<op::v0::Parameter>(element::i32, Shape{2, 3});
    auto data1 = std::make_shared<op::v0::Parameter>(element::i32, Shape{1, 3});
    auto convert0 = std::make_shared<op::v0::Convert>(data0, element::f32);
    auto convert1 = std::make_shared<op::v0::Convert>(data1, element::f32);
    const std::vector<float> const_values{3, 2, 10};
    auto const_data = std::make_shared<op::v0::Constant>(element::f32, Shape{1, 3}, const_values);
    auto indata0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto indata1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
    auto indata2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
    auto add = std::make_shared<op::v1::Add>(indata0, indata1);
    auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
    auto mul = std::make_shared<ngraph::snippets::op::Subgraph>(NodeVector{convert0, convert1, const_data},
                                          std::make_shared<ov::Model>(NodeVector{std::make_shared<op::v1::Multiply>(add, sub)},
                                                                  ParameterVector{indata0, indata1, indata2}));
    return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1});
}
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph