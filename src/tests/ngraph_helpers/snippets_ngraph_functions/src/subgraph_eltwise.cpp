// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_eltwise.hpp"
#include <snippets/op/subgraph.hpp>
using namespace ov;
using ngraph::snippets::op::Subgraph;
namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ov::Model> EltwiseFunction::getOriginal() {
    auto data0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto data1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
    const std::vector<float> const_values{3, 2, 10};
    auto const_data = std::make_shared<op::v0::Constant>(element::f32, Shape{1, 3}, const_values);
    auto add = std::make_shared<op::v1::Add>(data0, data1);
    auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
    auto mul = std::make_shared<op::v1::Multiply>(add, sub);
    return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> EltwiseFunction::getReference() {
    auto data0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto data1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
    const std::vector<float> const_values{3, 2, 10};
    auto const_data = std::make_shared<op::v0::Constant>(element::f32, Shape{1, 3}, const_values);
    auto indata0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto indata1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
    auto indata2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
    auto add = std::make_shared<op::v1::Add>(indata0, indata1);
    auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
    auto mul = std::make_shared<Subgraph>(NodeVector{data0, data1, const_data},
                                          std::make_shared<ov::Model>(NodeVector{std::make_shared<op::v1::Multiply>(add, sub)},
                                                                  ParameterVector{indata0, indata1, indata2}));
    return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> MatMulEltwiseBranchesFunction::getOriginal() {
    auto data_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 4, 4 });
    auto data_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 4, 4 });
    auto non_snippet_op = std::make_shared<op::v0::MatMul>(data_1, data_2);

    auto mul_const_1 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 4.f });
    auto mul_1 = std::make_shared<op::v1::Multiply>(non_snippet_op, mul_const_1);
    auto add_const_1 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 16.4f });
    auto add_1 = std::make_shared<op::v1::Add>(mul_1, add_const_1);
    auto elu = std::make_shared<op::v0::Elu>(add_1, 0.01);

    auto mul_const_2 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 4.f });
    auto mul_2 = std::make_shared<op::v1::Multiply>(non_snippet_op, mul_const_2);
    auto sub_const_2 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 16.4f });
    auto sub_2 = std::make_shared<op::v1::Subtract>(mul_2, sub_const_2);
    auto relu = std::make_shared<op::v0::Relu>(sub_2);

    auto add = std::make_shared<op::v1::Add>(elu, relu);
    auto result = std::make_shared<op::v0::Result>(add);

    return std::make_shared<Model>(ResultVector{ result }, ParameterVector{ data_1, data_2 });
}

std::shared_ptr<ov::Model> MatMulEltwiseBranchesFunction::getReference() {
    auto data_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 4, 4 });
    auto data_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 4, 4 });

    // snippet inputs
    auto non_snippet_op = std::make_shared<op::v0::MatMul>(data_1, data_2);
    auto mul_const_1 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 4.f });
    auto add_const_1 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 16.4f });
    auto mul_const_2 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 4.f });
    auto add_const_2 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 16.4f });

    // snippet function
    auto snippet_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 4, 4 });

    auto sn_mul_const_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 1, 1 });
    auto mul_1 = std::make_shared<op::v1::Multiply>(snippet_input, sn_mul_const_1);
    auto sn_add_const_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 1, 1 });
    auto add_1 = std::make_shared<op::v1::Add>(mul_1, sn_add_const_1);
    auto elu = std::make_shared<op::v0::Elu>(add_1, 0.01);

    auto sn_mul_const_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 1, 1 });
    auto mul_2 = std::make_shared<op::v1::Multiply>(snippet_input, sn_mul_const_2);
    auto sn_sub_const_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 1, 1 });
    auto sub_2 = std::make_shared<op::v1::Subtract>(mul_2, sn_sub_const_2);
    auto relu = std::make_shared<op::v0::Relu>(sub_2);

    auto add = std::make_shared<op::v1::Add>(elu, relu);
    ParameterVector subgraph_params{ snippet_input, sn_mul_const_1, sn_add_const_1, sn_mul_const_2, sn_sub_const_2 };
    auto snippet_function = std::make_shared<Model>(NodeVector{ add }, subgraph_params);

    ngraph::NodeVector snippet_inputs{ non_snippet_op, mul_const_1, add_const_1, mul_const_2, add_const_2 };
    auto snippet = std::make_shared<Subgraph>(snippet_inputs, snippet_function);
    auto result = std::make_shared<op::v0::Result>(snippet);

    return std::make_shared<Model>(NodeVector{ result }, ParameterVector{ data_1, data_2 });
}

std::shared_ptr<ov::Model> EltwiseLogLoop::getOriginal() {
    auto data0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto data1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
    auto add = std::make_shared<op::v1::Add>(data0, data1);
    auto hswish = std::make_shared<op::v4::HSwish>(add);
    auto log = std::make_shared<op::v0::Log>(add);
    auto mul = std::make_shared<op::v1::Multiply>(hswish, log);
    return std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> EltwiseLogLoop::getReference() {
    auto data0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto data1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
    auto indata0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto indata1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
    auto inAdd = std::make_shared<op::v1::Add>(indata0, indata1);
    auto inHswish = std::make_shared<op::v4::HSwish>(inAdd);
    auto body = std::make_shared<Model>(NodeVector{inAdd, inHswish}, ParameterVector{indata0, indata1});
    auto subgraph = std::make_shared<Subgraph>(NodeVector{data0, data1}, body);
    auto log = std::make_shared<op::v0::Log>(subgraph->output(0));
    //Note that log is not currently supported by snippets, so it won't be converted to subgraph.
    // Todo: Note that collapse_subgraph changes the output ports so that the input subgraph's outputs come
    //  before the node outputs. So the Subgraph{Add}.output(1)->Log{} becomes Subgraph{Add+Hswish}.output(0)->Log{}
    auto subgraph_param = std::make_shared<op::v0::Parameter>(element::f32, subgraph->get_output_shape(1));
    auto log_param = std::make_shared<op::v0::Parameter>(element::f32, log->get_output_shape(0));
    auto mul = std::make_shared<Subgraph>(OutputVector{subgraph->output(1), log->output(0)},
                                          std::make_shared<Model>(NodeVector{std::make_shared<op::v1::Multiply>(subgraph_param, log_param)},
                                                                  ParameterVector{subgraph_param, log_param}));
    return std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});
}
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph