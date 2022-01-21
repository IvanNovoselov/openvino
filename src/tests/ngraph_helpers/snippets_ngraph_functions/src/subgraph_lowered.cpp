// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_lowered.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/op/subgraph.hpp>
#include <snippets/snippets_isa.hpp>
using namespace ov;
using ngraph::snippets::op::Subgraph;
namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ov::Model> EltwiseFunctionLowered::initLowered() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto load0 = std::make_shared<snippets::op::Load>(data0);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto load1 = std::make_shared<snippets::op::Load>(data1);

    const std::vector<float> const_values = CommonTestUtils::generate_float_numbers(shape_size(input_shapes[1]), -10., 10.);
    auto const_data = std::make_shared<op::v0::Constant>(precision, data1->get_shape(), const_values);
    auto add = std::make_shared<op::v1::Add>(data0, data1);
    auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
    auto mul = std::make_shared<op::v1::Multiply>(add, sub);
    return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1});
}
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph