// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lowered/pass/assign_registers.hpp"
#include "subgraph_mha.hpp"
#include "lir_test_utils.hpp"
#include "snippets/lowered/pass/serialize_control_flow.hpp"


namespace ov {
namespace test {
namespace snippets {

using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;



void AssignRegistersTest::SetUp() {
    LoweringTests::SetUp();

    std::vector<PartialShape> input_shapes{{2, 68, 6, 92}, {2, 68, 6, 92}, {1, 1, 68, 68}, {2, 68, 6, 92}};
    std::vector<element::Type> input_precisions(4, element::f32);
    const auto& body = std::make_shared<ov::test::snippets::MHAFunction>(input_shapes, input_precisions, true, false)->getOriginal();
    NodeVector subgraph_inputs;
    for (const auto& par : body->get_parameters())
        subgraph_inputs.push_back(par->clone_with_new_inputs({}));
    m_subgraph = std::make_shared<ov::snippets::op::Subgraph>(subgraph_inputs, body);
    m_subgraph->set_generator(std::make_shared<DummyGenerator>());
    m_subgraph->set_tile_rank(2);

}

TEST_F(AssignRegistersTest, AssignRegistersTest) {
    m_subgraph->data_flow_transformations();
    m_subgraph->control_flow_transformations();
    auto linear_ir = ov::snippets::op::SubgarphTestAccessor::get_subgraph_lir(m_subgraph);
    ov::snippets::lowered::pass::SerializeControlFlow("snsdebug_lir.xml").run(*linear_ir);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
