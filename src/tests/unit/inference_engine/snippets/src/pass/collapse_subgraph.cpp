// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <subgraph_eltwise.hpp>
#include <snippets_helpers.hpp>

// Todo: Move this test to CPU-specific
TEST(TransformationTests, DoNotStartSubgraphAfterInputs) {
    // Do not start Subgraph after input parameters to avoid U8->FP32 and FP32->U8 conversion pairs
    // Todo: Remove this test when U8 support is enabled in SnippetS and StartSubgraph logics is updated
    GTEST_SKIP();
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        const std::vector<float> const_values{3, 2, 10};
        auto const_data = std::make_shared<op::v0::Constant>(element::f32, Shape{1, 3}, const_values);
        auto add = std::make_shared<op::v1::Add>(data0, data1);
        auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
        auto mul = std::make_shared<op::v1::Multiply>(add, sub);
        f = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});

        pass::Manager m;
        m.register_pass<InitNodeInfo>();
        // Todo: When moved to CPU-specific tests, uncomment the markup transformation below.
        //  m.register_pass<ov::intel_cpu::SnippetsMarkFused>();
        m.register_pass<EnumerateNodes>();
        m.register_pass<TokenizeSnippets>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    ASSERT_EQ(count_ops_of_type<Subgraph>(f), 0);
}

TEST_F(SnippetsCollapseSubgraphTests, EltwiseSubgraph) {
    function = ngraph::builder::subgraph::EltwiseFunction::getOriginal();
    function_ref = ngraph::builder::subgraph::EltwiseFunction::getReference();
    run(true);
}

TEST_F(SnippetsCollapseSubgraphTests, MatMulWithEltwiseBranchesSubgraph) {
    function = ngraph::builder::subgraph::MatMulEltwiseBranchesFunction::getOriginal();
    function_ref = ngraph::builder::subgraph::MatMulEltwiseBranchesFunction::getReference();
    run(true);
}

TEST_F(SnippetsCollapseSubgraphTests, AvoidLoopEltwiseSubgraphs) {
    function = ngraph::builder::subgraph::EltwiseLogLoop::getOriginal();
    function_ref = ngraph::builder::subgraph::EltwiseLogLoop::getReference();
    run(true);
}