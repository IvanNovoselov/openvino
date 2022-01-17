// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <subgraph_eltwise.hpp>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>

#include <snippets/snippets_isa.hpp>
#include <snippets/pass/collapse_subgraph.hpp>
#include <snippets/op/subgraph.hpp>

//#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
//#include "graph_comparator.hpp"
//#include "include/pass/collapse_subgraph.hpp"

using namespace testing;
using namespace ov;
using ngraph::snippets::op::Subgraph;
using ngraph::pass::InitNodeInfo;
using ngraph::snippets::pass::EnumerateNodes;
using ngraph::snippets::pass::TokenizeSnippets;

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

// todo: This transformation is required to pass ngraph::pass::CheckUniqueNames executed from TransformationTestsF
//  Note that other solutions are also possible:
//  - Modify TransformationTestsF to make it possible to disable CheckUniqueNames on demand.
//      Not favorable because several other checks inside CheckUniqueNames are quite useful (solution: modify CheckUniqueNames also?).
//  - Write our own class similar to TransformationTestsF, but a bit more flexible.
//      Not favorable because of code duplication.
//  - Slightly modify TokenizeSnippets subgraph naming policy. It currently tries to mimic the one used by MKLDNNPlugin
//      (node name before result may be changed, but tensor name must be preserved), but it's easy to update Subgraph name
//      (on demand) if it goes before the Result.
//      Not favorable because it seems that this alternative-naming-behavior is needed only to pass these tests.
//  - An alternative solution is to insert dummy non-tokenizable node before the Result (Such as Convert)
//      Not favorable because we won't be able to test that the tensor names are preserved for the Result inputs.
class SnippetsRestoreResultInputName: public ngraph::pass::MatcherPass {
public:
    SnippetsRestoreResultInputName(){
        auto label =
                std::make_shared<ngraph::pattern::op::Label>(ngraph::pattern::any_input(),
                                                             [](const std::shared_ptr<const Node> &n) {
                                                                      return is_type<op::v0::Result>(n) &&
                                                                             is_type<Subgraph>(n->get_input_source_output(0).get_node_shared_ptr());
                                                                  });
        ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher &m) -> bool {
            const auto& node = m.get_match_root();
            const auto& subgraph = as_type_ptr<Subgraph>(node->get_input_source_output(0).get_node_shared_ptr());
            bool not_set = true;
            for (unsigned int i = 0; i < subgraph->get_output_size() && not_set; i++) {
                for (const auto &in : subgraph->get_output_target_inputs(i)) {
                    if (ov::is_type<op::v0::Result>(in.get_node())) {
                        const auto& body_result = subgraph->get_body()->get_output_op(i);
                        const auto& body_result_input = body_result->get_input_source_output(0);
                        subgraph->set_friendly_name(body_result_input.get_node_shared_ptr()->get_friendly_name());
                        not_set = false;
                        break;
                    }
                }
            }
            return true;
        };
        auto matcher = std::make_shared<ngraph::pattern::Matcher>(label);
        register_matcher(matcher, callback);
    }
};
class SnippetsCollapseSubgraphTests : public TransformationTestsF {
public:
    void run(bool serialize = false) {
        ASSERT_TRUE(function);
        if (serialize) {
            auto formatName = [](const std::string& original_name) {
                std::string name(original_name);
                std::replace(name.begin(), name.end(), '\\', '_');
                std::replace(name.begin(), name.end(), '/', '_');
                std::replace(name.begin(), name.end(), ' ', '_');
                std::replace(name.begin(), name.end(), ':', '-');
                return name;
            };
            std::string name = formatName(function->get_friendly_name());
            if (name.empty())
                name = "subgraph";
            manager.register_pass<ov::pass::Serialize>(name + ".xml", name + ".bin");
        }
        manager.register_pass<EnumerateNodes>();
        manager.register_pass<TokenizeSnippets>();
        manager.register_pass<SnippetsRestoreResultInputName>();
    }
};

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