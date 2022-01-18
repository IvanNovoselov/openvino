// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "ngraph/function.hpp"
#include "ngraph/pass/manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"

using ngraph::snippets::op::Subgraph;
//using ngraph::pass::InitNodeInfo;
using ngraph::snippets::pass::EnumerateNodes;
using ngraph::snippets::pass::TokenizeSnippets;
using namespace ov;
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
    void run(bool serialize_before = false, bool serialize_after = false, bool serialize_ref = false) {
        ASSERT_TRUE(function);
        std::string name;
        if (serialize_before || serialize_after || serialize_ref) {
            auto formatName = [](const std::string& original_name) {
                std::string name(original_name);
                std::replace(name.begin(), name.end(), '\\', '_');
                std::replace(name.begin(), name.end(), '/', '_');
                std::replace(name.begin(), name.end(), ' ', '_');
                std::replace(name.begin(), name.end(), ':', '-');
                return name;
            };
            name = formatName(function->get_friendly_name());
            if (name.empty())
                name = "subgraph";
        }
        if (serialize_ref && function_ref)
            ov::pass::Serialize(name + "_ref.xml", name + "_ref.bin").run_on_model(function_ref);
        if (serialize_before)
            manager.register_pass<ov::pass::Serialize>(name + "_before.xml", name + "_before.bin");
        manager.register_pass<EnumerateNodes>();
        manager.register_pass<TokenizeSnippets>();
        if (serialize_after)
            manager.register_pass<ov::pass::Serialize>(name + "_after.xml", name + "_after.bin");
        manager.register_pass<SnippetsRestoreResultInputName>();
    }
};
/// Base class for snippets-related subgraphs
class SnippetsFunctionBase {
public:
    SnippetsFunctionBase() = delete;
    SnippetsFunctionBase(std::vector<Shape>& inputShapes) : input_shapes{inputShapes} {};

    std::shared_ptr<ov::Model> getReference() const {
        std::shared_ptr<Model> function_ref = initReference();
        validate_function(function_ref);
        return function_ref;
    }
    std::shared_ptr<ov::Model> getOriginal() const {
        std::shared_ptr<Model> function = initOriginal();
        validate_function(function);
        return function;
    }
    size_t getNumInputs() const {return input_shapes.size();}

protected:
    virtual std::shared_ptr<ov::Model> initOriginal() const = 0;
    virtual std::shared_ptr<ov::Model> initReference() const  = 0;
    // only fp32 is currently supported by snippets
    ov::element::Type_t precision  = element::f32;
    std::vector<Shape> input_shapes;

private:
    void validate_function(const std::shared_ptr<Model>& f) const {
        NGRAPH_CHECK(f != nullptr, "The test requires Model to be defined");
        const auto &params = f->get_parameters();
        NGRAPH_CHECK(params.size() == input_shapes.size(),
                     "Passed input shapes and produced function are inconsistent.");
        for (size_t i = 0; i < input_shapes.size(); i++)
            NGRAPH_CHECK(std::equal(input_shapes[i].begin(), input_shapes[i].end(), params[i]->get_shape().begin()),
                         "Passed input shapes and produced function are inconsistent.");
    }
};