// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <subgraph_lowered.hpp>
#include <snippets_helpers.hpp>
#include "snippets/op/subgraph.hpp"

class DummyEmitter : public ngraph::snippets::Emitter {
public:
    DummyEmitter() : ngraph::snippets::Emitter(std::make_shared<ov::op::v1::Add>()) {}
    void emit_code(const std::vector<size_t>& in,
                           const std::vector<size_t>& out,
                           const std::vector<size_t>& pool,
                           const std::vector<size_t>& gpr) const override {}
    void emit_data() const override {}
};

class DummyTargetMachine : public ngraph::snippets::TargetMachine {
public:
    DummyTargetMachine() {
        auto dummy_functor = [this](const std::shared_ptr<ngraph::Node>& n) {
            return std::make_shared<DummyEmitter>();
        };
        jitters[op::v0::Parameter::get_type_info_static()] = dummy_functor;
        jitters[op::v0::Constant::get_type_info_static()] = dummy_functor;
        jitters[op::v0::Result::get_type_info_static()] = dummy_functor;
        jitters[op::v1::Add::get_type_info_static()] = dummy_functor;
        jitters[op::v1::Subtract::get_type_info_static()] = dummy_functor;
        jitters[op::v1::Multiply::get_type_info_static()] = dummy_functor;
        jitters[op::v1::Multiply::get_type_info_static()] = dummy_functor;
        jitters[ngraph::snippets::op::Load::get_type_info_static()] = dummy_functor;
        jitters[ngraph::snippets::op::VectorLoad::get_type_info_static()] = dummy_functor;
        jitters[ngraph::snippets::op::ScalarLoad::get_type_info_static()] = dummy_functor;
        jitters[ngraph::snippets::op::BroadcastLoad::get_type_info_static()] = dummy_functor;

        jitters[ngraph::snippets::op::Store::get_type_info_static()] = dummy_functor;
        jitters[ngraph::snippets::op::VectorStore::get_type_info_static()] = dummy_functor;
        jitters[ngraph::snippets::op::ScalarStore::get_type_info_static()] = dummy_functor;

        jitters[ngraph::snippets::op::Scalar::get_type_info_static()] = dummy_functor;
        jitters[ngraph::snippets::op::BroadcastMove::get_type_info_static()] = dummy_functor;
        jitters[ngraph::snippets::op::Kernel::get_type_info_static()] = dummy_functor;
        jitters[ngraph::snippets::op::Tile::get_type_info_static()] = dummy_functor;
    }
    bool is_supported() const override { return true; }
    ngraph::snippets::code get_snippet() const override { return nullptr; }
    size_t get_lanes() const override { return 1; }
};

class DummyGenerator : public ngraph::snippets::Generator {
public:
    DummyGenerator() : ngraph::snippets::Generator(std::make_shared<DummyTargetMachine>()) {}
};


typedef std::tuple<
        InferenceEngine::Precision,  // Network Precision
        InferenceEngine::SizeVector, // Input 0 Shape
        InferenceEngine::SizeVector, // Input 1 Shape
        std::string                  // Target Device
> multiInputParams;
using ngraph::snippets::op::Subgraph;
class SnippetsLoweringTests : public TransformationTestsF {
public:
    void run() {
        ASSERT_TRUE(function);
        Subgraph::BlockedShapeVector empty;
        NGRAPH_CHECK(input_blocked_shapes != empty,
                     "Input blocking shapes must be set and non-empty for this test");
        NGRAPH_CHECK(output_blocked_shapes != empty,
                     "Output blocking shapes must be set and non-empty for this test");
        serialize("before", function);
        {
            ngraph::pass::Manager m;
            m.register_pass<EnumerateNodes>();
            m.register_pass<TokenizeSnippets>();
            m.run_passes(function);
        }
//        serialize("tokenized", function);
        std::shared_ptr<Subgraph> subgraph;
        for (const auto &op : function->get_ops()) {
            bool is_subgraph = is_type<Subgraph>(op);
            if (is_subgraph) {
                NGRAPH_CHECK(subgraph.use_count() == 0,
                             "Functions provided for lowering tests contains more than one subgraph.");
                subgraph = as_type_ptr<Subgraph>(op);
            }
            NGRAPH_CHECK(is_subgraph ||
                         is_type<ov::op::v0::Parameter>(op) ||
                         is_type<ov::op::v0::Constant>(op) ||
                         is_type<ov::op::v0::Result>(op),
                         "Functions provided for lowering tests is not fully tokenizable");
        }
        serialize("tokenized", subgraph->get_body());
        std::cerr << subgraph->get_body()->get_parameters().size() << "\n";
        subgraph->set_generator(std::make_shared<DummyGenerator>());
        subgraph->generate(output_blocked_shapes, input_blocked_shapes);
        function = subgraph->get_body();
        serialize("lowered", function);
//        manager.register_pass<SnippetsRestoreResultInputName>();
    }
    Subgraph::BlockedShapeVector input_blocked_shapes;
    Subgraph::BlockedShapeVector output_blocked_shapes;

private:
    static void serialize(const std::string& name, const std::shared_ptr<Model>& m) {
        ov::pass::Serialize(name + ".xml", name + ".bin").run_on_model(m);
    }
};

using namespace ngraph::builder::subgraph;
TEST_F(SnippetsLoweringTests, EltwiseSubgraph) {
//    std::vector<Shape> inputShapes{{1, 32, 2, 3}, {1, 32, 1, 3}};
    Subgraph::BlockedShape is1{{1, 4, 5, 3, 8}, {0, 1, 2, 3, 1}, ngraph::element::f32};
    Subgraph::BlockedShape is2{{1, 4, 1, 3, 8}, {0, 1, 2, 3, 1}, ngraph::element::f32};
    std::vector<Shape> inputShapes{std::get<0>(is1), std::get<0>(is2)};
    // input for internal constant
    Subgraph::BlockedShape is3{{1, 4, 1, 3, 8}, {0, 1, 2, 3, 1}, ngraph::element::f32};
    Subgraph::BlockedShapeVector inputBlockedShapes{is1, is2, is3};
    Subgraph::BlockedShapeVector outputBlockedShapes{{{1, 4, 5, 3, 8}, {0, 1, 2, 3, 1}, ngraph::element::f32}};
    const auto &f = EltwiseFunctionLowered(inputShapes, inputBlockedShapes, outputBlockedShapes);
    input_blocked_shapes = f.getInputBlockedShapes();
    output_blocked_shapes = f.getOutputBlockedShapes();
    function = f.getOriginal();
    function_ref = f.getLowered();
    run();
}