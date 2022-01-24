// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <subgraph_lowered.hpp>
#include <snippets_helpers.hpp>
#include "snippets/op/subgraph.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class DummyEmitter : public ngraph::snippets::Emitter {
public:
    // Here I pass Add to Emitter, but could be any other op, since it's ignored anyway.
    DummyEmitter() : ngraph::snippets::Emitter(std::make_shared<ov::op::v1::Add>()) {}
    void emit_code(const std::vector<size_t>&,
                   const std::vector<size_t>&,
                   const std::vector<size_t>&,
                   const std::vector<size_t>&) const override {}
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

class SnippetsLoweringTests : public TransformationTestsF {
private:
    static void serialize(const std::string& name, const std::shared_ptr<Model>& m) {
        ov::pass::Serialize(name + ".xml", name + ".bin").run_on_model(m);
    }

protected:
    void prepare() {
        ASSERT_TRUE(function);
        serialize("before", function);
        // Check that the function is fully tokenizable and obtain subgraph
        tokenize(function);
        getSubgraph(function);
        serialize("tokenized", subgraph->get_body());
    }
    Shape canonicalize(Subgraph::BlockedShapeVector& input_blocked_shapes, Subgraph::BlockedShapeVector& output_blocked_shapes) {
        return subgraph->canonicalize(output_blocked_shapes, input_blocked_shapes);
    }
    void lower() {
        subgraph->set_generator(std::make_shared<DummyGenerator>());
        subgraph->generate();
        function = subgraph->get_body();
        serialize("lowered", function);
    }

private:
    std::shared_ptr<Subgraph> subgraph;
    static void tokenize(std::shared_ptr<Model>& f) {
        ngraph::pass::Manager m;
        m.register_pass<EnumerateNodes>();
        m.register_pass<TokenizeSnippets>();
        m.run_passes(f);
    }

    void getSubgraph(std::shared_ptr<Model>& f) {
        for (const auto &op : f->get_ops()) {
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
    }
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
