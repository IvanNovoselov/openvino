// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/openvino.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/convolution_params.hpp"
#include "snippets/op/subgraph.hpp"

using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

class SubgraphSnippetSerializationTest : public ::testing::Test, public CPUTestsBase {};

TEST_F(SubgraphSnippetSerializationTest, SerializeSubgraph) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto model = ([] () -> std::shared_ptr<ov::Model> {
        auto shape = ov::Shape({2, 2});
        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto ininput0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto ininput1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto add = std::make_shared<ov::op::v1::Add>(ininput0, ininput1);
        auto subgraph_body = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{ininput0, ininput1});
        auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(ov::NodeVector{input0, input1}, ov::clone_model(*subgraph_body.get()));
        return std::make_shared<ov::Model>(ov::NodeVector{subgraph}, ov::ParameterVector{input0, input1});
    })();
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    std::stringstream stream;
    compiled_model.export_model(stream);
    ov::CompiledModel imported_compiled_model = core.import_model(stream, "CPU");
    auto compiled_model_runtime = ov::clone_model(*compiled_model.get_runtime_model());
    auto imported_compiled_model_runtime = ov::clone_model(*imported_compiled_model.get_runtime_model());
    const auto fc = FunctionsComparator::with_default()
                                .enable(FunctionsComparator::CONST_VALUES)
                                .enable(FunctionsComparator::ATTRIBUTES);
    const auto results = fc.compare(compiled_model_runtime, imported_compiled_model_runtime);

    ASSERT_TRUE(results.valid) << results.message;
}

TEST_F(SubgraphSnippetSerializationTest, SerializeSubgraphBody) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto model_add = ([] () -> std::shared_ptr<ov::Model> {
        auto shape = ov::Shape({2, 2});
        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto ininput0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto ininput1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto add = std::make_shared<ov::op::v1::Add>(ininput0, ininput1);
        auto subgraph_body = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{ininput0, ininput1});
        auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(ov::NodeVector{input0, input1}, ov::clone_model(*subgraph_body.get()));
        return std::make_shared<ov::Model>(ov::NodeVector{subgraph}, ov::ParameterVector{input0, input1});
    })();

    auto model_divide = ([] () -> std::shared_ptr<ov::Model> {
        auto shape = ov::Shape({2, 2});
        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto ininput0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto ininput1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto add = std::make_shared<ov::op::v1::Divide>(ininput0, ininput1);
        auto subgraph_body = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{ininput0, ininput1});
        auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(ov::NodeVector{input0, input1}, ov::clone_model(*subgraph_body.get()));
        return std::make_shared<ov::Model>(ov::NodeVector{subgraph}, ov::ParameterVector{input0, input1});
    })();
    ov::Core core;
    ov::CompiledModel compiled_model_add = core.compile_model(model_add, "CPU");
    std::stringstream stream;
    compiled_model_add.export_model(stream);
    ov::CompiledModel imported_compiled_model_add = core.import_model(stream, "CPU");
    auto compiled_model_runtime_add = ov::clone_model(*compiled_model_add.get_runtime_model());
    auto imported_compiled_model_runtime_add = ov::clone_model(*imported_compiled_model_add.get_runtime_model());

    {
        std::ofstream out_stream;
        // this model can be read with benchmark_app afterwards
        out_stream.open("compiled_model_add.blob");
        compiled_model_add.export_model(out_stream);
        out_stream.close();
        ov::pass::Serialize("compiled_model_add.xml", "compiled_model_add.bin").run_on_model(compiled_model_runtime_add);
    }
    {
        std::ofstream out_stream;
        out_stream.open("imported_model_add.blob");
        imported_compiled_model_add.export_model(out_stream);
        out_stream.close();
        ov::pass::Serialize("imported_model_add.xml", "imported_model_add.bin").run_on_model(imported_compiled_model_runtime_add);
    }
    const auto fc = FunctionsComparator::with_default()
            .enable(FunctionsComparator::CONST_VALUES)
            .enable(FunctionsComparator::ATTRIBUTES);
    const auto results = fc.compare(compiled_model_runtime_add, imported_compiled_model_runtime_add);

    ASSERT_TRUE(results.valid) << results.message;

    ov::CompiledModel compiled_model_divide = core.compile_model(model_divide, "CPU");
    auto compiled_model_runtime_divide = ov::clone_model(*compiled_model_divide.get_runtime_model());
    const auto results_false = fc.compare(compiled_model_runtime_divide, imported_compiled_model_runtime_add);
    ASSERT_FALSE(results_false.valid) << results.message;
    }

TEST_F(SubgraphSnippetSerializationTest, SerializeSubgraphWithScalarConst) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto model = ([] () -> std::shared_ptr<ov::Model> {
        auto shape = ov::Shape({1});
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto internal_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, shape, 2);
        auto internal_constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, shape, 2);
        auto add = std::make_shared<ov::op::v1::Add>(input, constant);
        auto internal_add = std::make_shared<ov::op::v1::Add>(internal_input, internal_constant);
        auto subgraph_body = std::make_shared<ov::Model>(ov::NodeVector{internal_add}, ov::ParameterVector{internal_input});
        auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(ov::NodeVector{add}, ov::clone_model(*subgraph_body.get()));
        return std::make_shared<ov::Model>(ov::NodeVector{subgraph}, ov::ParameterVector{input});
    })();
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    std::stringstream stream;
    compiled_model.export_model(stream);
    ov::CompiledModel imported_compiled_model = core.import_model(stream, "CPU");
    auto compiled_model_runtime = ov::clone_model(*compiled_model.get_runtime_model());
    auto imported_compiled_model_runtime = ov::clone_model(*imported_compiled_model.get_runtime_model());
    const auto fc = FunctionsComparator::with_default()
                                .enable(FunctionsComparator::CONST_VALUES)
                                .enable(FunctionsComparator::ATTRIBUTES);
    const auto results = fc.compare(compiled_model_runtime, imported_compiled_model_runtime);

    ASSERT_TRUE(results.valid) << results.message;
}
} // namespace SubgraphTestsDefinitions
