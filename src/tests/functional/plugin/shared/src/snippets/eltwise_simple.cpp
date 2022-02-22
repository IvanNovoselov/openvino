
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "snippets/eltwise_simple.hpp"
#include "snippets/op/subgraph.hpp"
#include "subgraph_simple.hpp"

namespace LayerTestsDefinitions {

    std::string CodegenAdd::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::multiInputParams> obj) {
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShapes0, inputShapes1, newInputShapes;
        std::string targetDevice;
        size_t num_nodes, num_subgraphs;
        std::tie(netPrecision, inputShapes0, inputShapes1, num_nodes, num_subgraphs, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS[0]=" << CommonTestUtils::vec2str(inputShapes0) << "_";
        result << "IS[1]=" << CommonTestUtils::vec2str(inputShapes1) << "_";
        result << "#N=" << num_nodes << "_";
        result << "#S=" << num_subgraphs << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void CodegenAdd::SetUp() {
        std::vector<size_t> inputShape0, inputShape1;
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, inputShape0, inputShape1, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
        init_input_shapes({{{}, {inputShape0, }}, {{}, {inputShape1, }}});

        auto f = ngraph::builder::subgraph::AddConvertFunction({inputShape0, inputShape1});
        function = f.getOriginal();
//        const auto ref_function = f.getReference();
    }

    void CodegenAdd::validateNumSubgraphs() {
        const auto& compiled_model = compiledModel.get_runtime_model();
        {
            auto m = ov::clone_model(*compiled_model);
            ov::pass::Serialize("compiled.xml", "compiled.bin").run_on_model(m);
        }
        size_t num_subgraphs = 0;
        size_t num_nodes = 0;
        for (const auto &op : compiled_model->get_ops()) {
            if (ov::is_type<ov::op::v0::Parameter>(op) ||
                ov::is_type<ov::op::v0::Constant>(op) ||
                ov::is_type<ov::op::v0::Result>(op)) {
                num_nodes++;
            } else {
                auto &rt = op->get_rt_info();
                const auto rinfo = rt.find("layerType");
                ASSERT_NE(rinfo, rt.end()) << "Failed to find layerType in " << op->get_friendly_name()
                                               << "rt_info.";
                const std::string layerType = rinfo->second.as<std::string>();
                num_subgraphs += layerType == "Subgraph";
            }
        }
        ASSERT_EQ(ref_num_nodes, num_nodes)
                                    << "Compiled model contains invalid number of nodes.";
        ASSERT_EQ(ref_num_subgraphs, num_subgraphs)
        << "Compiled model contains invalid number of subgraphs.";
    }

TEST_P(CodegenAdd, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
};

}  // namespace LayerTestsDefinitions
