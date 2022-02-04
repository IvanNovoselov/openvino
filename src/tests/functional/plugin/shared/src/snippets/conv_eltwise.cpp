
// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "snippets/conv_eltwise.hpp"
#include "subgraph_customizable.hpp"


namespace LayerTestsDefinitions {

    std::string CodegenConvEltwise::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::multiInputParams> obj) {
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShape0, inputShape1;
        std::shared_ptr<ov::Node> binaryEltwise;
        std::string targetDevice;
        std::tie(netPrecision, inputShape0, inputShape1, binaryEltwise, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS[0]=" << CommonTestUtils::vec2str(inputShape0) << "_";
        result << "IS[1]=" << CommonTestUtils::vec2str(inputShape1) << "_";
        result << "Op=" << binaryEltwise->get_type_name() << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    // the simplest possible eltwise operation with streaming access to the data
    void CodegenConvEltwise::SetUp() {
        std::vector<size_t> inputShape0, inputShape1;
        InferenceEngine::Precision netPrecision;
        std::shared_ptr<ov::Node> binaryEltwise;
        std::tie(netPrecision, inputShape0, inputShape1, binaryEltwise, targetDevice) = this->GetParam();
        std::vector<std::shared_ptr<Node>> eltwiseOps {binaryEltwise,
                                                       std::make_shared<ov::op::v0::Tanh>(),
                                                       std::make_shared<ov::op::v0::Sqrt>()};
        const auto f  = ngraph::builder::subgraph::ConvMulActivation({inputShape0, inputShape1}, eltwiseOps);
        function = f.getOriginal();
    }

TEST_P(CodegenConvEltwise, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
