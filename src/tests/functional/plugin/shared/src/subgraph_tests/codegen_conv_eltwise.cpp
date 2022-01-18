
// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "subgraph_tests/codegen_conv_eltwise.hpp"
#include "subgraph_convolution.hpp"


namespace LayerTestsDefinitions {

    std::string CodegenConvEltwise::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::multiInputParams> obj) {
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShapes0, inputShapes1;
        std::string targetDevice;
        std::tie(netPrecision, inputShapes0, inputShapes1, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS[0]=" << CommonTestUtils::vec2str(inputShapes0) << "_";
        result << "IS[1]=" << CommonTestUtils::vec2str(inputShapes1) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    // the simplest possible eltwise operation with streaming access to the data
    void CodegenConvEltwise::SetUp() {
        std::vector<size_t> inputShape0, inputShape1;
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, inputShape0, inputShape1, targetDevice) = this->GetParam();

        const auto f  = ngraph::builder::subgraph::ConvMulActivation({inputShape0, inputShape1});
        function = f.getOriginal();
    }

TEST_P(CodegenConvEltwise, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
