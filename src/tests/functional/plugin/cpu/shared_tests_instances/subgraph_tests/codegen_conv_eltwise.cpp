
// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/codegen_conv_eltwise.hpp"

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32
    };
    InferenceEngine::SizeVector convInputShape {1, 2, 16, 16};
    std::vector<InferenceEngine::SizeVector> mulInputShape {{1, 2, 16, 1}, {1, 2, 1, 1}, {1, 2, 1, 16}, {1, 2, 16, 16}};
    INSTANTIATE_TEST_SUITE_P(smoke_SnippetsSubgraph, CodegenConvEltwise,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(convInputShape),
            ::testing::ValuesIn(mulInputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            CodegenConvEltwise::getTestCaseName);
}  // namespace