// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,  // Network Precision
        InferenceEngine::SizeVector, // Input Shape #0
        InferenceEngine::SizeVector, // Input Shape #1
        std::shared_ptr<ov::Node>,   // The first binary eltwise op after the Convolution
        std::string                  // Target Device
> multiInputParams;

class CodegenConvEltwise : public testing::WithParamInterface<LayerTestsDefinitions::multiInputParams>,
virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::multiInputParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
