// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <thread>

#include "pugixml.hpp"

#include <openvino/pass/serialize.hpp>
#include <ngraph/opsets/opset.hpp>

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace LayerTestsUtils {

void SnippetsTestsCommon::Validate() {
    IE_ASSERT(functionRefs == nullptr) << "Expected reference function in Validate";

    auto expectedOutputs = CalculateRefs();
    const auto &actualOutputs = GetOutputs();

    if (expectedOutputs.empty()) {
        return;
    }

    IE_ASSERT(actualOutputs.size() == expectedOutputs.size())
    << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

    Compare(expectedOutputs, actualOutputs);
}

}  // namespace LayerTestsUtils
