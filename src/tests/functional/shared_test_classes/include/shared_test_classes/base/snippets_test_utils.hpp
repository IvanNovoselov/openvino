// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include "layer_test_utils.hpp"

namespace LayerTestsUtils {
class SnippetsTestsCommon : public LayerTestsCommon {
protected:
    void Validate() override;
};
}  // namespace LayerTestsUtils
