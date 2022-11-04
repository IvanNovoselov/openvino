// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/transpose.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Transpose, Transpose,
                     ::testing::Combine(
                             ::testing::Values(ov::PartialShape {2, 2, 3, 16}),
                             ::testing::Values(std::vector<int> {0, 2,  3, 1}),
                             ::testing::Values(2), // Sinh + Transpose
                             ::testing::Values(1), // Tokenized Transpose
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                     Transpose::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov