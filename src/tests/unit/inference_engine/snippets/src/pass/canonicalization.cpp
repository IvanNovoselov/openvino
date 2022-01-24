// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/canonicalization.hpp"
using namespace ngraph::builder::subgraph;

element::Type_t prec = element::f32;
std::tuple<Shape, Subgraph::BlockedShape> blockedInput0 {{1, 64, 2, 5}, {{1, 4, 2, 5, 16}, {0, 1, 2, 3, 1}, prec}};
Subgraph::BlockedShape output {{1, 4, 2, 5, 16}, {0, 1, 2, 3, 1}, prec};
Shape canonical_shape {1, 4, 2, 5, 16};

std::vector<std::tuple<Shape, Subgraph::BlockedShape>> blockedInput1 {{{1, 1, 2, 5}, {{1, 1, 2, 5, 1}, {0, 1, 2, 3, 1}, prec}},
                                                                      {{1, 1, 2, 1}, {{1, 1, 2, 1, 1}, {0, 1, 2, 3, 1}, prec}},
                                                                      {{1, 64, 1, 1}, {{1, 4, 1, 1, 16}, {0, 1, 2, 3, 1}, prec}}};
INSTANTIATE_TEST_SUITE_P(BroadcastBlockedBlocked, SnippetsCanonicalizationTests2,
                         ::testing::Combine(
                                 ::testing::Values(blockedInput0),
                                 ::testing::ValuesIn(blockedInput1),
                                 ::testing::Values(output),
                                 ::testing::Values(canonical_shape)),
                         SnippetsCanonicalizationTests2::getTestCaseName);

std::vector<std::tuple<Shape, Subgraph::BlockedShape>> planarInput1 {{{1, 1, 2, 5}, {{1, 2, 5}, {0, 1, 2}, prec}},
                                                                     {{1, 1, 2, 5}, {{2, 5}, {0, 1}, prec}},
                                                                     {{1, 2, 5}, {{2, 5}, {0, 1}, prec}},
                                                                     {{2, 5}, {{2, 5}, {0, 1}, prec}},
                                                                     {{5}, {{5}, {0}, prec}}};
INSTANTIATE_TEST_SUITE_P(BroadcastBlockedPlanar, SnippetsCanonicalizationTests2,
                         ::testing::Combine(
                                 ::testing::Values(blockedInput0),
                                 ::testing::ValuesIn(planarInput1),
                                 ::testing::Values(output),
                                 ::testing::Values(canonical_shape)),
                         SnippetsCanonicalizationTests2::getTestCaseName);