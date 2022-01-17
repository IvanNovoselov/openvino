// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
/// Simple Eltwise graph fully convertible to Subgraph.
/// Tokenized simply by attaching eltwises.
// in1   in2
//    Add
//   /   Subtract
//  Multiply
//   Result
class EltwiseFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal();
    static std::shared_ptr<ov::Model> getReference();
};
/// MatMul with two eltwise branches joined with Add just before the Result.
/// Tokenized by attaching eltwises to separate subgraphs, and then joining them together.
//                   in1   in2
//                     MatMul
//  [Eltwise sequence 1]   [Eltwise sequence 2]
//                      Add
//                     Result
class MatMulEltwiseBranchesFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal();
    static std::shared_ptr<ov::Model> getReference();
};
/// Add with HSwish and Log  joined Multiply.
/// Log is not tokenizable, so two Subgraphs are created to avoid loop introduction: Add+HSwish and Multiply.
//     in1   in2
//        Add
//  HSwish   Log
//      Multiply
//       Result
class EltwiseLogLoop {
public:
    static std::shared_ptr<ov::Model> getOriginal();
    static std::shared_ptr<ov::Model> getReference();
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
