// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "modifiers.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

class Add : public TensorProcessingPrimitive, public ov::op::v1::Add {
};

class Subtract : public TensorProcessingPrimitive, public ov::op::v1::Subtract {
};

class Multiply : public TensorProcessingPrimitive, public ov::op::v1::Multiply {
};

class Divide : public TensorProcessingPrimitive, public ov::op::v1::Divide {
};

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
