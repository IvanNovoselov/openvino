// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "eltwise.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

bool BinaryEltwiseTPP::is_supported(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v1::Add>(node) ||
           ov::is_type<ov::op::v1::Add>(node);
}

Add::Add(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
: BinaryEltwiseTPP(), ov::op::v1::Add(arg0, arg1, auto_broadcast) {
}

bool Add::visit_attributes(AttributeVisitor& visitor) {
    // todo: this is for debug purposes. remove before merge
    std::string tmp = "TPP";
    visitor.on_attribute("type", tmp);
    return MemoryAccess::visit_attributes(visitor);
}

Subtract::Subtract(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
        : BinaryEltwiseTPP(), ov::op::v1::Subtract(arg0, arg1, auto_broadcast) {
}

Multiply::Multiply(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
        : BinaryEltwiseTPP(), ov::op::v1::Multiply(arg0, arg1, auto_broadcast) {
}


Divide::Divide(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
        : BinaryEltwiseTPP(), ov::op::v1::Divide(arg0, arg1, auto_broadcast) {
}

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
