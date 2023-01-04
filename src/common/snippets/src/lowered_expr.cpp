// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered_expr.hpp"
#include "snippets/pass/assign_registers.hpp"
#include "snippets/pass/vector_to_scalar.hpp"
#include "snippets/pass/insert_load_store.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/op/kernel.hpp"
#include <snippets/itt.hpp>

#include <ngraph/pass/manager.hpp>
#include <openvino/core/type.hpp>

namespace ngraph {
namespace snippets {

LoweredExpr::LoweredExpr(const std::shared_ptr<Node>& n, const std::shared_ptr<const TargetMachine> &target) :
            m_source_node{n}, m_reg_info{getRegisters(n)},  m_emitter{target->get(n->get_type_info())(n)} {
}

ngraph::snippets::RegInfo LoweredExpr::getRegisters(const std::shared_ptr<const Node>& n) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::getRegisters")

    // ToDo: change to reg_t
    std::vector<size_t> rin, rout;

    for (const auto& output : n->outputs()) {
        const auto& rt = output.get_tensor_ptr()->get_rt_info();
        auto it_rt = rt.find("reginfo");
        if (it_rt != rt.end())
            rout.push_back(it_rt->second.as<size_t>());
    }

    for (const auto& input : n->inputs()) {
        auto rt = input.get_source_output().get_tensor_ptr()->get_rt_info();
        auto it_rt = rt.find("reginfo");
        if (it_rt != rt.end())
            rin.push_back(it_rt->second.as<size_t>());
    }

    return std::make_pair(rin, rout);
}

LoweredExprIR::LoweredExprIR(const std::vector<std::shared_ptr<ov::Node>>& ops, const std::shared_ptr<TargetMachine>& target) {
    for (const auto& n : ops)
        m_lowered_ops.emplace_back(n, target);
}

//LoweredExprIR::LoweredExprIR(std::vector<std::shared_ptr<ov::Node>> ops, std::shared_ptr<TargetMachine> target) {
//    for (const auto& n : ops)
//        m_lowered_ops.emplace_back(n, target);
//}

}// namespace snippets
}// namespace ngraph
