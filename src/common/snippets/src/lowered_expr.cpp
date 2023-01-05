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

LoweredExpr::LoweredExpr(const std::shared_ptr<Node>& n) : m_source_node{n}, m_reg_info{getRegisters(n)},  m_emitter{nullptr} {
}

std::shared_ptr<Emitter> LoweredExpr::get_emitter() const {
//    if (!m_emitter)
//        throw ngraph_error("An attempt to get uninitialized emitter. You need to call init_emitter() first");
    return  m_emitter;
}

void LoweredExpr::init_emitter(const std::shared_ptr<const TargetMachine>& target) {
    m_emitter = target->get(m_source_node->get_type_info())(m_source_node);
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

LoweredExprIR::LoweredExprIR(const std::vector<std::shared_ptr<ov::Node>>& ops) {
    for (const auto& n : ops)
        m_lowered_ops.emplace_back(n);
}
void LoweredExprIR::init_emitters(const std::shared_ptr<TargetMachine>& target) {
    for (auto& expr : m_lowered_ops) {
        if (!expr.get_emitter())
            expr.init_emitter(target);
    }
}

}// namespace snippets
}// namespace ngraph
