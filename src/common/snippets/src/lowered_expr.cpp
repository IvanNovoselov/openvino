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

LoweredExpr::LoweredExpr(const std::shared_ptr<Node>& n) : m_source_node{n}, m_reg_info{},  m_emitter{nullptr} {
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
IOLoweredExpr::IOLoweredExpr(const std::shared_ptr<Node>& n, int64_t index) : LoweredExpr(n), m_index(index) {
    if (ov::is_type<opset1::Parameter>(n))
        m_type = io_type::INPUT;
    else if (ov::is_type<opset1::Result>(n))
        m_type = io_type::OUTPUT;
    else
        throw ngraph_error("Can't derive io_type automatically for node " + n->get_friendly_name());
}

IOLoweredExpr::IOLoweredExpr(const std::shared_ptr<Node>& n, int64_t index, io_type type)
    : LoweredExpr(n), m_index(index), m_type(type) {
}

LoweredExprIR::LoweredExprIR(const std::shared_ptr<ov::Model>& model, const LoweringConfig config)
    : m_config{config} {
    for (const auto& n : model->get_ordered_ops()) {
        std::shared_ptr<LoweredExpr> expr;
        if (const auto& par = as_type_ptr<opset1::Parameter>(n)) {
            expr = std::make_shared<IOLoweredExpr>(n, model->get_parameter_index(par), IOLoweredExpr::io_type::INPUT);
        } else if (const auto& res = as_type_ptr<opset1::Result>(n)) {
            expr = std::make_shared<IOLoweredExpr>(n, model->get_result_index(res), IOLoweredExpr::io_type::OUTPUT);
        } else {
            expr = std::make_shared<LoweredExpr>(n);
        }
        m_lowered_ops.emplace_back(expr);
    }
    ParameterVector commonParams = model->get_parameters();
    // Note that topological sort parses node arguments in reversed order, but results are added  - in direct order
    // So ve need to pass the reversed results to LoopEnd to keep the original traversal order in topological sorter
    const auto& commonResults = model->get_results();
    const auto& body_rt_info = model->get_rt_info();
    const auto& plugin_shapes = body_rt_info.find("PluginShapesOverride");
    if (plugin_shapes == body_rt_info.end()) {
        throw ngraph_error("InsertLoops requires PluginShapesOverride rt_info field");
    } else {
        const auto& new_shapes = plugin_shapes->second.as<std::vector<std::vector<size_t>>>();
        if (new_shapes.size() != commonResults.size() + commonParams.size())
            throw ngraph_error("InsertLoops got invalid number of plugin-overriden shapes");
        for (int i = 0; i < commonParams.size(); i++)
            m_forcedIOShapes.emplace_back(new_shapes[i]);
        // reverse overriden_shapes for results since commonResults are reversed with respect to model->get_parameters()
        for (int i = 0; i < commonResults.size(); i++)
            m_forcedIOShapes.emplace_back(new_shapes[commonParams.size() + i]);
    }
}

LoweredExprIR LoweredExprIR::deep_copy() const {
    LoweredExprIR result;
    NodeVector original_nodes;
    for (const auto& expr : m_lowered_ops)
        original_nodes.push_back(expr->get_node());
    NodeMap node_map;
    const NodeVector& new_nodes = ngraph::clone_nodes(original_nodes,  node_map);
    auto& new_ops = result.get_ops();
    for (const auto& expr : m_lowered_ops) {
        LoweredExpr new_expr = *expr;
        new_expr.m_source_node = node_map[expr->get_node().get()];
        new_ops.emplace_back(std::make_shared<LoweredExpr>(new_expr));
    }
    return result;
}

void LoweredExprIR::debug_print() const {
    auto print_rinfo = [](RegInfo rinfo) {
        std::cerr << "   ";
        for (auto i : rinfo.first)
            std::cerr << i << " ";
        std::cerr << " => ";
        for (auto i : rinfo.second)
            std::cerr << i << " ";
        std::cerr << "\n";
    };
    for (const auto& expr : m_lowered_ops) {
        std::cerr << expr->get_node()->get_friendly_name() << ": \n";
        const auto& rinfo = expr->get_reg_info();
        if (!rinfo.first.empty() || !rinfo.second.empty())
            print_rinfo(expr->get_reg_info());
    }
}

void LoweredExprIR::init_emitters(const std::shared_ptr<TargetMachine>& target) {
    for (auto& expr : m_lowered_ops) {
        if (!expr->get_emitter())
            expr->init_emitter(target);
    }
}

}// namespace snippets
}// namespace ngraph
