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
#include <snippets/op/serialization_node.hpp>

#include <openvino/core/graph_util.hpp>
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

LoweredExprIR::LoweredExprIR(const std::shared_ptr<ov::Model>& model, LoweringConfig config)
    : m_config{std::move(config)}, m_io_lowered_ops{} {
    for (const auto& n : get_ordered_ops(model)) {
        std::shared_ptr<LoweredExpr> expr;
        if (const auto& par = as_type_ptr<opset1::Parameter>(n)) {
            auto io_expr = std::make_shared<IOLoweredExpr>(n, model->get_parameter_index(par), IOLoweredExpr::io_type::INPUT);
            m_io_lowered_ops.push_back(io_expr);
            expr = io_expr;
        } else if (const auto& res = as_type_ptr<opset1::Result>(n)) {
            auto io_expr = std::make_shared<IOLoweredExpr>(n, model->get_result_index(res), IOLoweredExpr::io_type::OUTPUT);
            m_io_lowered_ops.push_back(io_expr);
            expr = io_expr;
        } else {
            expr = std::make_shared<LoweredExpr>(n);
        }
        m_lowered_ops.emplace_back(expr);
    }
    const auto&  commonParams = model->get_parameters();
    const auto& commonResults = model->get_results();
    // todo:
    //  Current default topological sorter visits Parameters and Results in reversed order
    //  compared to what's returned by get_parameters() or get_results()
    //  This causes additional problems during code generation because we generally expect that
    //  Parameters and Results appear in the lowered code in the same order as corresponding memory
    //  pointers (passed at runtime, this is get_parameters() or get_results() order).
    //  To resolve the conflict, we force the desired order of Parameters and Results in the lowered code.
    //  This approach works because registers for Parameters and Results are assigned manually, and
    //  from reg assignment standpoint it's not important in what order they were visited.
    //  But if we ever want to handle them as all the other nodes, we'll have to use custom topo sorter
    //  (even the same sorter, but with different starting node list)

    // todo: this could be optimized using emplace_front for lists (on reversed parameters)
//    for (const auto& par : commonParams) {
//        m_io_lowered_ops.emplace_back(std::make_shared<IOLoweredExpr>(par, model->get_parameter_index(par), IOLoweredExpr::io_type::INPUT));
//    }
//    for (const auto& res : model->get_results()) {
//        m_io_lowered_ops.emplace_back(std::make_shared<IOLoweredExpr>(res, model->get_result_index(res), IOLoweredExpr::io_type::OUTPUT));
//    }
//    auto params_end_it = std::next(m_io_lowered_ops.begin(), commonParams.size());
//    m_lowered_ops.insert(m_lowered_ops.begin(), m_io_lowered_ops.begin(), params_end_it);
//    m_lowered_ops.insert(m_lowered_ops.end(), params_end_it, m_io_lowered_ops.end());
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

ov::NodeVector LoweredExprIR::get_ordered_ops(const std::shared_ptr<ov::Model>& m) {
    if (!m->get_sinks().empty())
        throw ngraph_error("Linear IR is not supposed to work for model with sinks. Check your transformation pipeline.");

    // Note that an important difference between this impl and Model::get_ordered_ops is that Results and Parameters
    // are added in REVERSE order, so they will be visited in DIRECT order compared to get_parameters() and get_results()
    NodeVector nodes;
    const auto& results = m->get_results();
    std::copy(results.rbegin(), results.rend(), std::back_inserter(nodes));
    const auto& params = m->get_parameters();
    std::copy(params.rbegin(), params.rend(), std::back_inserter(nodes));

    return ov::topological_sort(nodes);
}

void LoweredExprIR::serialize(const std::string& xml, const std::string& bin) {
    auto first_node = std::make_shared<opset1::Parameter>(element::f32, Shape{});
    first_node->set_friendly_name("Start");
    first_node->get_rt_info()["execTimeMcs"] = 0;
    std::shared_ptr<Node> body_node = first_node;
    for (const auto& expr : m_lowered_ops) {
        body_node = std::make_shared<op::SerializationNode>(body_node, expr->get_node());
    }
    auto last_node = std::make_shared<opset1::Result>(body_node);
    last_node->set_friendly_name("End");
    const auto tmp_model = std::make_shared<ov::Model>(ResultVector {last_node},
                                                       ParameterVector {first_node},
                                                       "Lowered_IR_Serialization");
    ov::pass::Serialize(xml, bin).run_on_model(tmp_model);
}

LoweredExprIR::container LoweredExprIR::deep_copy_range(LoweredExprIR::container::const_iterator begin, LoweredExprIR::container::const_iterator end) {
    LoweredExprIR::container result;
    NodeVector original_nodes;
    for (auto it = begin; it != end; it++)
        original_nodes.push_back((*it)->get_node());
    NodeMap node_map;
    const NodeVector& new_nodes = ngraph::clone_nodes(original_nodes,  node_map);
    for (auto it = begin; it != end; it++) {
        // copy by value, so result shared_pointer point to new objects
        LoweredExpr new_expr = **it;
        new_expr.m_source_node = node_map[(*it)->get_node().get()];
        result.emplace_back(std::make_shared<LoweredExpr>(new_expr));
    }
    return result;
}

LoweredExprIR LoweredExprIR::deep_copy() const {
    LoweredExprIR result;
    auto& result_ops = result.m_lowered_ops;
    for (const auto& expr : deep_copy_range(m_lowered_ops.begin(), m_lowered_ops.end()))
        result_ops.emplace_back(expr);
    result.m_config = m_config;
    result.m_forcedIOShapes = m_forcedIOShapes;
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

LoweredExprIR::exprtPtr LoweredExprIR::get_expr_by_node(const std::shared_ptr<Node>& n) {
    auto found = m_node2expression_map.find(n);
    return found == m_node2expression_map.end() ? nullptr : found->second;
}

void LoweredExprIR::register_expression(const exprtPtr& expr) {
    const auto& node = expr->get_node();
    if (m_node2expression_map.count(node) != 0)
        throw ngraph_error("Duplicate node is detected in linear IR: " + std::string(node->get_friendly_name()));
    m_node2expression_map.insert({node, expr});
}

void LoweredExprIR::unregister_expression(const exprtPtr& expr) {
    const auto& node = expr->get_node();
    if (m_node2expression_map.count(node) != 1)
        throw ngraph_error("Attempt to erase non-existing expression from linear IR: " + std::string(node->get_friendly_name()));
    m_node2expression_map.erase(node);
}

LoweredExprIR::exprIt LoweredExprIR::insert(constExprIt pos, container::value_type&& value) {
    register_expression(value);
    return m_lowered_ops.insert(pos, value);
}

LoweredExprIR::exprIt LoweredExprIR::insert(constExprIt pos, const container::value_type& value) {
    register_expression(value);
    return m_lowered_ops.insert(pos, value);
}

LoweredExprIR::exprIt LoweredExprIR::insert(constExprIt pos, exprIt begin, exprIt end) {
    for (auto b = begin; b != end; b++)
        register_expression(*b);
    return m_lowered_ops.insert(pos, begin, end);
}

LoweredExprIR::exprIt LoweredExprIR::insert(constExprIt pos, constExprIt begin, constExprIt end) {
    for (auto b = begin; b != end; b++)
        register_expression(*b);
    return m_lowered_ops.insert(pos, begin, end);
}

LoweredExprIR::exprIt LoweredExprIR::insert(LoweredExprIR::constExprIt pos, const NodeVector& nodes) {
    container new_exprs;
    std::transform(nodes.begin(), nodes.end(), std::back_inserter(new_exprs),
                   [](const std::shared_ptr<Node>& n) {return std::make_shared<LoweredExpr>(n);});
    return insert(pos, new_exprs.begin(), new_exprs.end());
}

LoweredExprIR::exprIt LoweredExprIR::erase(LoweredExprIR::exprIt pos) {
    unregister_expression(*pos);
    return m_lowered_ops.erase(pos);
}

LoweredExprIR::exprIt LoweredExprIR::erase(LoweredExprIR::constExprIt pos) {
    unregister_expression(*pos);
    return m_lowered_ops.erase(pos);
}

}// namespace snippets
}// namespace ngraph
