// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include <openvino/cc/selective_build.h>
#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/pass/assign_registers_new.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>

#include <iterator>

bool ngraph::snippets::pass::AssignRegistersNew::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(AssignRegistersNew);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::AssignRegistersNew")
    using Reg = size_t;
    using tensor =std::shared_ptr<descriptor::Tensor>;
    auto ops = f->get_ordered_ops();
    // Note that currently there are 3 types of ops:
    //  * gpr->gpr: (Parameter, Result, TileBegin, TileEnd) will also be Buffer?
    //  * gpr->vec: or vec->gpr Load/LoadConvert, Store/StoreConvert, BroadcastLoad etc.
    //  * vec->vec: all other "normal" operations that perform calculations on vector registers: Add, BroadcastMove, Power, etc.
    enum op_reg_type {gpr2gpr, gpr2vec, vec2gpr, vec2vec};

    auto get_op_reg_type = [](const std::shared_ptr<Node>& op) {
        if (std::dynamic_pointer_cast<opset1::Parameter>(op) ||
                std::dynamic_pointer_cast<opset1::Result>(op) ||
                std::dynamic_pointer_cast<op::TileBegin>(op) ||
                std::dynamic_pointer_cast<op::TileEnd>(op))
            return gpr2gpr;
        else if (std::dynamic_pointer_cast<snippets::op::Load>(op))
            return gpr2vec;
        else if (std::dynamic_pointer_cast<snippets::op::Store>(op))
            return vec2gpr;
        else
            return vec2vec;

    };
    decltype(ops) ops_vec, ops_gpr, ops_mixed;
    for (const auto& op : ops) {
        switch (get_op_reg_type(op)) {
            case vec2vec: ops_vec.push_back(op); break;
            case gpr2gpr: ops_gpr.push_back(op); break;
            case gpr2vec:
            case vec2gpr: ops_mixed.push_back(op); break;
        }
    }
    std::vector<std::pair<op_reg_type, std::shared_ptr<Node>>> typed_ops;
    for (const auto& op : ops)
        typed_ops.emplace_back(std::make_pair(get_op_reg_type(op), op));
    // todo: lets create a statements map by register type stmts['vec2vec'],stmts['gpr2gpr'], stmts['mixed']
    // enumerate all used tensors
    size_t counter_vec = 0;
    size_t counter_gpr = 0;
    std::map<tensor, Reg> regs_vec, regs_gpr;
    auto enumerate_out_tensors = [] (const std::shared_ptr<ov::Node>& op,
                                     decltype(regs_vec)& reg_map,
                           size_t& counter) {
        for (const auto& output : op->outputs()) {
            reg_map[output.get_tensor_ptr()] = counter++;
        }
    };
    for (const auto& t_op : typed_ops) {
        switch (t_op.first) {
            case vec2vec:
            case gpr2vec:
                enumerate_out_tensors(t_op.second, regs_vec, counter_vec);
                break;
            case gpr2gpr:
            case vec2gpr:
                enumerate_out_tensors(t_op.second, regs_gpr, counter_gpr);
                break;
        }
    }
    // todo: make one for gpr and one for vector
    std::vector<std::set<Reg>> used_gpr(ops.size(), std::set<Reg>()); // used = used as an input
    std::vector<std::set<Reg>> defined_gpr(ops.size(), std::set<Reg>()); // defined = used as output
    std::vector<std::set<Reg>> used_vec(ops.size(), std::set<Reg>()); // used = used as an input
    std::vector<std::set<Reg>> defined_vec(ops.size(), std::set<Reg>()); // defined = used as output

    auto tensor2reg = [] (const std::vector<tensor>& tensors, const std::map<tensor, Reg>& reg_map) {
        std::set<Reg> result;
        for (const auto& t : tensors)
            result.insert(reg_map.at(t));
        return result;
    };
    for (int i = 0; i < typed_ops.size(); i++) {
        const auto& t_op = typed_ops[i];
        std::vector<tensor> used_tensors, defined_tensors;
        for (const auto& in : t_op.second->inputs())
            used_tensors.push_back(in.get_tensor_ptr());
        for (const auto& out : t_op.second->outputs())
            defined_tensors.push_back(out.get_tensor_ptr());
        switch (t_op.first) {
            case vec2vec:
                used_vec[i] = tensor2reg(used_tensors, regs_vec);
                defined_vec[i] = tensor2reg(defined_tensors, regs_vec);
                break;
            case gpr2gpr:
                used_gpr[i] = tensor2reg(used_tensors, regs_gpr);
                defined_gpr[i] = tensor2reg(defined_tensors, regs_gpr);
                break;
            case gpr2vec:
                used_gpr[i] = tensor2reg(used_tensors, regs_gpr);
                defined_vec[i] = tensor2reg(defined_tensors, regs_vec);
                break;
            case vec2gpr:
                used_vec[i] = tensor2reg(used_tensors, regs_vec);
                defined_gpr[i] = tensor2reg(defined_tensors, regs_gpr);
                break;
        }
    }

    // define life intervals
    // liveOut[i] - regs that are live on exit from i-th (topologically ordered) operation
    // liveIn[i] - regs that are live on entering the i-th (topologically ordered) operation
    std::vector<std::set<Reg>> life_in_vec(std::move(used_vec));
    std::vector<std::set<Reg>> life_out_vec(typed_ops.size(), std::set<Reg>());
    std::vector<std::set<Reg>> life_in_gpr(std::move(used_gpr));
    std::vector<std::set<Reg>> life_out_gpr(typed_ops.size(), std::set<Reg>());

    // todo: this part if O(N*N), so it's slow for large subgraphs. Can we simplify it?
    for (size_t i = 0; i < typed_ops.size(); i++) {
        for (size_t n = 0; n < typed_ops.size(); n++) {
            // Regs that are live on entering the operation = regs used by the op + (all other regs alive - regs defined by the op)
            // copy regs from lifeOut to lifeIn while ignoring regs in def
            std::set_difference(life_out_gpr[n].begin(), life_out_gpr[n].end(),
                                defined_gpr[n].begin(), defined_gpr[n].end(),
                                std::inserter(life_in_gpr[n], life_in_gpr[n].begin()));
            std::set_difference(life_out_vec[n].begin(), life_out_vec[n].end(),
                                defined_vec[n].begin(), defined_vec[n].end(),
                                std::inserter(life_in_vec[n], life_in_vec[n].begin()));
            //lifeIn[n].insert(used[n].begin(), used[n].end());
        }
        for (size_t n = 0; n < typed_ops.size(); n++) {
            auto op = typed_ops[n].second;
            for (const auto& out : op->outputs()) {
                for (const auto& port : out.get_target_inputs()) {
                    auto k = std::find(ops.begin(), ops.end(), port.get_node()->shared_from_this()) - ops.begin();
                    if (k == ops.size())
                        throw ngraph_error("assign registers can't find target op in the body");
                    switch (typed_ops[k].first) {
                        case vec2vec:
                        case vec2gpr:
                            life_out_vec[n].insert(life_in_vec[k].begin(), life_in_vec[k].end());
                            break;
                        case gpr2gpr:
                        case gpr2vec:
                            life_out_gpr[n].insert(life_in_gpr[k].begin(), life_in_gpr[k].end());
                            break;
                    }
                }
            }
        }
    }

    struct by_starting {
        auto operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const -> bool {
            return lhs.first < rhs.first|| (lhs.first == rhs.first && lhs.second < rhs.second);
        }
    };

    struct by_ending {
        auto operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const -> bool {
            return lhs.second < rhs.second || (lhs.second == rhs.second && lhs.first < rhs.first);
        }
    };

    std::set<std::pair<int, int>, by_starting> live_intervals_vec, live_intervals_gpr;

    std::reverse(life_in_vec.begin(), life_in_vec.end());
    std::reverse(life_in_gpr.begin(), life_in_gpr.end());
    auto find_last_use = [](decltype(life_in_gpr) life_in, int i) -> int {
        int ln = static_cast<int>(life_in.size()) - 1;
        for (auto& x : life_in) {
            if (x.find(i) != x.end()) {
                return ln;
            }
            ln--;
        }
        return i;
    };

    for (size_t i = 0; i < ops.size(); i++) {
        live_intervals_vec.insert(std::make_pair(static_cast<int>(i), find_last_use(life_in_vec, static_cast<int>(i))));
        live_intervals_gpr.insert(std::make_pair(static_cast<int>(i), find_last_use(life_in_gpr, static_cast<int>(i))));
    }
    {
//        std::cerr << "Live intervals: " << "\n";
//        for (int i = 0; i < ops.size(); i++) {
//            auto v = live_intervals_vec[i];
//            auto g = live_intervals_gpr[i];
//            std::cerr << i << " | " << ops[i]->get_friendly_name() << " V: " << p.first << " : " << p.second << "\n";
//        }
    }
    auto linescan_assign_registers = [](const decltype(live_intervals_vec)& live_intervals) {
        // http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf
        std::multiset<std::pair<int, int>, by_ending> active;
        std::map<Reg, Reg> register_map;
        std::stack<Reg> bank;
        for (int i = 0; i < 16; i++) bank.push(16 - 1 - i);

        for (auto interval : live_intervals) {
            // check expired
            while (!active.empty()) {
                auto x = *active.begin();
                if (x.second >= interval.first) {
                    break;
                }
                active.erase(x);
                bank.push(register_map[x.first]);
            }
            // allocate
            if (active.size() == 16) {
                throw ngraph::ngraph_error("can't allocate registers for a snippet ");
            } else {
                register_map[interval.first] = bank.top();
                bank.pop();
                active.insert(interval);
            }
        }
        return register_map;
    };
    auto register_map_vec = linescan_assign_registers(live_intervals_vec);
    auto register_map_gpr = linescan_assign_registers(live_intervals_gpr);

    std::map<std::shared_ptr<descriptor::Tensor>, Reg> physical_regs_gpr, physical_regs_vec;

    for (const auto& reg : regs_vec)
        physical_regs_vec[reg.first] = register_map_vec[reg.second];
    for (const auto& reg : regs_gpr)
        physical_regs_gpr[reg.first] = register_map_gpr[reg.second];

    {
//        int i = 0;
//        std::cerr << "Physical regs:\n";
//        for (const auto& s : stmts) {
//            std::cerr << i << " | " << stmts[i]->get_friendly_name() << " | ";
//            for (const auto& output : s->outputs()) {
//                std::cerr << physical_regs[ output.get_tensor_ptr()];
//            }
//            i++;
//            std::cerr << "\n";
//        }
    }

    for (const auto& t_op : typed_ops) {
        std::vector<tensor> out_tensors;
        for (const auto& out : t_op.second->outputs())
            out_tensors.push_back(out.get_tensor_ptr());

        switch (t_op.first) {
            case vec2vec:
            case gpr2vec:
                for (auto& t : out_tensors) {
                    auto& rt = t->get_rt_info();
                    rt["reginfo_new"] = std::vector<size_t>{physical_regs_vec[t]};
                }
                break;
            case gpr2gpr:
            case vec2gpr:
                for (auto& t : out_tensors) {
                    auto& rt = t->get_rt_info();
                    rt["reginfo_new"] = std::vector<size_t>{physical_regs_gpr[t]};
                }
                break;
        }
    }

    /*
    const auto num_parameters = f->get_parameters().size();
    for (const auto& n : f->get_ordered_ops()) {
        // The main idea here is that each operation stores its output regs in rt["reginfo"]. Input and output regs are
        // then derived by parsing node's and parent's rt["reginfo"], look into ngraph::snippets::getRegisters for details.
        // Note also that Parameter and Result store general-purpose register index, because they work with memory
        // (memory pointer is stored in gpr). All other "regular" ops store vector regs indexes, since calculations are
        // performed on registers.
        if (is_type<ov::op::v0::Result>(n)) {
            continue;
        } else if (const auto& param = ov::as_type_ptr<ov::op::v0::Parameter>(n)) {
            auto& rt  =  n->get_output_tensor(0).get_rt_info();
            rt["reginfo"] = std::vector<size_t>{static_cast<size_t>(f->get_parameter_index(param))};
        } else if (const auto& store = ov::as_type_ptr<ngraph::snippets::op::Store>(n)) {
            auto& rt  = n->get_output_tensor(0).get_rt_info();
            rt["reginfo"] = std::vector<size_t>{static_cast<size_t>(f->get_result_index(store) + num_parameters)};
        } else {
            for (const auto& output : n->outputs()) {
                auto out_tensor = output.get_tensor_ptr();
                auto& rt  = out_tensor->get_rt_info();
                rt["reginfo"] = std::vector<size_t>{physical_regs[out_tensor]};
            }
        }
    }
    */

    return false;
}

