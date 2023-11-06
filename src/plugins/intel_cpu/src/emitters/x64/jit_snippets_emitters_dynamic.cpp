// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_snippets_emitters_dynamic.hpp"
#include "emitters/utils.hpp"

using namespace InferenceEngine;
using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

using jit_generator = dnnl::impl::cpu::x64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

KernelDynamicEmitter::KernelDynamicEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
        : KernelEmitter(h, isa), SnippetsDynamicEmitter(), reg_runtime_params_idx(abi_param1.getIdx()) {
    init_body_parameters(expr);
    init_reg_pools({reg_runtime_params_idx}, {});

    mapping_info gpr_map_pool({}, gp_regs_pool);
    mapping_info vec_map_pool({}, vec_regs_pool);
    map_abstract_registers(gpr_map_pool, vec_map_pool, mem_access_exprs);
    for (const auto& abstract_to_physical : gpr_map_pool.first)
        data_ptr_regs_idx.push_back(abstract_to_physical.second);
    map_abstract_registers(gpr_map_pool, vec_map_pool, general_exprs);
}


void KernelDynamicEmitter::init_data_pointers(const Xbyak::Reg64& reg_runtime_params, const std::vector<Xbyak::Reg64>& data_ptr_regs) const {
    const auto num_params = num_inputs + num_outputs;
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        h->mov(data_ptr_regs[num_params + i], h->ptr[reg_runtime_params + GET_OFF_DYN(buffer_scratchpad_ptr)]);
    }
    for (size_t i = 0; i < num_params; i++) {
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF_DYN(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF_DYN(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
    }
}
void KernelDynamicEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    h->preamble();

    Reg64 reg_runtime_params = Reg64(static_cast<int>(reg_runtime_params_idx));
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_regs_idx, data_ptr_regs);

    init_data_pointers(reg_runtime_params, data_ptr_regs);

//    RegPrinter::print<int>(*h,  data_ptr_regs[0], "data_ptr_regs_0");
//    auto Vmm = Xbyak::Zmm(0);
//    h->uni_vmovups(Vmm, h->ptr[data_ptr_regs[0]]);
//    RegPrinter::print<float>(*h,  Vmm, "data_ptr_regs_val_0");
//
//    RegPrinter::print<int>(*h,  data_ptr_regs[1], "data_ptr_regs_1");
//    h->uni_vmovups(Vmm, h->ptr[data_ptr_regs[1]]);
//    RegPrinter::print<float>(*h,  Vmm, "data_ptr_regs_val_1");
//
//    RegPrinter::print<int>(*h,  data_ptr_regs[2], "data_ptr_regs_2");
//    h->uni_vmovups(Vmm, h->ptr[data_ptr_regs[2]]);
//    RegPrinter::print<float>(*h,  Vmm, "data_ptr_regs_val_2");

    for (const auto& expression : body) {
        const auto& emitter = expression->get_emitter();
        std::vector<size_t> in_regs, out_regs;
        std::tie(in_regs, out_regs) = expression->get_reg_info();
        // Note all DynamicEmitters should have access to the runtime_params argument,
        // since parameters computed by configurator are stored there.
        if (std::dynamic_pointer_cast<SnippetsDynamicEmitter>(emitter))
            in_regs.push_back(reg_runtime_params_idx);
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, gp_regs_pool);
    }
    h->postamble();
}

LoopBeginDynamicEmitter::LoopBeginDynamicEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa), loop_begin_label{new Xbyak::Label()} {
    const auto& loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node());
    // todo: disabled for debug purposes. re-enable before merge
    // OPENVINO_ASSERT(loop_begin && loop_begin->is_dynamic(), "LoopBeginDynamicEmitter invoked with invalid op argument");
    const auto& out_connectors = expr->get_output_port_connectors();
    const auto& consumers = out_connectors[0]->get_consumers();
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(consumers.begin()->get_expr()->get_node());
    OPENVINO_ASSERT(out_connectors.size() == 1 && consumers.size() == 1 && loop_end,
                    "LoopBeginDynamicEmitter invoked with invalid configuration");
    loop_id = loop_end->get_id();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void LoopBeginDynamicEmitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                        const std::vector<size_t> &pool_vec, const std::vector<size_t> &pool_gpr) const {
    validate_arguments(in, out);
    jit_emitter::emit_code(in, out, pool_vec, pool_gpr);
}

void LoopBeginDynamicEmitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    // Note: the only expected input is the reg_runtime_params_idx
    OPENVINO_ASSERT(in.size() == 1, "Invalid inputs size: expected 1 got " + std::to_string(in.size()));
    // Note: the only expected output is work amount register (communicated to LoopEndEmitter)
    OPENVINO_ASSERT(out.size() == 1, "Invalid outputs size: expected 1 got " + std::to_string(out.size()));
}

void LoopBeginDynamicEmitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    Reg64 reg_work_amount = Reg64(static_cast<int>(out.back()));
    Reg64 reg_runtime_params = Reg64(static_cast<int>(in.back()));
    Reg64 reg_loop_args_ptr = Reg64(static_cast<int>(aux_gpr_idxs[0]));
    const auto id_offset = loop_id * sizeof(jit_snippets_dynamic_call_args::loop_args_t);
    h->mov(reg_loop_args_ptr, h->ptr[reg_runtime_params + GET_OFF_DYN(loop_args)]);
    h->mov(reg_work_amount, h->ptr[reg_loop_args_ptr + id_offset + GET_OFF_LOOP_ARGS(m_work_amount)]);
    h->L(*loop_begin_label);
//    loop_begin->begin_address = h->getCurr();
}

LoopEndDynamicEmitter::LoopEndDynamicEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    const auto& loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    OPENVINO_ASSERT(loop_end, "LoopEndDynamicEmitter invoked with invalid op argument");
    // todo: can we rely on the fact that LoopBegin connected to the last port / uses the last connector?
    for (const auto& in_conn : expr->get_input_port_connectors()) {
        const auto& begin_expr = in_conn->get_source().get_expr();
        if (ov::is_type<snippets::op::LoopBegin>(begin_expr->get_node())) {
            const auto& loop_begin_emitter = std::dynamic_pointer_cast<LoopBeginDynamicEmitter>(begin_expr->get_emitter());
            OPENVINO_ASSERT(loop_begin_emitter, "Invalid emitter detected for LoopBegin operation");
            loop_begin_label = loop_begin_emitter->get_begin_label();
            break;
        }
    }
    OPENVINO_ASSERT(loop_begin_label, "LoopEndDynamicEmitter couldn't find connected LoopEndDynamicEmitter");
    // Note that 1 edge connects LoopBegin and LoopEnd
    num_inputs = expr->get_input_count();
    num_outputs = expr->get_output_count();
    wa_increment = static_cast<int64_t>(loop_end->get_increment());
    io_data_size = loop_end->get_element_type_sizes();
    // Note: io_data_size is less than num_inputs because the last input is LoopBegin
    OPENVINO_ASSERT(io_data_size.size() == num_inputs - 1, "LoopEndDynamicEmitter detected invalid number of io_data_size elements");
    loop_id = loop_end->get_id();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void LoopEndDynamicEmitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                      const std::vector<size_t> &pool_vec, const std::vector<size_t> &pool_gpr) const {
    validate_arguments(in, out);
    jit_emitter::emit_code(in, out, pool_vec, pool_gpr);
}


void LoopEndDynamicEmitter::validate_arguments(const std::vector<size_t> &in,
                                        const std::vector<size_t> &out) const {
    // Note: there must be additional input argument for runtime parameters
    OPENVINO_ASSERT(in.size() == num_inputs + 1, "Invalid number of in arguments.");
    OPENVINO_ASSERT(out.size() == num_outputs, "Invalid number of out arguments.");
}

void LoopEndDynamicEmitter::emit_impl(const std::vector<size_t>& in,
                               const std::vector<size_t>& out) const {
    Reg64 reg_runtime_params = Reg64(static_cast<int>(in[in.size() - 1]));
    Reg64 reg_work_amount = Reg64(static_cast<int>(in[in.size() - 2]));
    Reg64 reg_increments = Reg64(static_cast<int>(aux_gpr_idxs[0]));
    const auto id_offset = loop_id * sizeof(jit_snippets_dynamic_call_args::loop_args_t);

    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(std::vector<size_t>(in.begin(), in.end() - 2), data_ptr_regs);

    // todo: Note that we can pre-save reg_loop_args_ptr in LoopBeginEmitter and pass it here like work_amount_reg
    //  this would save us one dereferencing here and in finalization offsets
    h->mov(reg_increments, h->ptr[reg_runtime_params + GET_OFF_DYN(loop_args)]);
    h->mov(reg_increments, h->ptr[reg_increments + id_offset + GET_OFF_LOOP_ARGS(m_ptr_increments)]);
    for (size_t idx = 0; idx < data_ptr_regs.size(); idx++) {
        h->add(data_ptr_regs[idx], h->ptr[reg_increments + idx * sizeof(int64_t)]);
    }
    h->sub(reg_work_amount, wa_increment);
    h->cmp(reg_work_amount, wa_increment);
    h->jge(*loop_begin_label);

    h->mov(reg_increments, h->ptr[reg_runtime_params + GET_OFF_DYN(loop_args)]);
    h->mov(reg_increments, h->ptr[reg_increments + id_offset + GET_OFF_LOOP_ARGS(m_finalization_offsets)]);
    for (size_t idx = 0; idx < data_ptr_regs.size(); idx++) {
            h->add(data_ptr_regs[idx], h->ptr[reg_increments + idx * sizeof(int64_t)]);
    }
}


}   // namespace intel_cpu
}   // namespace ov
