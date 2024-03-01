// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_vnni_transform_emitter.hpp"
#include "snippets/utils.hpp"
#include "transformations/snippets/tpp/op/vnni_transform.hpp"

namespace ov {
namespace intel_cpu {

VnniTransformTppEmitter::VnniTransformTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                                                 dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                 const  ov::snippets::lowered::ExpressionPtr& expr) :
                                                 TppEmitter(h, isa, expr) {
    OV_CPU_JIT_EMITTER_ASSERT(isa == dnnl::impl::cpu::x64::avx512_core,
                              "Requires at least avx512_core instruction set");
    const auto& transform_node = std::dynamic_pointer_cast<tpp::op::VnniTransform>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(transform_node, "Invalid TPP node type detected");

    const auto& subtensor_in0 = get_projected_subtensor(io_port_descriptors[0]);
    const auto& planar_shape = snippets::utils::get_planar_vdims(io_port_descriptors[0]->get_shape(),
                                                                 io_port_descriptors[0]->get_layout());
    // todo: subtensors are ignored, since blocking loop is incorporated into emitter
    //m_N_subtensor = static_cast<libxsmm_blasint>(*subtensor_in0.rbegin());
    m_N_subtensor = transform_node->get_n_block_size();
    // todo: M_blocking is currently ignored to reproduce BrgemmCopyB behavior
    // const auto M_subtensor = static_cast<libxsmm_blasint>(*++subtensor_in0.rbegin());
    const auto M_subtensor = static_cast<libxsmm_blasint>(*++planar_shape.rbegin());
    m_N_full = static_cast<libxsmm_blasint>(*planar_shape.rbegin());
    m_N_subtensor = std::min(m_N_subtensor, m_N_full);
    m_N_tail = m_N_full % m_N_subtensor;

    m_dtype_size = expr->get_node()->get_input_element_type(0).size();

    m_op_type = transform_node->get_op_type();
    m_vnni_factor = transform_node->get_vnni_factor(transform_node->get_input_element_type(0));
    // Note: libxsmm implies column-major layout, so we have to swap M and N here
    exec_dtype = io_dtypes[0];
    m_shape = libxsmm_create_meltw_unary_shape(m_N_subtensor, M_subtensor,
                                               io_strides[0], io_strides[1],
                                               io_dtypes[0], io_dtypes[1],
                                               exec_dtype);
    m_block_kernel = libxsmm_dispatch_meltw_unary(m_op_type, m_shape, m_compile_flags);
    if (m_N_tail > 0) {
        m_shape = libxsmm_create_meltw_unary_shape(m_N_tail, M_subtensor,
                                                   io_strides[0], io_strides[1],
                                                   io_dtypes[0], io_dtypes[1],
                                                   exec_dtype);
        m_tail_kernel = libxsmm_dispatch_meltw_unary(m_op_type, m_shape, m_compile_flags);
    }
    m_compile_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
}

void VnniTransformTppEmitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    Xbyak::Reg64 src(static_cast<int>(in[0]));
    Xbyak::Reg64 dst(static_cast<int>(out[0]));
    const int in_ptr_shift = m_N_subtensor * m_dtype_size;
    const int out_ptr_shift = in_ptr_shift * m_vnni_factor;
    const int num_iter = m_N_full / m_N_subtensor;

    internal_call_preamble();
    h->push(src);
    h->push(dst);
    int i = 0;
    auto update_ptrs = [&]() {
        h->mov(src, h->ptr[h->rsp + gpr_size]);
        h->mov(dst, h->ptr[h->rsp]);
        h->add(src, i * in_ptr_shift);
        h->add(dst, i * out_ptr_shift);
    };
    for (; i < num_iter; i++) {
        if (i)
            update_ptrs();
        emit_call(in, out, get_execute_function_ptr(), reinterpret_cast<const uintptr_t>(m_block_kernel));
    }
    // If tail kernel was created, then tail is required
    if (m_tail_kernel) {
        update_ptrs();
        emit_call(in, out, get_execute_function_ptr(), reinterpret_cast<const uintptr_t>(m_tail_kernel));
    }
    h->add(h->rsp, gpr_size * 2);
    internal_call_postamble();
}

void VnniTransformTppEmitter::execute_kernel(libxsmm_meltwfunction_unary eltwise_kernel, void *in0, void *out0) {
    libxsmm_meltw_unary_param param;
    param.op.primary = nullptr;
    param.in.primary = in0;
    param.out.primary = out0;
    eltwise_kernel(&param);
}

std::set<std::vector<element::Type>> VnniTransformTppEmitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::i8}, {element::u8}, {element::bf16}};
}

void VnniTransformTppEmitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 1, "Expects 1 input registers, got " + std::to_string(in.size()));
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output register, got " + std::to_string(out.size()));
}

}  // namespace intel_cpu
}  // namespace ov
