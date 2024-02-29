// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_tpp_emitter.hpp"
namespace ov {
namespace intel_cpu {

// todo: VnniTransformTppEmitter is very similar to UnaryEltwiseTppEmitter exept for incorporated loop semantics
//  It can be simplified as UnaryEltwiseTppEmitter when loop is moved to LIR
class VnniTransformTppEmitter : public TppEmitter {
public:
    VnniTransformTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                            dnnl::impl::cpu::x64::cpu_isa_t isa,
                            const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override { return 1; }

    static void execute_kernel(libxsmm_meltwfunction_unary eltwise_kernel, void *in0, void *out0);
    const uintptr_t get_compiled_kernel_ptr() const override {return 0;}
    const uintptr_t get_execute_function_ptr() const override { return reinterpret_cast<const uintptr_t>(execute_kernel); }
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

protected:
    libxsmm_meltw_unary_shape m_shape;
    libxsmm_meltw_unary_type m_op_type;
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    int m_dtype_size = 0;
    int m_vnni_factor = 0;
    libxsmm_blasint m_N_subtensor = 0;
    libxsmm_blasint m_N_tail = 0;
    libxsmm_blasint m_N_full = 0;
    libxsmm_meltwfunction_unary m_block_kernel = nullptr;
    libxsmm_meltwfunction_unary m_tail_kernel = nullptr;
};

}   // namespace intel_cpu
}   // namespace ov
