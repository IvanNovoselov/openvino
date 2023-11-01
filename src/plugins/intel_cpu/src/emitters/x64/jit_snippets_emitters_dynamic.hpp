// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_snippets_emitters.hpp"

namespace ov {
namespace intel_cpu {

#define GET_OFF_DYN(field) offsetof(jit_snippets_dynamic_call_args, field)
#define GET_OFF_LOOP_ARGS(field) offsetof(jit_snippets_dynamic_call_args::loop_args_t, field)
struct jit_snippets_dynamic_call_args {
    struct loop_args_t {
        //todo: can we use smaller data types?
        int64_t work_amount = 0;
        int64_t num_data_ptrs = 0;
        int64_t* ptr_increments = nullptr;
        int64_t* finalization_offsets = nullptr;
    };
    int32_t num_loops = 0;
    loop_args_t* loop_args = nullptr;
    const void *src_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *dst_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    // Src and dst offsets calculated once for every set of input shapes,
    // equivalent to the result of offset_calculation() in static kernel
    int64_t data_offsets[SNIPPETS_MAX_SNIPPETS_DIMS][SNIPPETS_DYNAMIC_MASTER_SHAPE_RANK] = {};
    void *buffer_scratchpad_ptr = nullptr;
};

// All emitters for dynamic operations should be derived from this class.
// This should be done to distinguish between static and dynamic emitters.
class SnippetsDynamicEmitter {
public:
    virtual ~SnippetsDynamicEmitter() = default;
};

class KernelDynamicEmitter : public KernelEmitter, public SnippetsDynamicEmitter {
public:
    KernelDynamicEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                         dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}

private:
    using jit_emitter::emit_code;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;
    void init_data_pointers(const Xbyak::Reg64&, const Xbyak::Reg64&, const std::vector<Xbyak::Reg64>&) const;
};

class LoopBeginDynamicEmitter : public jit_emitter, public SnippetsDynamicEmitter {
public:
    LoopBeginDynamicEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                            dnnl::impl::cpu::x64::cpu_isa_t isa,
                            const ov::snippets::lowered::ExpressionPtr& expr);

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const override;
    // Todo: do we use get_inputs_num anywhere? Can we get rig of it?
    size_t get_inputs_num() const override {return 1;}
    size_t aux_gprs_count() const override {return 1;}
    std::shared_ptr<const Xbyak::Label> get_begin_label() {return loop_begin_label;}

private:
    using jit_emitter::emit_code;
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    std::shared_ptr<Xbyak::Label> loop_begin_label;
    size_t loop_id;
};

class LoopEndDynamicEmitter : public jit_emitter, public SnippetsDynamicEmitter {
public:
    LoopEndDynamicEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                          dnnl::impl::cpu::x64::cpu_isa_t isa,
                          const ov::snippets::lowered::ExpressionPtr& expr);
    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const override;
    size_t get_inputs_num() const override {return 0;}
    size_t aux_gprs_count() const override {return 2;}

private:
    using jit_emitter::emit_code;
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out) const override;

    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    std::shared_ptr<const Xbyak::Label> loop_begin_label;

    size_t num_inputs = 0;
    size_t num_outputs = 0;
    size_t loop_id;
    // keep data_size int64_t to avoid conversion to size_t (and overflow) when multiplied by negative increments or offsets
    std::vector<int64_t> io_data_size {};
    int64_t wa_increment = 0;
};

}   // namespace intel_cpu
}   // namespace ov
