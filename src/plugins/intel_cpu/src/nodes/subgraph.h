// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>

#include <onednn/dnnl.h>
#include <cpu/x64/jit_generator.hpp>
#include "emitters/x64/jit_snippets_emitters.hpp"

#include <node.h>
#include "snippets/op/subgraph.hpp"

#include <array>

namespace ov {
namespace intel_cpu {
namespace node {

/// Snippet represents subgraph node in CPU plugin
/// potentially, snippet can be placed as a postop to any support operation while it doesn't support postops itself
/// precision: fp32
class Snippet : public Node {
public:
    Snippet(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr& context);
    ~Snippet() override = default;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    void initOptimalPrimitiveDescriptor() override;
    InferenceEngine::Precision getRuntimePrecision() const override;

    // Here we convert to canonical for & jit everything
    void prepareParams() override;
    bool needPrepareParams() const override;
    //IShapeInfer::Result shapeInfer() const override;

    bool canBeInPlace() const override;
    bool created() const override;

    // if generator is set, it would execute generated code otherwise it would fallback to nGraph reference
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;

    struct SnippetAttrs {
        // Local copy of subgraph node for canonization & code generation
        std::shared_ptr<snippets::op::Subgraph> snippet;
        uint64_t bodyHash;
        std::vector<std::vector<size_t>> inMemBlockedDims;
        std::vector<std::vector<size_t>> inMemOrders;
        std::vector<InferenceEngine::Precision> inMemPrecs;
        std::vector<std::vector<size_t>> outMemBlockedDims;
        std::vector<std::vector<size_t>> outMemOrders;
        std::vector<InferenceEngine::Precision> outMemPrecs;
    };

private:
    typedef void (*kernel)(const void *, const void *);

    void init_body_hash();

    size_t inputNum = 0;
    size_t outputNum = 0;

    // Original subgraph node
    std::shared_ptr<snippets::op::Subgraph> original_snippet;
    mutable std::shared_ptr<snippets::op::Subgraph> local_snippet;

    // Holds ISA version used is codeGeneration target
    dnnl::impl::cpu::x64::cpu_isa_t host_isa;

    std::vector<MemoryPtr> srcMemPtrs = {};
    std::vector<MemoryPtr> dstMemPtrs = {};

    mutable SnippetAttrs snippetAttrs;
    bool is_dynamic = false;

    class SnippetExecutor {
        public:
            SnippetExecutor(SnippetAttrs attrs, bool is_dynamic, bool enforceBF16);
            virtual void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;
            virtual ~SnippetExecutor() = default;
            std::shared_ptr<IShapeInfer> shapeInference = nullptr;

        protected:
            SnippetAttrs snippetAttrs;
            bool is_dynamic = false;
            bool enforceBF16 = false;
    };

    std::shared_ptr<SnippetExecutor> execPtr = nullptr;

    class SnippetJitExecutor : public SnippetExecutor {
        public:
            SnippetJitExecutor(const SnippetAttrs& attrs, bool is_dynamic, bool enforceBF16);
            void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;

            bool schedule_created();

        private:
            static const size_t rank6D {6};
            // Note: these are hint for domain optimization and tileRank selection
            const size_t min_parallel_work_amount {16};
            const size_t min_jit_work_amount {256};

            typedef void (*kernel)(const void *, const void *);

            size_t numInput = 0;
            size_t numOutput = 0;

            void generate(const jit_snippets_compile_args*);
            inline void update_ptrs(jit_snippets_call_args&, const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs);
            // Evaluates generated snippet using parallel backend
            void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs);
            void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs);

            std::shared_ptr<snippets::op::Subgraph> snippet_for_generation;

            // Holds generated snippet with information about how to schedule it
            snippets::Schedule schedule;

            // Holds index of output used as in execution domain
            // it should be compatible with a schedule's work size
            std::vector<size_t> exec_domain = {};

            /// scheduling info
            size_t tensorRank = 0;
            size_t harnessWorkAmount = 0;
            const size_t maxTileRank = 2;

            std::vector<size_t> dataSize = {};

            std::vector<ptrdiff_t> start_offset_in = {};
            std::vector<ptrdiff_t> start_offset_out = {};

            // Buffer scratchpad
            std::vector<uint8_t> buffer_scratchpad = {};
            size_t buffer_scratchpad_size = 0;
    };
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
