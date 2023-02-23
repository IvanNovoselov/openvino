// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"
#include "snippets/lowered_expr.hpp"
#include "snippets/pass/assign_registers.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/op/kernel.hpp"
#include <snippets/itt.hpp>
#include "snippets/pass/lowered/assign_registers.hpp"
#include "snippets/pass/lowered/insert_buffer.hpp"
#include "snippets/pass/lowered/insert_tail_loop.hpp"
#include "snippets/pass/lowered/insert_loops_layout.hpp"
#include "snippets/pass/lowered/transpose_decomposition.hpp"
#include "snippets/pass/lowered/buffer_propagate_offset_and_reset.hpp"
#include "snippets/pass/lowered/propagate_layout.hpp"
#include "snippets/pass/lowered/softmax_decomposition.hpp"
#include "snippets/lowered_expr.hpp"
#include "snippets/tensor_descriptor.hpp"

namespace ngraph {
namespace snippets {

Generator::LoweringResult Generator::generate(std::shared_ptr<ov::Model>& m, const LoweringConfig& config, const void* compile_params) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")
    if (!target->is_supported())
        throw ngraph_error("unsupported architecture for code generation");
    auto linear_ir = LoweredExprIR(m, config);
    auto propagate_buffer_offsets = std::make_shared<pass::lowered::PropagateOffsetAndResetBuffer>();
    std::vector<std::shared_ptr<pass::lowered::LinearIRTransformation>> transformation_pipeline {
//            std::make_shared<pass::lowered::TransposeDecomposition>(),
            std::make_shared<pass::lowered::InsertLoopsLayout>(target->get_lanes(), config.m_explicit_loop_insertion),
            std::make_shared<pass::lowered::SoftmaxDecomposition>(),
            std::make_shared<pass::lowered::PropagateLayout>(),
            propagate_buffer_offsets,
            std::make_shared<pass::lowered::AssignRegisters>(),
            // todo: modify this pass so if no vector loop is needed, then the appropriate work_amounts are set at insertion time
            std::make_shared<pass::lowered::InsertTailLoop>()
    };
    for (const auto& transform : transformation_pipeline) {
        std::string name(transform->get_type_name());
//        if (name == "InsertLoopsLayout") {
//        if (name == "InsertBuffer") {
//        if (name == "PropagateOffsetAndResetBuffer") {
        if (name == "InsertLoopsLayout") {
//            std::cerr << "Before:\n";
            linear_ir.debug_print(1);
            std::cerr << ".....................\n";

        }
        transform->run(linear_ir);
        if (name == "InsertLoopsLayout") {
            linear_ir.debug_print(1);
            linear_ir.serialize("snsdebug_linear.xml", "snsdebug_linear.bin");
//            throw ngraph_error("FINITA");
        }
    }
    const auto buffer_scratchpad_size = propagate_buffer_offsets->get_scratchpad_size();
//    throw ngraph_error("FINITA");
    linear_ir.init_emitters(target);

    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")
    //todo: Kernel need info on i/o data access pattern and data shapes to calculate data offsets
    // pass Params and Results
    // todo: it's probably better to move AllocaledEmitter creation inside Kernel constructor
    //  So Kernel accepts only model ptr and target, and creates AllocatedEmitter inside
    //emission
//    ov::pass::Serialize("snsdebug_linear.xml", "snsdebug_linear.bin").run_on_model(m);
    auto loops2DKernel = std::make_shared<op::Kernel>(linear_ir);
    loops2DKernel->compile_params = compile_params;
    std::shared_ptr<Emitter> kernel = target->get(op::Kernel::get_type_info_static())(loops2DKernel);

    kernel->emit_code({}, {});

    OV_ITT_TASK_NEXT(GENERATE, "::EmitData")
    for (auto& l : linear_ir.get_ops()) {
        l->get_emitter()->emit_data();
    }
    OV_ITT_TASK_NEXT(GENERATE, "::GetSnippet")

    // todo: we save lowered to access compiled brgemm kernels on execution time (normally lowered is destructed by then)
    //  remove this when kernel caching is implemented. Don't forget to make generate const method.
    if (config.m_save_lowered_code)
        lowered_saved = linear_ir;

    return {target->get_snippet(), buffer_scratchpad_size};
}

std::shared_ptr<const TargetMachine> Generator::get_target_machine() const {
    return target;
}

}// namespace snippets
}// namespace ngraph
