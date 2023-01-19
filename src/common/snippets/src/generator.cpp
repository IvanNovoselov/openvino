// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"
#include "snippets/lowered_expr.hpp"
#include "snippets/pass/assign_registers.hpp"
#include "snippets/pass/vector_to_scalar.hpp"
#include "snippets/pass/insert_load_store.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/op/kernel.hpp"
#include <snippets/itt.hpp>
#include "snippets/pass/lowered/assign_registers.hpp"
#include "snippets/pass/lowered/insert_tail_loop.hpp"
#include "snippets/pass/lowered/insert_loops.hpp"
#include "snippets/pass/lowered/transpose_decomposition.hpp"
#include "snippets/pass/lowered/buffer_propagate_offset_and_reset.hpp"
#include "snippets/lowered_expr.hpp"
#include <ngraph/pass/manager.hpp>
#include <openvino/core/type.hpp>

namespace ngraph {
namespace snippets {

code Generator::generate(std::shared_ptr<ov::Model>& m, const LoweringConfig& config, const void* compile_params) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")
    if (!target->is_supported())
        throw ngraph_error("unsupported architecture for code generation");
    /*
    auto nodes = old_lowering(m, target, config);
    auto old_linear_ir = LoweredExprIR();
    for (const auto& n : nodes) {
        auto expr = std::make_shared<LoweredExpr>(n);
        auto rinfo = LoweredExpr::getRegisters(n);
        expr->set_reg_info(rinfo);
        old_linear_ir.get_ops().emplace_back(expr);
    }
    */
    auto print_rinfo = [](RegInfo rinfo) {
        std::cerr << "   ";
        for (auto i : rinfo.first)
            std::cerr << i << " ";
        std::cerr << " => ";
        for (auto i : rinfo.second)
            std::cerr << i << " ";
        std::cerr << "\n";
    };
    auto linear_ir = LoweredExprIR(m, config);
//    linear_ir = std::move(old_linear_ir);
//    linear_ir.debug_print();
    pass::transposeDecomposition(linear_ir);
    ov::pass::Serialize("snsdebug_lowered2.xml", "snsdebug_lowered2.bin").run_on_model(m);
    std::cerr << "AFTER Transpose Decomp: =====================\n";
    linear_ir.debug_print();
    pass::insertLoopsLowered(linear_ir, target->get_lanes(), config.m_explicit_loop_insertion);
    pass::buffer_propagate_offset_and_reset(linear_ir);
    std::cerr << "AFTER LOOP INS: =====================\n";
    linear_ir.debug_print();
    std::cerr << "=====================\n";
    m->validate_nodes_and_infer_types();
    pass::assignRegisters(linear_ir);
//    std::string failed_ops("");
//    for (const auto&  expr : linear_ir.get_ops()) {
//        auto rinfo = expr->get_reg_info();
//        auto rinfo_expected = LoweredExpr::getRegisters(expr->get_node());
//        if (rinfo != rinfo_expected) {
////            expr->set_reg_info(rinfo_expected);
//            failed_ops += expr->get_node()->get_friendly_name() + "\n";
//            std::cerr << expr->get_node()->get_friendly_name() << "\n";
//            std::cerr << "Expected:\n";
//            print_rinfo(rinfo_expected);
//            std::cerr << "Actual:\n";
//            print_rinfo(rinfo);
//        }
//    }
//    if (!failed_ops.empty()) {
//        std::cerr << "register assignment error\n";
//        throw ngraph_error("register assignment error");
//    }
    linear_ir.debug_print();
//    int i = 0;
//    for (auto it = linear_ir.get_ops().begin(); i < 64; i++) {
//        std::cerr << i << " : " <<(*it++)->get_node()->get_friendly_name() << "\n";
//    }
//    throw ngraph_error("FINITA!");
    // todo: modify this pass so if no vector loop is needed, then the appropriate work_amounts are set at insertion time
    pass::insertTailLoop(linear_ir);
    linear_ir.serialize("snsdebug_linear.xml", "snsdebug_linear.bin");

    linear_ir.init_emitters(target);

    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")
    //todo: Kernel need info on i/o data access pattern and data shapes to calculate data offsets
    // pass Params and Results
    // todo: it's probably better to move AllocaledEmitter creation inside Kernel constructor
    //  So Kernel accepts only model ptr and target, and creates AllocatedEmitter inside
    //emission
    auto loops2DKernel = std::make_shared<op::Kernel>(linear_ir, m);
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

    return target->get_snippet();
}

std::shared_ptr<const TargetMachine> Generator::get_target_machine() const {
    return target;
}

}// namespace snippets
}// namespace ngraph
