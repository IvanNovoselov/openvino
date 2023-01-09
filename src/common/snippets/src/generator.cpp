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
#include "snippets/pass/lowered_ir_transformations.hpp"
#include "snippets/lowered_expr.hpp"
#include <ngraph/pass/manager.hpp>
#include <openvino/core/type.hpp>

namespace ngraph {
namespace snippets {

code Generator::generate(std::shared_ptr<ov::Model>& m, const LoweringConfig& config, const void* compile_params) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")
    if (!target->is_supported())
        throw ngraph_error("unsupported architecture for code generation");

    auto linear_ir = LoweredExprIR(m, config);
    for (const auto& expr : linear_ir.get_ops()) {
        std::cerr << expr->get_node()->get_friendly_name() << " ";
//        if (auto io = std::dynamic_cast<IOLoweredExpr>(exp))
        std::cerr << "\n";
    }
    std::cerr << "\n\n================ " << __PRETTY_FUNCTION__ << "  :  " <<  __LINE__ << "\n";


    pass::assignRegisters(linear_ir);
    auto print_rinfo = [](RegInfo rinfo) {
        for (auto i : rinfo.first)
            std::cerr << i << " ";
        std::cerr << " => ";
        for (auto i : rinfo.second)
            std::cerr << i << " ";
        std::cerr << "\n";
    };
//    bool terminate{false};
//    for (auto expr : linear_ir.get_ops()) {
//        auto rinfo = expr->get_reg_info();
//        auto rinfo_expected = LoweredExpr::getRegisters(expr->get_node());
//        expr->set_reg_info(rinfo_expected);
//        if (rinfo != rinfo_expected) {
//            expr->set_reg_info(rinfo_expected);
//            std::cerr << expr->get_node()->get_friendly_name() << " :\n";
//            std::cerr << "      Exp: ";
//            print_rinfo(rinfo_expected);
//            std::cerr << "      Got: ";
//            print_rinfo(rinfo);
//            terminate = true;
//        }
//    }
//    if (terminate)
//        throw ngraph_error("register assignment error");
    pass::insertTailLoop(linear_ir);

    std::cerr << "\n\n================ " << __PRETTY_FUNCTION__ << "  :  " <<  __LINE__ << "\n";
    for (const auto& expr : linear_ir.get_ops()) {
        std::cerr << expr->get_node()->get_friendly_name() << " ";
//        if (auto io = std::dynamic_cast<IOLoweredExpr>(exp))
        std::cerr << "\n";
    }
    std::cerr << "\n\n---------------------------------------\n";

    linear_ir.init_emitters(target);

//    for (const auto& expr : lowered_ir.get_ops() )
//        std::cerr << expr.get_node()->get_friendly_name() << "\n";
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
    for (const auto& l : linear_ir.get_ops()) {
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
