// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "domain_optimization.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "snippets/lowered/shape_inference/shape_inference.hpp"


namespace ov {
namespace intel_cpu {
namespace pass {
using VectorDims = snippets::IShapeInferSnippets::VectorDims;

DomainOptimization::DomainOptimization(size_t min_parallel_work_amount, size_t min_jit_work_amount)
                  : Pass(), m_min_parallel_work_amount{min_parallel_work_amount}, m_min_jit_work_amount{min_jit_work_amount} {
}
bool DomainOptimization::run(snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::DomainOptimization")
    if (linear_ir.empty())
        return false;

    std::vector<std::shared_ptr<snippets::lowered::IOExpression>> input_exprs;
    std::vector<snippets::IShapeInferSnippets::VectorDims> input_shapes;
    VectorDims master_shape{1};
    for (const auto& expr : linear_ir.get_IO_ops()) {
        if (expr->get_type() == snippets::lowered::IOExpression::io_type::INPUT) {
            input_exprs.push_back(expr);
            const auto& shape = expr->get_output_port_descriptor(0)->get_shape();
            OPENVINO_ASSERT(std::any_of(shape.begin(), shape.end(),
                                        [](size_t d) {return d == snippets::IShapeInferSnippets::DYNAMIC_DIMENSION; }),
                            "DomainOptimization pass does not support dynamic shapes");
            OPENVINO_ASSERT(ov::snippets::broadcast_merge_into(master_shape, shape),
                            "Failed to merge input shapes in DomainOptimization pass");
            input_shapes.emplace_back(shape);
        }
    }

    const auto total_work_amount = std::accumulate(master_shape.begin(),
                                                   master_shape.end(),
                                                   1,
                                                   std::multiplies<size_t>());
    if (master_shape.size() <= 2 ||                                               // Nothing to collapse
        *master_shape.rbegin() >= m_min_jit_work_amount ||                        // Already enough work for JIT kernel, no need to collapse
        total_work_amount < m_min_parallel_work_amount * m_min_jit_work_amount) { // There won't be enough work for every thread, no point to collapse
        std::cerr << "aborted\n" << std::flush;
        return false;
    }

    auto CollapseLastDim = [](VectorDims& dims) {
        OPENVINO_ASSERT(dims.size() >= 2, "CollapseLastDim can't process shape with less than two dims");
        dims[dims.size() - 2] *= dims.back();
        for (auto i = dims.size() - 2; i > 0; i--)
            dims[i] = dims[i - 1];
    };
    auto LastDimsNotBroadcasted = [&input_shapes, &master_shape] () {
        for (const auto& s : input_shapes) {
            if (s.back() != master_shape.back())
                return false;
        }
        return true;
    };

    size_t jit_work_amount = master_shape.back();
    size_t next_jit_work_amount = jit_work_amount * master_shape[master_shape.size() - 2];
    bool some_dims_collapsed {false};
    while (jit_work_amount < m_min_jit_work_amount &&
           next_jit_work_amount * m_min_parallel_work_amount < total_work_amount &&
           master_shape.size() > 2 &&
           LastDimsNotBroadcasted()) {

        for (auto &s : input_shapes)
            CollapseLastDim(s);

        CollapseLastDim(master_shape);

        jit_work_amount = next_jit_work_amount;
        next_jit_work_amount *= master_shape[master_shape.size() - 2];
        some_dims_collapsed = true;
    }
    if (some_dims_collapsed) {
        for (auto i = 0; i < input_exprs.size(); i++) {
            const auto& expr = input_exprs[i];
            const auto& par = ov::as_type_ptr<ov::op::v0::Parameter>(expr->get_node());
            OPENVINO_ASSERT(par, "Input expression does not contain Parameter node.");
            par->set_partial_shape(ov::Shape(input_shapes[i]));
        }
        // todo: we need to trigger shapeInfer here to reshape the graph
        //  Subgraph::LIRShapeInferSnippets::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes)
    }
    return some_dims_collapsed;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov