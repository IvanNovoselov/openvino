// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/domain_optimization.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/lowered/shape_inference/shape_inference.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

using VectorDims = IShapeInferSnippets::VectorDims;
DomainOptimization::DomainOptimization() : Pass() {
}
bool DomainOptimization::optimize(std::vector<VectorDims>& input_shapes,
                                  VectorDims& master_shape,
                                  const size_t min_parallel_work_amount,
                                  const size_t min_jit_work_amount) {
    const auto total_work_amount = std::accumulate(master_shape.begin(),
                                                   master_shape.end(),
                                                   (size_t)1,
                                                   std::multiplies<size_t>());
    if (master_shape.size() <= 2 ||                                           // Nothing to collapse
        *master_shape.rbegin() >= min_jit_work_amount ||                      // Already enough work for JIT kernel, no need to collapse
        total_work_amount < min_parallel_work_amount * min_jit_work_amount) { // There won't be enough work for every thread, no point to collapse
        return false;
    }

    auto CollapseLastDim = [](VectorDims& dims) {
        OPENVINO_ASSERT(dims.size() >= 2, "CollapseLastDim can't process shape with less than two dims");
        dims[dims.size() - 2] *= dims.back();
        for (auto i = dims.size() - 2; i > 0; i--)
            dims[i] = dims[i - 1];
    };
    // Check that neither of the two last dims is broadcasted, so they can be collapsed
    auto LastDimsNotBroadcasted = [] (const std::vector<VectorDims>& input_shapes, const VectorDims& master_shape) {
        const auto master_last = *master_shape.rbegin();
        const auto master_prelast = *++master_shape.rbegin();
        return std::all_of(input_shapes.begin(), input_shapes.end(),
                           [=](const VectorDims& s) {
                               return *s.rbegin() == master_last &&
                                      *++s.rbegin() != master_prelast;
                            });
    };

    size_t jit_work_amount = master_shape.back();
    size_t next_jit_work_amount = jit_work_amount * master_shape[master_shape.size() - 2];
    bool some_dims_collapsed {false};
    while (jit_work_amount < min_jit_work_amount &&
           next_jit_work_amount * min_parallel_work_amount < total_work_amount &&
           master_shape.size() > 2 &&
           LastDimsNotBroadcasted(input_shapes, master_shape)) {
        for (auto &s : input_shapes)
            CollapseLastDim(s);

        CollapseLastDim(master_shape);

        jit_work_amount = next_jit_work_amount;
        next_jit_work_amount *= master_shape[master_shape.size() - 2];
        some_dims_collapsed = true;
    }
    return some_dims_collapsed;
}

bool DomainOptimization::run(snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::DomainOptimization")
    const auto& config = linear_ir.get_config();
    if (linear_ir.empty())
        return false;
    if (!config.m_enable_domain_optimization)
        return false;

    std::vector<std::shared_ptr<snippets::lowered::IOExpression>> input_exprs;
    std::vector<VectorDims> input_shapes;
    VectorDims master_shape{1};
    for (const auto& expr : linear_ir.get_IO_ops()) {
        if (expr->get_type() == snippets::lowered::IOExpression::io_type::INPUT) {
            input_exprs.push_back(expr);
            const auto& shape = expr->get_output_port_descriptor(0)->get_shape();
            OPENVINO_ASSERT(std::none_of(shape.begin(), shape.end(),
                                        [](size_t d) {return d == snippets::IShapeInferSnippets::DYNAMIC_DIMENSION; }),
                            "DomainOptimization pass does not support dynamic shapes");
            OPENVINO_ASSERT(ov::snippets::broadcast_merge_into(master_shape, shape),
                            "Failed to merge input shapes in DomainOptimization pass");
            input_shapes.emplace_back(shape);
        }
    }
    const bool some_dims_collapsed = optimize(input_shapes,
                                              master_shape,
                                              config.m_min_parallel_work_amount,
                                              config.m_min_jit_work_amount);
    if (some_dims_collapsed) {
        std::vector<std::reference_wrapper<const VectorDims>> infer_shapes;
        infer_shapes.reserve(input_shapes.size());
        for (size_t i = 0; i < input_exprs.size(); i++) {
            const auto& expr = input_exprs[i];
            const auto& par = ov::as_type_ptr<ov::op::v0::Parameter>(expr->get_node());
            OPENVINO_ASSERT(par, "Input expression does not contain Parameter node.");
            // todo: we won't really need to update nGraph shapes when we move to LIR shape infer
            par->set_partial_shape(ov::Shape(input_shapes[i]));
            par->validate_and_infer_types();
            infer_shapes.emplace_back(input_shapes[i]);
        }
        // Need to propagate updated shapes through LIR
        linear_ir.shape_infer(infer_shapes);
    }
    return some_dims_collapsed;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov