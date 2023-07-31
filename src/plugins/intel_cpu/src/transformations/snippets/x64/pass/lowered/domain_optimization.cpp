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


namespace ov {
namespace intel_cpu {
namespace pass {
using LoopManager = snippets::lowered::LinearIR::LoopManager;
using LoopInfoPtr = LoopManager::LoopInfoPtr;
using LoopPort = LoopManager::LoopPort;

DomainOptimization::DomainOptimization(size_t min_parallel_work_amount, size_t min_jit_work_amount)
                  : Pass(), m_min_parallel_work_amount{min_parallel_work_amount}, m_min_jit_work_amount{min_jit_work_amount} {
}
bool DomainOptimization::run(snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmBlocking")
    if (linear_ir.empty())
        return false;

    std::vector<std::shared_ptr<snippets::lowered::IOExpression>> input_exprs;
    std::vector<snippets::lowered::IShaVectorDims>
    for (const auto& expr : linear_ir.get_IO_ops()) {
        if (expr->get_type() == snippets::lowered::IOExpression::io_type::INPUT)
            input_exprs.push_back(expr);
    }


    const size_t ds = domain.size();
    if ( ds <= 2 || // not enough dimensions to collapse
         domain[ds-1] >= minimalJitWorkAmount || // There is enough work for 1D Tiles, no need to collapse
         domain[ds-1] * domain[ds-2] >= fullWorkAmount / minimalConcurrency) { // There won't be enough work for every thread (even one iter) if we collapse
        std::cerr << "aborted\n" << std::flush;
        return false;
    }
    auto findDimsToCollapse = [&]() {
        auto collapseLastDims = [](VectorDims& dims, size_t dimsToCollapse) {
            if (dimsToCollapse >= dims.size() - 1)
                IE_THROW() << "Got invalid number of dims to collapse. Expected < " << dims.size() - 1 << " got " << dimsToCollapse;
            for (int i = dims.size() - 2; i > static_cast<int>(dims.size() - dimsToCollapse - 2); i--) {
                dims[dims.size() - 1] *= dims[i];
            }

            for (int i = dims.size() - 2; i >= static_cast<int>(dimsToCollapse); i--) {
                dims[i] = dims[i - dimsToCollapse];
            }

            for (int i = dimsToCollapse - 1; i >= 0; i--) {
                dims[i] = 1;
            }
        };
        int collapsedDims = 0;
        size_t currentJitWorkAmount = domain[domain.size() - 1];
        while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount) {
            if (static_cast<int>(domain.size()) - collapsedDims - 2 < 0)
                break;

            bool canCollapse = true;
            for (size_t i = 0; i < inputShapes.size(); i++) {
                const size_t last = inputShapes[i].size() - 1;
                if ((inputShapes[i][last - 1] != 1 && inputShapes[i][last] == 1) ||
                    (inputShapes[i][last - 1] == 1 && inputShapes[i][last] != 1)) {
                    canCollapse = false;
                    break;
                }
            }

            size_t nextJitWorkAmount = currentJitWorkAmount * domain[domain.size() - 2];
            if (fullWorkAmount / nextJitWorkAmount >= minimalConcurrency) {
                currentJitWorkAmount = nextJitWorkAmount;
                // if we cannot use dim collapsing we should use tile2D
                if (!canCollapse) {
                    if (TileRank < maxTileRank) {
                        std::cerr << "Tile rank increased\n" << std::flush;
                        TileRank++;
                        continue;
                    }

                    break;
                }
                collapsedDims++;
                for (auto &d : inputShapes)
                    collapseLastDims(d, 1);
                for (auto &d : outputShapes)
                    collapseLastDims(d, 1);
                collapseLastDims(domain, 1);
            } else {
                break;
            }
        }
        std::cerr << "Collapse last dims: " << collapsedDims << "\n" << std::flush;
        return collapsedDims > 0;
    };
    return findDimsToCollapse();












    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;
        const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
        if (!brgemm || blocking_loop_exists(expr, brgemm))
            continue;

        const auto& input_shape_0 = expr->get_input_port_descriptor(0)->get_shape();
        const auto& input_layout_0 = expr->get_input_port_descriptor(0)->get_layout();
        const auto& dim = *(input_layout_0.rbegin() + dim_idx);
        const auto& m = input_shape_0[dim];

        const auto block_size = brgemm->get_m_block_size();
        brgemm->set_input_count(block_size);

        const auto work_amount = m;
        const auto increment = block_size;

        std::vector<LoopPort> entries{LoopPort(expr->get_input_port(0), true), LoopPort(expr->get_input_port(1), false)};
        if (brgemm->is_with_scratchpad())
            entries.emplace_back(expr->get_input_port(2), false);
        std::vector<LoopPort> exits{LoopPort(expr->get_output_port(0), true)};
        loop_manager->mark_loop(expr_it, std::next(expr_it), work_amount, increment, dim_idx, entries, exits);
    }

    return modified;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov