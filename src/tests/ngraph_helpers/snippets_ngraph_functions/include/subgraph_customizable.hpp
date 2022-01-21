// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "snippets_helpers.hpp"
#include "openvino/op/util/op_types.hpp"

/* This file contains definitions of rather complex functions (models) that support (and require)
 * specification of some the internal operations. This flexibility is required to extend coverage of
 * different tokenization scenarios in parametrized tests. All the functions are expected to be direct
 * descendants of SnippetsFunctionCustomizable (defined here).
 */

namespace ngraph {
namespace builder {
namespace subgraph {
/// \brief Base class for snippets subgraphs with customizable embedded op sequences. Note that the custom_ops allowed types are
/// model-specific and expected to be checked inside a child class constructor.
/// \param  custom_ops  vector of ops to be inserted in the graph. Required vector size and acceptable op types are subgraph-specific.
/// The ops are expected to be default-constructed to facilitate test development, the class will take care of the ops inputs for you.
/// \param  customOpsNumInputs  size_t vector that specifies the number of inputs for each of the custom_ops. Not that an rvalue is expected,
/// since it should be hard-coded along with the Original and Reference functions.
class SnippetsFunctionCustomizable : public SnippetsFunctionBase {
public:
    SnippetsFunctionCustomizable() = delete;
    SnippetsFunctionCustomizable(std::vector<Shape>& inputShapes,
                                 std::vector<std::shared_ptr<Node>>& customOps,
                                 std::vector<size_t>&& customOpsNumInputs)
            : SnippetsFunctionBase(inputShapes), custom_ops{customOps} {
        custom_ops_num_inputs = std::move(customOpsNumInputs);
        NGRAPH_CHECK(custom_ops_num_inputs.size() == custom_ops.size(), "Got inconsistent numbers of custom ops and custom ops inputs");
        // We need to set dummy inputs to increase input arguments count,
        // so clone_with_new_inputs() could pass without errors inside initOriginal() and initReference().
        ResetCustomOpsInputs();
    };

protected:
    std::vector<std::shared_ptr<Node>> custom_ops;
    std::vector<size_t> custom_ops_num_inputs;
    void ResetCustomOpsInputs() {
        auto dummy_input = std::make_shared<ov::op::v0::Parameter>(precision, Shape{});
        for (size_t i = 0; i < custom_ops.size(); i++) {
            const NodeVector inputs(custom_ops_num_inputs[i], dummy_input);
            custom_ops[i]->set_arguments(inputs);
        }
    }
};
/// Convolution followed by a two-input Multiply, Relu and Sqrt
/// Tokenized by attaching eltwises, but becomes non-tokenizable if Multiply is substituted with Add (CPU-specific fusing)
//    in1          in2
// Convolution   Convert
//         Multiply
//           Relu
//           Sqrt
//          Result
class ConvMulActivation : public SnippetsFunctionCustomizable {
public:
    explicit ConvMulActivation(std::vector<Shape> inputShapes, std::vector<std::shared_ptr<Node>> customOps)
            : SnippetsFunctionCustomizable(inputShapes, customOps, {2, 1, 1}) {
            NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
            NGRAPH_CHECK(input_shapes[0].size() == 4, "Only 4D input shapes are currently supported");
            NGRAPH_CHECK(ov::op::util::is_binary_elementwise_arithmetic(customOps[0]) &&
                         ov::op::util::is_unary_elementwise_arithmetic(customOps[1]) &&
                         ov::op::util::is_unary_elementwise_arithmetic(customOps[2]),
                         "Got invalid custom ops: expected binary and two unary operations");
    }
private:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
