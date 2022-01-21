// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "snippets_helpers.hpp"
#include "subgraph_simple.hpp"

/* This file provides lowered representations (after the generate() was calles) for some simple functions.
 * This is required to test snippets lowering and optimization passes. All the functions are expected to be direct
 * descendants of SnippetsFunctionCustomizable (defined here) and one of the SnippetsFunctionBase derived classes
 * (declared in subgraph_simple.hpp). Note that the corresponding SnippetsFunctionBase child should use virtual inheritance
 * from SnippetsFunctionBase (typically "virtual public") to avoid creation of two internal copies of SnippetsFunctionBase.
 */

namespace ngraph {
namespace builder {
namespace subgraph {

class AddFunctionLoweredBroadcast : public AddFunction {
public:
    explicit AddFunctionLoweredBroadcast(std::vector<Shape> inputShapes, std::vector<Shape> broadcastShapes) :
        AddFunction(std::move(inputShapes)), broadcast_shapes{std::move(broadcastShapes)} {
        NGRAPH_CHECK(input_shapes.size() == broadcast_shapes.size(),
                     "Broadcast shapes should have the same size as input_shapes");
    }

protected:
    std::shared_ptr<ov::Model> initLowered() const override;

private:
    std::vector<Shape> broadcast_shapes;
};

class EltwiseFunctionLowered : public EltwiseFunction {
public:
    EltwiseFunctionLowered(std::vector<Shape> inputShapes,
                                    Subgraph::BlockedShapeVector inputBlockedShapes,
                                    Subgraph::BlockedShapeVector outputBlockedShapes) :
                                    EltwiseFunction{std::move(inputShapes)},
                                    input_blocked_shapes{std::move(inputBlockedShapes)},
                                    output_blocked_shapes{std::move(outputBlockedShapes)} {
        // Blocked shapes include Constant that are processed as additional inputs
        NGRAPH_CHECK(input_shapes.size() <= input_blocked_shapes.size(), "Input shapes and blocked shapes have inconsistent sizes");
        for (size_t i = 0; i < input_shapes.size(); i++) {
            Shape shape;
            AxisVector order;
            element::Type prec;
            std::tie(shape, order, prec) = input_blocked_shapes[i];
            NGRAPH_CHECK(prec == precision, "Invalid input precision provided");
            NGRAPH_CHECK(shape_size(shape) == shape_size(input_shapes[i]),
                         "Got invalid input blocked shape. It must have the same num elements as the plain input shape.");
            NGRAPH_CHECK(shape.size() == order.size(), "Got invalid number of elements in the order AxisVector.");
        }
        for (size_t i = 0; i < output_blocked_shapes.size(); i++) {
            Shape shape;
            AxisVector order;
            element::Type prec;
            std::tie(shape, order, prec) = input_blocked_shapes[i];
            NGRAPH_CHECK(prec == precision, "Invalid input precision provided");
            // todo: Can we write more detailed checks for the output blocked shapes?
            NGRAPH_CHECK(shape.size() == order.size(), "Got invalid number of elements in the order AxisVector.");
        }
    }
    Subgraph::BlockedShapeVector getInputBlockedShapes() const {return input_blocked_shapes;}
    Subgraph::BlockedShapeVector getOutputBlockedShapes() const {return output_blocked_shapes;}

private:
    std::shared_ptr<ov::Model> initLowered() const override;
    Subgraph::BlockedShapeVector input_blocked_shapes;
    Subgraph::BlockedShapeVector output_blocked_shapes;
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph

