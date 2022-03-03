// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/model.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
using ov::Model;
// todo: This transformation is required to pass ngraph::pass::CheckUniqueNames executed from TransformationTestsF
//  Note that other solutions are also possible:
//  - Modify TransformationTestsF to make it possible to disable CheckUniqueNames on demand.
//      Not favorable because several other checks inside CheckUniqueNames are quite useful (solution: modify CheckUniqueNames also?).
//  - Write our own class similar to TransformationTestsF, but a bit more flexible.
//      Not favorable because of code duplication.
//  - Slightly modify TokenizeSnippets subgraph naming policy. It currently tries to mimic the one used by MKLDNNPlugin
//      (node name before result may be changed, but tensor name must be preserved), but it's easy to update Subgraph name
//      (on demand) if it goes before the Result.
//      Not favorable because it seems that this alternative-naming-behavior is needed only to pass these tests.
//  - An alternative solution is to insert dummy non-tokenizable node before the Result (Such as Convert)
//      Not favorable because we won't be able to test that the tensor names are preserved for the Result inputs.
class SnippetsRestoreResultInputName : public ngraph::pass::MatcherPass {
public:
    SnippetsRestoreResultInputName();
};

class SnippetsCollapseSubgraphTests : public TransformationTestsF {
public:
    virtual void run();
};

/// \brief Base class for snippets-related subgraphs. Note that inputShapes size is model-specific
/// and expected to be checked inside a child class constructor.
/// \param inputShapes vector of input shapes accepted by the underlying model
class SnippetsFunctionBase {
public:
    SnippetsFunctionBase() = delete;

    explicit SnippetsFunctionBase(std::vector<Shape> &inputShapes) : input_shapes{inputShapes} {};

    std::shared_ptr<Model> getReference() const {
        std::shared_ptr<Model> function_ref = initReference();
        validate_function(function_ref);
        return function_ref;
    }

    std::shared_ptr<Model> getOriginal() const {
        std::shared_ptr<Model> function = initOriginal();
        validate_function(function);
        return function;
    }

    std::shared_ptr<Model> getLowered() const {
        std::shared_ptr<Model> function_low = initLowered();
        validate_function(function_low);
        return function_low;
    }

    size_t getNumInputs() const { return input_shapes.size(); }

protected:
    virtual std::shared_ptr<Model> initOriginal() const = 0;

    virtual std::shared_ptr<Model> initReference() const {
        IE_THROW(NotImplemented) << "initReference() for this class is not implemented";
    }

    virtual std::shared_ptr<Model> initLowered() const {
        IE_THROW(NotImplemented) << "initLowered() for this class is not implemented";
    }

    // only fp32 is currently supported by snippets
    ov::element::Type_t precision = element::f32;
    std::vector<Shape> input_shapes;

    virtual void validate_function(const std::shared_ptr<Model> &f) const;
};

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
                                 std::vector<size_t>&& customOpsNumInputs);

protected:
    std::vector<std::shared_ptr<Node>> custom_ops;
    std::vector<size_t> custom_ops_num_inputs;
    void ResetCustomOpsInputs();
};
} // namespace subgraph
} // namespace builder
} // namespace ngraph