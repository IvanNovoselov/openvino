// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <openvino/core/node.hpp>
#include "emitter.hpp"
#include "target_machine.hpp"

namespace ngraph {
namespace snippets {

using code = const uint8_t *;
using RegInfo = std::pair<std::vector<size_t>, std::vector<size_t>>;

/**
 * @interface Emitter
 * @brief Base class for all target specific code emitters used by generator.
 * @ingroup snippets
 */
class LoweredExpr {
public:
    /**
     * @brief Default constructor
     */
    LoweredExpr(const std::shared_ptr<Node>& n, const std::shared_ptr<const TargetMachine>& target);
    LoweredExpr() = default;
    // todo: shall we return pointers to const?
    std::shared_ptr<Node> get_node() {return  m_source_node;}
    std::shared_ptr<const Node> get_node() const {return  m_source_node;}
    std::shared_ptr<Emitter> get_emitter() {return  m_emitter;}
    std::shared_ptr<const Emitter> get_emitter() const {return  m_emitter;}
    RegInfo get_reg_info() const {return  m_reg_info;}
//    void set_reg_info(const RegInfo& rinfo) {m_reg_info = rinfo;}
    void set_reg_info(RegInfo rinfo) {m_reg_info = std::move(rinfo);}

private:
    // todo: const pointer to const node?
    std::shared_ptr<Node> m_source_node{nullptr};
    std::shared_ptr<Emitter> m_emitter{nullptr};
    static ngraph::snippets::RegInfo getRegisters(const std::shared_ptr<const Node>& n);
    RegInfo m_reg_info{{}, {}};
};

class LoweredExprIR {
public:
    using loweredContainer = std::list<LoweredExpr>;
    /**
     * @brief Default constructor
     */
    LoweredExprIR(const std::vector<std::shared_ptr<ov::Node>>& ops, const std::shared_ptr<TargetMachine>& target);
    LoweredExprIR() = default;
//    LoweredExprIR(std::vector<std::shared_ptr<ov::Node>> ops, std::shared_ptr<TargetMachine> target);

//    LoweredExprIR(std::vector<std::shared_ptr<ov::Node>> vector1, std::shared_ptr<TargetMachine> sharedPtr);
    loweredContainer& get_ops() {return m_lowered_ops; }
    const loweredContainer& get_ops() const {return m_lowered_ops; }

    bool empty() const noexcept {return m_lowered_ops.empty(); }
//
//    loweredContainer::iterator begin() noexcept {
//        return m_lowered_ops.begin();
//    }
//    loweredContainer::iterator end() noexcept {
//        return m_lowered_ops.end();
//    }
//    loweredContainer::const_iterator begin() const noexcept {
//        return cbegin();
//    }
//    loweredContainer::const_iterator end() const noexcept {
//        return cend();
//    }
//    loweredContainer::const_iterator cbegin() const noexcept {
//        return m_lowered_ops.cbegin();
//    }
//    loweredContainer::const_iterator cend() const noexcept {
//        return m_lowered_ops.cend();
//    }
//    loweredContainer::reverse_iterator rbegin() noexcept {
//        return m_lowered_ops.rbegin();
//    }
//    loweredContainer::reverse_iterator rend() noexcept {
//        return m_lowered_ops.rend();
//    }
//    loweredContainer::const_reverse_iterator crbegin() const noexcept {
//        return m_lowered_ops.crbegin();
//    }
//    loweredContainer::const_reverse_iterator crend() const noexcept {
//        return m_lowered_ops.crend();
//    }


private:
    loweredContainer m_lowered_ops{};
};

using AllocatedEmitter = std::pair<std::shared_ptr<Emitter>, ngraph::snippets::RegInfo>;

} // namespace snippets
} // namespace ngraph