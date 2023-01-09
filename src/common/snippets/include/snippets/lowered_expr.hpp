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

class LoweringConfig {
public:
    // True if the lowered Emitters need to be accessed during runtime. Normally they're destroyed after code emission.
    bool m_save_lowered_code = false;
    // True if we can optimize tails for single evaluation during code generation
    // More details with optimization examples you can see in generate() method
    // For example, tails with Buffer ops doesn't support single evaluation optimizations
    //              because of that we should always reset memory pointer using finalization offsets
    //              after data storing to Buffer
    bool m_optimize_single_evaluation = true;
    // True if we should check runtime info for nodes to call specific needed transformations
    bool m_need_fill_tail_register = false;
};

/**
 * @interface Emitter
 * @brief Base class for all target specific code emitters used by generator.
 * @ingroup snippets
 */
class LoweredExprIR;
class LoweredExpr {
    friend LoweredExprIR;
public:
    /**
     * @brief Default constructor
     */
    LoweredExpr(const std::shared_ptr<Node>& n);
    LoweredExpr() = default;
    virtual ~LoweredExpr() = default;
    // todo: shall we return pointers to const?
    std::shared_ptr<Node> get_node() const {return  m_source_node;}
    std::shared_ptr<Emitter> get_emitter() const;
    void init_emitter(const std::shared_ptr<const TargetMachine>& target);
    RegInfo get_reg_info() const {return  m_reg_info;}
    void set_reg_info(RegInfo rinfo) {m_reg_info = std::move(rinfo);}
    static ngraph::snippets::RegInfo getRegisters(const std::shared_ptr<const Node>& n);

protected:
    // todo: const pointer to const node?
    std::shared_ptr<Node> m_source_node{nullptr};
    std::shared_ptr<Emitter> m_emitter{nullptr};
    RegInfo m_reg_info{{}, {}};
};

class IOLoweredExpr : public LoweredExpr {
public:
    enum class io_type {INPUT, OUTPUT, UNDEFINED};
    IOLoweredExpr(const std::shared_ptr<Node>& n, int64_t index);
    IOLoweredExpr(const std::shared_ptr<Node>& n, int64_t index, io_type type);
    int64_t get_index() const  {return m_index;}
    io_type get_type() const {return m_type; }
private:
    int64_t m_index = -1;
    io_type m_type = io_type::UNDEFINED;
};

class LoweredExprIR {
public:
    using container = std::list<std::shared_ptr<LoweredExpr>>;
    /**
     * @brief Default constructor
     */
    LoweredExprIR(const std::shared_ptr<ov::Model>& m, LoweringConfig config = {});
    LoweredExprIR() = default;
    LoweredExprIR deep_copy() const;
//    LoweredExprIR(std::vector<std::shared_ptr<ov::Node>> ops, std::shared_ptr<TargetMachine> target);

//    LoweredExprIR(std::vector<std::shared_ptr<ov::Node>> vector1, std::shared_ptr<TargetMachine> sharedPtr);
    container& get_ops() {return m_lowered_ops; }
    const container& get_ops() const {return m_lowered_ops; }
    void init_emitters(const std::shared_ptr<TargetMachine>& target);
    LoweringConfig get_config() {return m_config; }

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
    container m_lowered_ops{};
    LoweringConfig m_config{};
};

using AllocatedEmitter = std::pair<std::shared_ptr<Emitter>, RegInfo>;

} // namespace snippets
} // namespace ngraph