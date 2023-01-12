// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <openvino/core/node.hpp>
#include "emitter.hpp"
#include "target_machine.hpp"
//#include "snippets/pass/lowered/insert_tail_loop.hpp"

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
    bool m_explicit_loop_insertion = false;
    ov::PartialShape m_master_shape{};
    size_t m_loop_depth = 1;
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
    explicit LoweredExpr(const std::shared_ptr<Node>& n);
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
    using io_container = std::list<std::shared_ptr<IOLoweredExpr>>;
    using exprIt = container::iterator;
    using constExprIt = container::const_iterator;
    /**
     * @brief Default constructor
     */
    explicit LoweredExprIR(const std::shared_ptr<ov::Model>& m, LoweringConfig config = {});
    LoweredExprIR() = default;
    LoweredExprIR deep_copy() const;
    static LoweredExprIR::container deep_copy_range(LoweredExprIR::container::const_iterator begin, LoweredExprIR::container::const_iterator end);
//    LoweredExprIR(std::vector<std::shared_ptr<ov::Node>> ops, std::shared_ptr<TargetMachine> target);

//    LoweredExprIR(std::vector<std::shared_ptr<ov::Node>> vector1, std::shared_ptr<TargetMachine> sharedPtr);
//    container& get_ops() {return m_lowered_ops; }
    const container& get_ops() const {return m_lowered_ops; }
    const io_container& get_IO_ops() const {return m_io_lowered_ops; }
    void init_emitters(const std::shared_ptr<TargetMachine>& target);
    LoweringConfig get_config() {return m_config; }
    std::vector<PartialShape> get_forced_shapes() const {return m_forcedIOShapes;}
    // todo: We need to check if Result or Parameter is inserted and update m_io_lowered_ops accordingly
    exprIt insert(constExprIt pos, container::value_type&& value);
    exprIt insert(constExprIt pos, const container::value_type& value);
    exprIt insert(constExprIt pos, exprIt begin, exprIt end);
    exprIt insert(constExprIt pos, constExprIt begin, constExprIt end);

    bool empty() const noexcept {return m_lowered_ops.empty(); }
    void debug_print() const;

    container::reference back() noexcept {return m_lowered_ops.back();}
    container::const_reference back() const noexcept {return m_lowered_ops.back();}
    container::reference front() noexcept {return m_lowered_ops.front();}
    container::const_reference front() const noexcept {return m_lowered_ops.front();}

    exprIt begin() noexcept {
        return m_lowered_ops.begin();
    }
    exprIt end() noexcept {
        return m_lowered_ops.end();
    }
    constExprIt begin() const noexcept {
        return cbegin();
    }
    constExprIt end() const noexcept {
        return cend();
    }
    constExprIt cbegin() const noexcept {
        return m_lowered_ops.cbegin();
    }
    constExprIt cend() const noexcept {
        return m_lowered_ops.cend();
    }
    container ::reverse_iterator rbegin() noexcept {
        return m_lowered_ops.rbegin();
    }
    container::reverse_iterator rend() noexcept {
        return m_lowered_ops.rend();
    }
    container::const_reverse_iterator crbegin() const noexcept {
        return m_lowered_ops.crbegin();
    }
    container::const_reverse_iterator crend() const noexcept {
        return m_lowered_ops.crend();
    }

private:
    container m_lowered_ops{};
    io_container m_io_lowered_ops;
    LoweringConfig m_config{};
    std::vector<PartialShape> m_forcedIOShapes{};
};

using AllocatedEmitter = std::pair<std::shared_ptr<Emitter>, RegInfo>;

} // namespace snippets
} // namespace ngraph