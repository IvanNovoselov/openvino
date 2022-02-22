// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <thread>

#include "pugixml.hpp"

#include <openvino/pass/serialize.hpp>
#include <ngraph/opsets/opset.hpp>

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
void SnippetsTestsCommon::validateNumSubgraphs() {
    const auto& compiled_model = compiledModel.get_runtime_model();
    // todo: remove graph serialization when converting to regular PR
    {
        auto m = ov::clone_model(*compiled_model);
        ov::pass::Serialize("compiled.xml", "compiled.bin").run_on_model(m);
    }
    size_t num_subgraphs = 0;
    size_t num_nodes = 0;
    for (const auto &op : compiled_model->get_ops()) {
        if (ov::is_type<ov::op::v0::Parameter>(op) ||
            ov::is_type<ov::op::v0::Constant>(op) ||
            ov::is_type<ov::op::v0::Result>(op))
            continue;

        auto &rt = op->get_rt_info();
        const auto rinfo = rt.find("layerType");
        ASSERT_NE(rinfo, rt.end()) << "Failed to find layerType in " << op->get_friendly_name()
                                   << "rt_info.";
        const std::string layerType = rinfo->second.as<std::string>();
        num_subgraphs += layerType == "Subgraph";
        num_nodes++;
    }
    ASSERT_EQ(ref_num_nodes, num_nodes)
                                << "Compiled model contains invalid number of nodes.";
    ASSERT_EQ(ref_num_subgraphs, num_subgraphs)
                                << "Compiled model contains invalid number of subgraphs.";
}

}  // namespace test
}  // namespace ov
