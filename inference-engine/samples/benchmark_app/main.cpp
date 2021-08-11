// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//#include <ie_cnn_network.h>
#include <ie_core.hpp>
//#include <ie_executable_network.hpp>
//#include <ie_infer_request.hpp>
#include <ie_input_info.hpp>
#include <vector>
#include "ie_common.h"
#include "IENetwork.h"
#include <iostream>


namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

int test() {
    InferenceEngine::Core ie(plugin_path);
    InferenceEngine::CNNNetwork deserialized_net = ie.ReadNetwork(xml_path, bin_path);
    auto shared_net = std::shared_ptr<InferenceEngine::CNNNetwork>(&deserialized_net);
    auto mPlugin = std::make_shared<IENetwork>(shared_net);
    mPlugin->loadNetwork();

//    InferenceEngine::Core ie1(plugin_path);
//    InferenceEngine::CNNNetwork deserialized_net = ie1.ReadNetwork("ngraph_ir.xml", "ngraph_ir.bin");
//    //mPlugin = std::make_shared<IENetwork>(ngraph_net);
//    auto mPlugin = std::make_shared<IENetwork>(std::shared_ptr<InferenceEngine::CNNNetwork>(&deserialized_net));
//
//    InferenceEngine::Core ie(std::string(plugin_path));
//    //InferenceEngine::Core ie;
//    //auto network = ie.ReadNetwork("/tmp/ngraph_ir.xml", "/tmp/ngraph_ir.bin");
//    auto network = ie.ReadNetwork("ngraph_ir.xml", "ngraph_ir.bin");
//    std::cout << "Reading network is done: " << network.getName() << std::endl;
//    auto mExecutableNw = ie.LoadNetwork(network, "CPU");
//    std::cout << "Loading network is done: " << mExecutableNw.GetExecGraphInfo().getName()<< std::endl;
    return 0;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

int main(){
    return android::hardware::neuralnetworks::nnhal::test();
}