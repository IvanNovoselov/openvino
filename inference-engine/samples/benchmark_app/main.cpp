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
#include <iostream>
int main(){
    InferenceEngine::Core ie(std::string("/usr/local/lib64/plugins.xml"));
    //InferenceEngine::Core ie;
    //auto network = ie.ReadNetwork("/tmp/ngraph_ir.xml", "/tmp/ngraph_ir.bin");
    auto network = ie.ReadNetwork("ngraph_ir.xml", "ngraph_ir.bin");
    std::cout << "Reading network is done: " << network.getName() << std::endl;
    auto mExecutableNw = ie.LoadNetwork(network, "CPU");
    std::cout << "Loading network is done: " << mExecutableNw.GetExecGraphInfo().getName()<< std::endl;
    return 0;
}