// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include <format_reader_ptr.h>
// clang-format on

/**
 * @brief Main with support Unicode paths, wide strings
 */
int tmain(int argc, tchar* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 2) {
            slog::info << "Usage : " << argv[0] << " <path_to_model>" << slog::endl;
            return EXIT_FAILURE;
        }

        const std::string args = TSTRING2STRING(argv[0]);
        const std::string model_path = TSTRING2STRING(argv[1]);

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

           // -------- Step 2. Read a model --------
        slog::info << "## Loading model files is: " << model_path << slog::endl;
        std::shared_ptr<ov::Model> model0 = core.read_model(model_path);
        std::shared_ptr<ov::Model> model1 = core.read_model(model_path);
        auto compileModel = [&](std::shared_ptr<ov::Model> aModel, std::string devName) -> int {

            for (int i = 0; i < 100; i++) {
                std::cout << "Model" << &aModel << " "<< std::endl;
                ov::CompiledModel compiled_model =
                    core.compile_model(aModel,
                                       devName,
                                       ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
                std::cout << "device " << devName << " compile model finished: " << i << " " <<std::endl;
            }
            return 0;
        };
        //std::string aStrDevName[] = {"GPU.0", "GPU.0"};
        std::vector<std::string> aStrDevName = {"CPU", "CPU"};
        std::shared_ptr<ov::Model> aModels[] = {model0, model1};
        for (int j = 0; j < 2; j++) {
            std::cout << "iteration " << j << "th" << std::endl;
            std::vector<std::thread> threads;
            for (int i = 0; i < aStrDevName.size(); i++) {
                threads.push_back(std::thread(compileModel, aModels[i], aStrDevName[i]));
            }
            for (auto& iter : threads) {
                if (iter.joinable())
                    iter.join();
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "### exit success ###" << std::endl;
    return EXIT_SUCCESS;
}