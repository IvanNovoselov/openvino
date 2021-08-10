// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "threading/ie_itask_executor.hpp"

#include <future>
#include <memory>
#include <utility>
#include <vector>
#include <iostream>

namespace InferenceEngine {

void ITaskExecutor::runAndWait(const std::vector<Task>& tasks) {
    std::vector<std::packaged_task<void()>> packagedTasks;
    std::vector<std::future<void>> futures;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        std::cerr << "runAndWait FILL:" << i << std::endl;
        packagedTasks.emplace_back([&tasks, i] {tasks[i]();});
        futures.emplace_back(packagedTasks.back().get_future());
    }
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        std::cerr << "runAndWait RUN:" << i << std::endl;
        run([&packagedTasks, i]{packagedTasks[i]();});
    }
    // std::future::get will rethrow exception from task.
    // We should wait all tasks before any exception is thrown.
    // So wait() and get() for each future moved to separate loops
    int i = 0;
    for (auto&& future : futures) {
        std::cerr << "runAndWait wait:" << i++ << std::endl;
        future.wait();
    }
    std::cerr << __FILE__ << " | passed on line | " << __LINE__ << std::endl;
    i = 0;
    for (auto&& future : futures) {
        std::cerr << "runAndWait get:" << i++ << std::endl;
        future.get();
    }
}
}  // namespace InferenceEngine
