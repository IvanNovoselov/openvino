#define LOG_TAG "IENetwork"
#include "IENetwork.h"
#include "ie_common.h"

//#include <android-base/logging.h>
//#include <android/log.h>
#include <ie_blob.h>
//#include <log/log.h>
#include <iostream>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
bool IENetwork::loadNetwork() {
    InferenceEngine::Core ie(plugin_path);
    std::map<std::string, std::string> config;

    try {
        auto network = ie.ReadNetwork(xml_path, bin_path);
        std::cerr << "Readnetwork is successfull...." << std::endl;
        
        if (mNetwork) {
            //mExecutableNw = ie.LoadNetwork(*mNetwork, "CPU");
            mExecutableNw = ie.LoadNetwork(network, "CPU");
            std::cout << "LoadNetwork is done...." << std::endl;
            mInferRequest = mExecutableNw.CreateInferRequest();
            std::cout << "CreateInfereRequest is done...." << std::endl;

            mInputInfo = mNetwork->getInputsInfo();
            mOutputInfo = mNetwork->getOutputsInfo();

            //#ifdef NN_DEBUG
            for (auto input : mInputInfo) {
                auto dims = input.second->getTensorDesc().getDims();
                for (auto i : dims) {
                    std::cout << " Dimes : %lu" << i << std::endl;
                }
                std::cout << "Name: %s " << input.first.c_str() << std::endl;
            }
            for (auto output : mOutputInfo) {
                auto dims = output.second->getTensorDesc().getDims();
                for (auto i : dims) {
                    std::cout << " Dimes : %lu" << i << std::endl;
                }
                std::cout << "Name: %s " << output.first.c_str() << std::endl;
            }
            //#endif
        } else {
            std::cerr << "Invalid Network pointer" << std::endl;
            return false;
        }
    } catch (const std::exception& ex) {
        std::cerr << "%s Exception !!!" << ex.what() << std::endl;
    }

    return true;
}

// Need to be called before loadnetwork.. But not sure whether need to be called for
// all the inputs in case multiple input / output
void IENetwork::prepareInput(InferenceEngine::Precision precision, InferenceEngine::Layout layout) {

    auto inputInfoItem = *mInputInfo.begin();
    inputInfoItem.second->setPrecision(precision);
    inputInfoItem.second->setLayout(layout);
}

void IENetwork::prepareOutput(InferenceEngine::Precision precision,
                              InferenceEngine::Layout layout) {
    InferenceEngine::DataPtr& output = mOutputInfo.begin()->second;
    output->setPrecision(precision);
    output->setLayout(layout);
}

void IENetwork::setBlob(const std::string& inName, const InferenceEngine::Blob::Ptr& inputBlob) {
    std::cout << "setBlob input or output blob name : " << inName.c_str() << std::endl;
    mInferRequest.SetBlob(inName, inputBlob);
}

InferenceEngine::TBlob<float>::Ptr IENetwork::getBlob(const std::string& outName) {
    InferenceEngine::Blob::Ptr outputBlob;
    outputBlob = mInferRequest.GetBlob(outName);
    return android::hardware::neuralnetworks::nnhal::As<InferenceEngine::TBlob<float>>(outputBlob);
}

void IENetwork::infer() {
    std::cout << "Infer Network\n";
    mInferRequest.StartAsync();
    mInferRequest.Wait(10000);
    std::cout << "infer request completed\n";
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
