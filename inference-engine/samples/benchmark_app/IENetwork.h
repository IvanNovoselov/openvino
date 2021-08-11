//#ifndef __DEVICE_PLUGIN_H
//#define __DEVICE_PLUGIN_H

//#include <ie_cnn_network.h>
#include <ie_core.hpp>
//#include <ie_executable_network.hpp>
//#include <ie_infer_request.hpp>
#include <ie_input_info.hpp>
#include <vector>

//#include "utils.h"
// #include "ie_blob.h"
// #include "ie_common.h"
// #include "ie_core.hpp"
// #include "inference_engine.hpp"

#ifndef plugin_path
#define plugin_path  "/usr/local/lib64/plugins.xml"
//"plugins.xml"
#define xml_path  "/tmp/ngraph_ir.xml"
//"ngraph_ir.xml"
#define bin_path  "/tmp/ngraph_ir.bin"
//"ngraph_ir.bin"
#endif


namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class IIENetwork {
public:
    virtual ~IIENetwork() {}
    virtual bool loadNetwork() = 0;
    virtual InferenceEngine::InferRequest getInferRequest() = 0;
    virtual void infer() = 0;
    virtual void queryState() = 0;
    virtual InferenceEngine::TBlob<float>::Ptr getBlob(const std::string& outName) = 0;
    virtual void prepareInput(InferenceEngine::Precision precision,
                              InferenceEngine::Layout layout) = 0;
    virtual void prepareOutput(InferenceEngine::Precision precision,
                               InferenceEngine::Layout layout) = 0;
    virtual void setBlob(const std::string& inName,
                         const InferenceEngine::Blob::Ptr& inputBlob) = 0;
};

// Abstract this class for all accelerators
class IENetwork : public IIENetwork {
private:
    std::shared_ptr<InferenceEngine::CNNNetwork> mNetwork;
    InferenceEngine::ExecutableNetwork mExecutableNw;
    InferenceEngine::InferRequest mInferRequest;
    InferenceEngine::InputsDataMap mInputInfo;
    InferenceEngine::OutputsDataMap mOutputInfo;

public:
    IENetwork() : IENetwork(nullptr) {}
    IENetwork(std::shared_ptr<InferenceEngine::CNNNetwork> network) : mNetwork(network) {}

    virtual bool loadNetwork();
    void prepareInput(InferenceEngine::Precision precision, InferenceEngine::Layout layout);
    void prepareOutput(InferenceEngine::Precision precision, InferenceEngine::Layout layout);
    void setBlob(const std::string& inName, const InferenceEngine::Blob::Ptr& inputBlob);
    InferenceEngine::TBlob<float>::Ptr getBlob(const std::string& outName);
    InferenceEngine::InferRequest getInferRequest() { return mInferRequest; }
    void queryState() {}
    void infer();
};
template <typename T, typename S>
std::shared_ptr<T> As(const std::shared_ptr<S>& src) {
    return std::static_pointer_cast<T>(src);
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
//#endif