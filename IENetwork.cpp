#include "IENetwork.h"
#include <ie_blob.h>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include "ie_common.h"

#undef LOG_TAG
#define LOG_TAG "IENetwork"

namespace android::hardware::neuralnetworks::nnhal {

bool IENetwork::createNetwork(std::shared_ptr<ov::Model> network, const std::string& ir_xml,
                              const std::string& ir_bin) {
    ALOGV("%s", __func__);

#if __ANDROID__
    ov::Core ie(std::string("/vendor/etc/openvino/plugins.xml"));
#else
    ov::Core ie(std::string("/usr/local/lib64/plugins.xml"));
#endif
    std::map<std::string, std::string> config;
    std::string deviceStr;
    switch (mTargetDevice) {
        case IntelDeviceType::GNA:
            deviceStr = "GNA";
            break;
        case IntelDeviceType::VPU:
            deviceStr = "VPUX";
            break;
        case IntelDeviceType::CPU:
        default:
            deviceStr = "CPU";
            break;
    }

    ALOGD("creating infer request for Intel Device Type : %s", deviceStr.c_str());

    if (!network) {
        ALOGE("Invalid Network pointer");
        return false;
    } else {
        ie.set_property(deviceStr, {{ov::hint::inference_precision.name(), "f32"}});
        ov::CompiledModel compiled_model = ie.compile_model(network, deviceStr);
        ALOGD("createNetwork is done....");
#if __ANDROID__
        ov::serialize(network, ir_xml, ir_bin, ov::pass::Serialize::Version::IR_V11);
#else
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::Serialize>("/tmp/model.xml", "/tmp/model.bin");
        manager.run_passes(network);
#endif
    }

    return true;
}

void IENetwork::loadNetwork(const std::string& modelName) {
#if __ANDROID__
    ov::Core ie(std::string("/vendor/etc/openvino/plugins.xml"));
#else
    ov::Core ie(std::string("/usr/local/lib64/plugins.xml"));
#endif

    std::string deviceStr;
    switch (mTargetDevice) {
        case IntelDeviceType::GNA:
            deviceStr = "GNA";
            break;
        case IntelDeviceType::VPU:
            deviceStr = "VPUX";
            break;
        case IntelDeviceType::CPU:
        default:
            deviceStr = "CPU";
            break;
    }

    ALOGD("loading infer request for Intel Device Type : %s", deviceStr.c_str());

    ie.set_property(deviceStr, {{ov::hint::inference_precision.name(), "f32"}});
    ov::CompiledModel compiled_model = ie.compile_model(modelName, deviceStr);
    mInferRequest = compiled_model.create_infer_request();
    isLoaded = true;
    ALOGD("Load InferRequest is done....");
}

// Need to be called before loadnetwork.. But not sure whether need to be called for
// all the inputs in case multiple input / output
ov::Tensor IENetwork::getTensor(const std::string& outName) {
    return mInferRequest.get_tensor(outName);
}

ov::Tensor IENetwork::getInputTensor(const std::size_t index) {
    return mInferRequest.get_input_tensor(index);
}

ov::Tensor IENetwork::getOutputTensor(const std::size_t index) {
    return mInferRequest.get_output_tensor(index);
}

void IENetwork::infer() {
    ALOGI("infer requested");
    mInferRequest.infer();
    ALOGI("infer completed");
}

}  // namespace android::hardware::neuralnetworks::nnhal
