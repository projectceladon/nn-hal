#include "IENetwork.h"
#include "ie_common.h"
#include <ie_blob.h>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>

#undef LOG_TAG
#define LOG_TAG "IENetwork"

namespace android::hardware::neuralnetworks::nnhal {

bool IENetwork::loadNetwork(const std::string& ir_xml, const std::string& ir_bin) {
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

    ALOGD("Creating infer request for Intel Device Type : %s", deviceStr.c_str());

    if (mNetwork) {
        compiled_model = ie.compile_model(mNetwork, deviceStr);
        ALOGD("loadNetwork is done....");
#if __ANDROID__
        ov::serialize(mNetwork, ir_xml, ir_bin,
                        ov::pass::Serialize::Version::IR_V11);
#else
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::Serialize>("/tmp/model.xml", "/tmp/model.bin");
        manager.run_passes(mNetwork);
#endif
        std::vector<ov::Output<ov::Node>> modelInput = mNetwork->inputs();
        mInferRequest = compiled_model.create_infer_request();
        ALOGD("CreateInferRequest is done....");

    } else {
        ALOGE("Invalid Network pointer");
        return false;
    }

    return true;
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
