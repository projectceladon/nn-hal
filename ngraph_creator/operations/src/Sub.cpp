#include <Sub.hpp>
#undef LOG_TAG
#define LOG_TAG "Sub"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Sub::Sub(int operationIndex, GraphMetadata graphMetadata)
    : OperationsBase(operationIndex, graphMetadata) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Sub::validate() {
    if (!isValidInputTensor(0) || !isValidInputTensor(1)) {
        ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> Sub::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input1, input2;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    auto activationFn = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);

    auto subNode =
        std::make_shared<ov::opset3::Subtract>(input1, input2, ov::op::AutoBroadcastType::NUMPY);

    auto outputNode = applyActivation(subNode, activationFn);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
