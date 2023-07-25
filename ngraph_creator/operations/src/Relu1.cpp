#include <Relu1.hpp>
#undef LOG_TAG
#define LOG_TAG "Relu1"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Relu1::Relu1(int operationIndex, GraphMetadata graphMetadata)
    : OperationsBase(operationIndex, graphMetadata) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Relu1::validate() {
    // check Input are of valid dimension or not
    if (!isValidInputTensor(0)) {
        ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> Relu1::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;

    input = getInputNode(0);

    std::shared_ptr<ov::Node> outputNode;

    outputNode = std::make_shared<ov::opset3::Clamp>(input, -1, 1);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
