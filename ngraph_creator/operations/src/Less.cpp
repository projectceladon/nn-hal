#include <Less.hpp>
#undef LOG_TAG
#define LOG_TAG "Less"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Less::Less(int operationIndex, GraphMetadata graphMetadata)
    : OperationsBase(operationIndex, graphMetadata) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Less::validate() {
    // Check for zero sized or zero dimension tesnors
    if (!isValidInputTensor(0) || !isValidInputTensor(1)) {
        ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> Less::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input1, input2;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    std::shared_ptr<ov::Node> outputNode;

    outputNode =
        std::make_shared<ov::opset3::Less>(input1, input2, ov::op::AutoBroadcastType::NUMPY);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
