#include <Div.hpp>
#undef LOG_TAG
#define LOG_TAG "Div"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Div::Div(int operationIndex, GraphMetadata graphMetadata)
    : OperationsBase(operationIndex, graphMetadata) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Div::validate() {
    if (!isValidInputTensor(0) || !isValidInputTensor(1)) {
        ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}
std::shared_ptr<ov::Node> Div::createNode() {
    // Creating input nodes
    auto input1 = getInputNode(0);
    auto input2 = getInputNode(1);

    auto activationFn = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);

    auto DivNode =
        std::make_shared<ov::opset3::Divide>(input1, input2, ov::op::AutoBroadcastType::NUMPY);

    auto outputNode = applyActivation(DivNode, activationFn);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
