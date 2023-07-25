#include <Logistic.hpp>
#undef LOG_TAG
#define LOG_TAG "Logistic"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Logistic::Logistic(int operationIndex, GraphMetadata graphMetadata)
    : OperationsBase(operationIndex, graphMetadata) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Logistic::validate() {
    // check Input are of valid dimension or not
    if (!isValidInputTensor(0)) {
        ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
        return false;
    }

    const auto& inputDimensionsSize = getInputOperandDimensions(0).size();
    if (inputDimensionsSize > 4) {
        ALOGE("%s Invalid dimensions size for input(%lu)", __func__, inputDimensionsSize);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> Logistic::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;

    input = getInputNode(0);

    std::shared_ptr<ov::Node> outputNode;
    outputNode = std::make_shared<ov::opset3::Sigmoid>(input);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
