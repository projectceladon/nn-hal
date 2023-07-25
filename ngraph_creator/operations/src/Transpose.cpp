#include <Transpose.hpp>
#undef LOG_TAG
#define LOG_TAG "Transpose"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Transpose::Transpose(int operationIndex, GraphMetadata graphMetadata)
    : OperationsBase(operationIndex, graphMetadata) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Transpose::validate() {
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

    const auto& inputsSize = mOpModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    if (inputsSize == 2) {
        const auto& dimsOperandIndex = mOpModelInfo->getOperationInput(mNnapiOperationIndex, 1);
        if (!isValidInputTensor(1) || !mOpModelInfo->isOperandLifeTimeConst(dimsOperandIndex)) {
            ALOGE("%s Invalid operand type or operand lifetime", __func__);
            return false;
        }
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> Transpose::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;

    input = getInputNode(0);

    std::shared_ptr<ov::Node> order;

    const auto& dims = getInputOperandDimensions(1);
    if (!dims.empty() && dims[0] != 0) {
        order = getInputNode(1);
    } else {
        order = createConstNode(ov::element::i32, {0}, convertToVector(0));
    }

    std::shared_ptr<ov::Node> outputNode;

    outputNode = std::make_shared<ov::opset3::Transpose>(input, order);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
