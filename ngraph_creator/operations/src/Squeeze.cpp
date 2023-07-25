#include <Squeeze.hpp>
#undef LOG_TAG
#define LOG_TAG "Squeeze"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Squeeze::Squeeze(int operationIndex, GraphMetadata graphMetadata)
    : OperationsBase(operationIndex, graphMetadata) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Squeeze::validate() {
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

    // TODO: Add Support for all_tensors_as_inputs
    const auto& dimsOperandIndex = mOpModelInfo->getOperationInput(mNnapiOperationIndex, 1);

    // TODO: Support OmittedInput.
    // The empty 2nd argument in Squeeze op causes dynamic output
    // To add support, the dims will have to be calculated statically
    if (mOpModelInfo->isOmittedInput(mNnapiOperationIndex, 1) ||
        !mOpModelInfo->isOperandLifeTimeConst(dimsOperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    const auto& inputsSize = mOpModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    if (inputsSize == 2) {
        if (!isValidInputTensor(1)) {
            ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
            return false;
        }
    }

    return true;
}

std::shared_ptr<ov::Node> Squeeze::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;

    input = getInputNode(0);

    std::shared_ptr<ov::Node> dims;

    if (!mOpModelInfo->isOmittedInput(mNnapiOperationIndex, 1))
        dims = getInputNode(1);
    else
        dims = createConstNode(ov::element::i32, {0}, std::vector<int64_t>{});

    std::shared_ptr<ov::Node> outputNode;

    outputNode = std::make_shared<ov::opset3::Squeeze>(input, dims);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
