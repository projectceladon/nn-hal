#include <Sub.hpp>
#undef LOG_TAG
#define LOG_TAG "Sub"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Sub::Sub(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Sub::validate() {
    auto operandIndex1 = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    auto operandIndex2 = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    const auto& elementType1 = sModelInfo->getOperandType(operandIndex1);
    const auto& elementType2 = sModelInfo->getOperandType(operandIndex2);
    if ( !isValidInputTensor(0) || !isValidInputTensor(1) ) {
         ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
         return false;
    }

    // check if both tensors are of same type
    if(elementType1 != elementType2 ) {
        ALOGE("%s Input type mismatch", __func__);
        return false;
    } else if ( elementType1 == OperandType::TENSOR_INT32 ) {
        //In 1.3 For a {@link OperandType::TENSOR_INT32} tensor,
        //the {@link FusedActivationFunc} must be "NONE".
        auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);
        if (activationFn != 0) {
            ALOGE("%s Activation type must be none for TENSOR_INT32 type", __func__);
            return false;
        }
    }
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Sub::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);

    auto subNode = std::make_shared<ngraph::opset3::Subtract>(input1, input2,
                                                              ngraph::op::AutoBroadcastType::NUMPY);

    auto outputNode = applyActivation(subNode, activationFn);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
