#include <Div.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Div::Div(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Div::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    // Check all input types
    for (int i = 0; i <= 1; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    if (!checkInputOperandType(2, (int32_t)OperandType::INT32)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Div::createNode() {
    // Creating input nodes
    auto input1 = getInputNode<float>(0);
    auto input2 = getInputNode<float>(1);

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);

    auto DivNode = std::make_shared<ngraph::opset3::Divide>(input1, input2,
                                                            ngraph::op::AutoBroadcastType::NUMPY);

    auto outputNode = applyActivation(DivNode, activationFn);

    const auto op = sModelInfo->getOperand(mDefaultOutputIndex);
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(mDefaultOutputIndex, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android