#include <Quantize.hpp>
#undef LOG_TAG
#define LOG_TAG "Quantize"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Quantize::Quantize(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

void Quantize::connectOperationToGraph() { createNode(); }

std::shared_ptr<ov::Node> Quantize::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);
    ov::element::Type elementType;
    const auto& outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
    if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        elementType = ov::element::u8;
    } else if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM_SIGNED)) {
        elementType = ov::element::i8;
    } else {
        ALOGE("Invalid Output Operand Type %d",sModelInfo->getOperandType(outputIndex));
    }
    auto outputNode = QuantizeNode(input, outputIndex, elementType);

    if (outputNode != nullptr)
    {
        mNgraphNodes->setOutputAtOperandIndex(outputIndex, outputNode);
    }
    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
