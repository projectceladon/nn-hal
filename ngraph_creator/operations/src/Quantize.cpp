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
    const auto& outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
    auto outputNode = QuantizeNode(input, outputIndex, ov::element::u8);

    mNgraphNodes->setOutputAtOperandIndex(outputIndex, outputNode);

    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
