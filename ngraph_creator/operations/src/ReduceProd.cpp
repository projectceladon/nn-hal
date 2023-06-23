#include <ReduceProd.hpp>
#undef LOG_TAG
#define LOG_TAG "ReduceProd"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

ReduceProd::ReduceProd(int operationIndex, GraphMetadata graphMetadata)
    : OperationsBase(operationIndex, graphMetadata) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> ReduceProd::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);
    auto reduction_axes = getInputNode(1);
    auto keep_dims = mOpModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);

    auto outputNode = std::make_shared<ov::opset3::ReduceProd>(input, reduction_axes, keep_dims);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
