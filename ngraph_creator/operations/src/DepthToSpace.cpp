#include <DepthToSpace.hpp>
#undef LOG_TAG
#define LOG_TAG "DepthToSpace"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

DepthToSpace::DepthToSpace(int operationIndex, GraphMetadata graphMetadata ) : OperationsBase(operationIndex, graphMetadata ) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> DepthToSpace::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;
    bool useNchw = false;
    const auto& inputsSize = mOpModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    if (inputsSize == 3) {
        auto layout = mOpModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);
        if (layout) useNchw = true;
    }

    input = getInputNode(0);
    auto block_size = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 1);

    if (!useNchw)  // No conversion needed if useNchw set
        input = transpose(NHWC_NCHW, input);

    std::shared_ptr<ov::Node> outputNode;

    outputNode = std::make_shared<ov::opset3::DepthToSpace>(
        input, ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, block_size);

    if (!useNchw) outputNode = transpose(NCHW_NHWC, outputNode);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
