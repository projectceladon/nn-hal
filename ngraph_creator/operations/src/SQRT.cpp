#include <SQRT.hpp>
#undef LOG_TAG
#define LOG_TAG "SQRT"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

SQRT::SQRT(int operationIndex, GraphMetadata graphMetadata ) : OperationsBase(operationIndex, graphMetadata ) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> SQRT::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);

    auto outputNode = std::make_shared<ov::opset3::Sqrt>(input);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
