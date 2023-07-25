#include <ReduceSum.hpp>
#undef LOG_TAG
#define LOG_TAG "ReduceSum"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

ReduceSum::ReduceSum(int operationIndex, GraphMetadata graphMetadata)
    : OperationsBase(operationIndex, graphMetadata) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool ReduceSum::validate() {
    // check Input are of valid dimension or not
    if (!isValidInputTensor(0) || !isValidInputTensor(1)) {
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

std::shared_ptr<ov::Node> ReduceSum::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);
    auto reduction_axes = getInputNode(1);
    auto keep_dims = mOpModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);

    auto outputNode = std::make_shared<ov::opset3::ReduceSum>(input, reduction_axes, keep_dims);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
