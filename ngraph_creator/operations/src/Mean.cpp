#include <Mean.hpp>
#undef LOG_TAG
#define LOG_TAG "Mean"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Mean::Mean(int operationIndex, GraphMetadata graphMetadata)
    : OperationsBase(operationIndex, graphMetadata) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Mean::validate() {
    // TODO: Add Support for all_tensors_as_inputs
    const auto& axesOperandIndex = mOpModelInfo->getOperationInput(mNnapiOperationIndex, 1);

    if (!mOpModelInfo->isOperandLifeTimeConst(axesOperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    return true;
}

std::shared_ptr<ov::Node> Mean::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;

    input = getInputNode(0);

    auto reduction_axes = getInputNode(1);
    auto reduce_dims = mOpModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 2);
    bool keep_dims = (reduce_dims > 0) ? true : false;

    std::shared_ptr<ov::Node> outputNode;
    outputNode = std::make_shared<ov::opset3::ReduceMean>(input, reduction_axes, keep_dims);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
