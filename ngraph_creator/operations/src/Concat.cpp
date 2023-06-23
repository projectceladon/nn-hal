#include <Concat.hpp>
#undef LOG_TAG
#define LOG_TAG "Concat"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Concat::Concat(int operationIndex, GraphMetadata graphMetadata)
    : OperationsBase(operationIndex, graphMetadata) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Concat::validate() {
    // check concatenation axis
    auto n = mOpModelInfo->getOperationInputsSize(mNnapiOperationIndex) -
             1;  // 0 ~ n-1: The list of n input tensors
    for (size_t i = 0; i < n; i++) {
        if (!isValidInputTensor(i)) {
            ALOGE("%s Invalid dimensions for input", __func__);
            return false;
        }
    }
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> Concat::createNode() {
    auto n = mOpModelInfo->getOperationInputsSize(mNnapiOperationIndex) -
             1;  // 0 ~ n-1: The list of n input tensors
    auto axis = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex,
                                                            n);  // n: concatenation axis
    std::vector<ov::Output<ov::Node>> inputs;
    ALOGV("createNode n %lu, axis %d", n, axis);
    for (size_t i = 0; i < n; i++) {
        auto inputIndex = mOpModelInfo->getOperationInput(mNnapiOperationIndex, i);
        auto inputOp = getInputNode(i);
        const auto op = mOpModelInfo->getOperand(inputIndex);
        ALOGV("createNode inputIndex %d, lifetime %d", inputIndex, op.lifetime);
        inputs.push_back(inputOp);
    }

    std::shared_ptr<ov::Node> outputNode = std::make_shared<ov::opset3::Concat>(inputs, axis);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
