#include <Argmin.hpp>
#undef LOG_TAG
#define LOG_TAG "Argmin"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Argmin::Argmin(int operationIndex, GraphMetadata graphMetadata ) : OperationsBase(operationIndex, graphMetadata ) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> Argmin::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;

    input = getInputNode(0);

    int32_t axis = mOpModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    ALOGD("createNode axis %d", axis);

    auto k_node = createConstNode(ov::element::i32, {}, convertToVector(1));

    auto topk = std::make_shared<ov::opset3::TopK>(input, k_node, axis, ov::opset3::TopK::Mode::MIN,
                                                   ov::opset3::TopK::SortType::NONE);

    const auto axis_to_remove =
        createConstNode(ov::element::u32, {}, convertToVector(topk->get_axis()));
    auto outputNode = std::make_shared<ov::opset3::Squeeze>(topk->output(1), axis_to_remove);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
