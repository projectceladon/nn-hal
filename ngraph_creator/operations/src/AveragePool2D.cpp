#include <AveragePool2D.hpp>
#undef LOG_TAG
#define LOG_TAG "AveragePool2D"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

AveragePool2D::AveragePool2D(int operationIndex, GraphMetadata graphMetadata ) : OperationsBase(operationIndex, graphMetadata ) {
    mDefaultOutputIndex = mOpModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool AveragePool2D::validate() {
    // Check Input Dimension size
    const auto& inputDimensionsSize = getInputOperandDimensions(0).size();
    if (inputDimensionsSize != 4) {
        ALOGE("%s Invalid dimensions size for input(%lu)", __func__, inputDimensionsSize);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> AveragePool2D::createNode() {
    std::shared_ptr<ov::Node> inputNode;
    const auto& inDims = getInputOperandDimensions(0);
    const auto& inputsSize = mOpModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    inputNode = getInputNode(0);

    ALOGD("%s inputsSize %lu", __func__, inputsSize);

    bool isImplicit = false, isExplicit = false;

    int32_t layout = 0;
    bool useNchw = false;
    int32_t padding_scheme;
    std::vector<size_t> pad_begin;
    std::vector<size_t> pad_end;
    std::vector<size_t> strides;
    std::vector<size_t> kernel;
    int32_t padding_left, padding_right;
    int32_t padding_top, padding_bottom;
    int32_t stride_width, stride_height;
    int32_t filter_width, filter_height;
    int32_t input_width, input_height;
    int32_t activationFn;
    ov::op::PadType auto_pad;

    if (inputsSize >= 10 && inputsSize <= 11) {
        isExplicit = true;
    } else if (inputsSize >= 7 && inputsSize <= 8) {
        isImplicit = true;
    } else {
        ALOGE("%s inputsSize %lu NOT SUPPORTED", __func__, inputsSize);
        return inputNode;
    }

    if (isExplicit) {
        padding_left = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 1);
        padding_right = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);
        padding_top = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);
        padding_bottom = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);

        stride_width = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);
        stride_height = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        filter_width = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 7);
        filter_height = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 8);

        activationFn = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 9);

        if (inputsSize == 11) {
            layout = mOpModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 10);
        }

        if (layout) useNchw = true;

        auto_pad = ov::op::PadType::EXPLICIT;
    }

    if (isImplicit) {
        padding_scheme = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 1);

        stride_width = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);
        stride_height = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);

        filter_width = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);
        filter_height = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);

        activationFn = mOpModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        if (inputsSize == 8) {
            layout = mOpModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 7);
        }

        if (layout) useNchw = true;

        if (useNchw) {
            input_width = inDims[3];
            input_height = inDims[2];
        } else {
            input_width = inDims[2];
            input_height = inDims[1];
        }

        if (padding_scheme == 1) {
            calculateExplicitPadding(input_width, stride_width, filter_width, 1, &padding_left,
                                     &padding_right);
            calculateExplicitPadding(input_height, stride_height, filter_height, 1, &padding_top,
                                     &padding_bottom);

            auto_pad = ov::op::PadType::SAME_UPPER;

        } else {
            padding_left = 0;
            padding_right = 0;
            padding_top = 0;
            padding_bottom = 0;
            auto_pad = ov::op::PadType::VALID;
        }
    }

    if (!useNchw) {  // No conversion needed if useNchw set
        inputNode = transpose(NHWC_NCHW, inputNode);
    }

    strides = {(size_t)stride_height, (size_t)stride_width};
    pad_begin = {(size_t)padding_top, (size_t)padding_left};
    pad_end = {(size_t)padding_bottom, (size_t)padding_right};
    kernel = {(size_t)filter_height, (size_t)filter_width};

    std::shared_ptr<ov::Node> outputNode = std::make_shared<ov::opset3::AvgPool>(
        inputNode, ov::Strides(strides), ov::Shape(pad_begin), ov::Shape(pad_end),
        ov::Shape(kernel), true, ov::op::RoundingType::FLOOR, auto_pad);

    outputNode = applyActivation(outputNode, activationFn);

    if (!useNchw) {
        outputNode = transpose(NCHW_NHWC, outputNode);
    }

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
