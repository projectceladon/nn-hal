#include <OperationsFactory.hpp>
#undef LOG_TAG
#define LOG_TAG "OperationsFactory"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

OperationsFactory::OperationsFactory(IntelDeviceType deviceType,
                                     std::shared_ptr<NnapiModelInfo> modelInfo,
                                     std::shared_ptr<NgraphNodes> nodes)
    : mGraphMetadata{modelInfo, deviceType} {
    ALOGV("%s Constructed", __func__);
}
OperationsFactory::~OperationsFactory() { ALOGV("%s Destructed", __func__); }
std::shared_ptr<OperationsBase> OperationsFactory::getOperation(
    int operationIndex, const OperationType& operationType) {
    switch (operationType) {
        case OperationType::ABS:
            return std::make_shared<Abs>(operationIndex, mGraphMetadata);
        case OperationType::ADD:
            return std::make_shared<Add>(operationIndex, mGraphMetadata);
        case OperationType::ARGMAX:
            return std::make_shared<Argmax>(operationIndex, mGraphMetadata);
        case OperationType::ARGMIN:
            return std::make_shared<Argmin>(operationIndex, mGraphMetadata);
        case OperationType::AVERAGE_POOL_2D:
            return std::make_shared<AveragePool2D>(operationIndex, mGraphMetadata);
        case OperationType::BATCH_TO_SPACE_ND:
            return std::make_shared<BatchToSpace>(operationIndex, mGraphMetadata);
        case OperationType::BIDIRECTIONAL_SEQUENCE_RNN:
            return std::make_shared<BidirectionalSequenceRNN>(operationIndex, mGraphMetadata);
        case OperationType::CAST:
            return std::make_shared<Cast>(operationIndex, mGraphMetadata);
        case OperationType::CHANNEL_SHUFFLE:
            return std::make_shared<ChannelShuffle>(operationIndex, mGraphMetadata);
        case OperationType::CONCATENATION:
            return std::make_shared<Concat>(operationIndex, mGraphMetadata);
        case OperationType::CONV_2D:
            return std::make_shared<Conv2d>(operationIndex, mGraphMetadata);
        case OperationType::DEPTH_TO_SPACE:
            return std::make_shared<DepthToSpace>(operationIndex, mGraphMetadata);
        case OperationType::DEPTHWISE_CONV_2D:
            return std::make_shared<DepthwiseConv2d>(operationIndex, mGraphMetadata);
        case OperationType::DEQUANTIZE:
            return std::make_shared<Dequantize>(operationIndex, mGraphMetadata);
        case OperationType::DIV:
            return std::make_shared<Div>(operationIndex, mGraphMetadata);
        case OperationType::EMBEDDING_LOOKUP:
            return std::make_shared<EmbeddingLookup>(operationIndex, mGraphMetadata);
        case OperationType::EQUAL:
            return std::make_shared<Equal>(operationIndex, mGraphMetadata);
        case OperationType::EXP:
            return std::make_shared<Exp>(operationIndex, mGraphMetadata);
        case OperationType::EXPAND_DIMS:
            return std::make_shared<ExpandDims>(operationIndex, mGraphMetadata);
        case OperationType::FULLY_CONNECTED:
            return std::make_shared<FullyConnected>(operationIndex, mGraphMetadata);
        case OperationType::FLOOR:
            return std::make_shared<Floor>(operationIndex, mGraphMetadata);
        case OperationType::GATHER:
            return std::make_shared<Gather>(operationIndex, mGraphMetadata);
        case OperationType::GREATER:
            return std::make_shared<Greater>(operationIndex, mGraphMetadata);
        case OperationType::GREATER_EQUAL:
            return std::make_shared<GreaterEqual>(operationIndex, mGraphMetadata);
        case OperationType::GROUPED_CONV_2D:
            return std::make_shared<GroupedConv2d>(operationIndex, mGraphMetadata);
        case OperationType::HARD_SWISH:
            return std::make_shared<HardSwish>(operationIndex, mGraphMetadata);
        case OperationType::INSTANCE_NORMALIZATION:
            return std::make_shared<InstanceNormalization>(operationIndex, mGraphMetadata);
        case OperationType::L2_POOL_2D:
            return std::make_shared<L2Pooling2D>(operationIndex, mGraphMetadata);
        case OperationType::L2_NORMALIZATION:
            return std::make_shared<L2Normalization>(operationIndex, mGraphMetadata);
        case OperationType::LSTM:
            return std::make_shared<LSTM>(operationIndex, mGraphMetadata);
        case OperationType::LESS:
            return std::make_shared<Less>(operationIndex, mGraphMetadata);
        case OperationType::LESS_EQUAL:
            return std::make_shared<LessEqual>(operationIndex, mGraphMetadata);
        case OperationType::LOG_SOFTMAX:
            return std::make_shared<LogSoftmax>(operationIndex, mGraphMetadata);
        case OperationType::LOG:
            return std::make_shared<Log>(operationIndex, mGraphMetadata);
        case OperationType::LOGICAL_AND:
            return std::make_shared<LogicalAnd>(operationIndex, mGraphMetadata);
        case OperationType::LOGICAL_NOT:
            return std::make_shared<LogicalNot>(operationIndex, mGraphMetadata);
        case OperationType::LOGICAL_OR:
            return std::make_shared<LogicalOr>(operationIndex, mGraphMetadata);
        case OperationType::LOGISTIC:
            return std::make_shared<Logistic>(operationIndex, mGraphMetadata);
        case OperationType::MAXIMUM:
            return std::make_shared<Maximum>(operationIndex, mGraphMetadata);
        case OperationType::MAX_POOL_2D:
            return std::make_shared<MaxPool2d>(operationIndex, mGraphMetadata);
        case OperationType::MEAN:
            return std::make_shared<Mean>(operationIndex, mGraphMetadata);
        case OperationType::MINIMUM:
            return std::make_shared<Minimum>(operationIndex, mGraphMetadata);
        case OperationType::MUL:
            return std::make_shared<Mul>(operationIndex, mGraphMetadata);
        case OperationType::NEG:
            return std::make_shared<Neg>(operationIndex, mGraphMetadata);
        case OperationType::NOT_EQUAL:
            return std::make_shared<NotEqual>(operationIndex, mGraphMetadata);
        case OperationType::PAD:
            return std::make_shared<Pad>(operationIndex, mGraphMetadata);
        case OperationType::PAD_V2:
            return std::make_shared<PadV2>(operationIndex, mGraphMetadata);
        case OperationType::POW:
            return std::make_shared<Pow>(operationIndex, mGraphMetadata);
        case OperationType::PRELU:
            return std::make_shared<PRelu>(operationIndex, mGraphMetadata);
        case OperationType::QUANTIZE:
            return std::make_shared<Quantize>(operationIndex, mGraphMetadata);
        case OperationType::REDUCE_ALL:
            return std::make_shared<ReduceAll>(operationIndex, mGraphMetadata);
        case OperationType::REDUCE_ANY:
            return std::make_shared<ReduceAny>(operationIndex, mGraphMetadata);
        case OperationType::REDUCE_MAX:
            return std::make_shared<ReduceMax>(operationIndex, mGraphMetadata);
        case OperationType::REDUCE_MIN:
            return std::make_shared<ReduceMin>(operationIndex, mGraphMetadata);
        case OperationType::REDUCE_PROD:
            return std::make_shared<ReduceProd>(operationIndex, mGraphMetadata);
        case OperationType::REDUCE_SUM:
            return std::make_shared<ReduceSum>(operationIndex, mGraphMetadata);
        case OperationType::RELU:
            return std::make_shared<Relu>(operationIndex, mGraphMetadata);
        case OperationType::RELU1:
            return std::make_shared<Relu1>(operationIndex, mGraphMetadata);
        case OperationType::RELU6:
            return std::make_shared<Relu6>(operationIndex, mGraphMetadata);
        case OperationType::RESHAPE:
            return std::make_shared<Reshape>(operationIndex, mGraphMetadata);
        case OperationType::RNN:
            return std::make_shared<RNN>(operationIndex, mGraphMetadata);
        case OperationType::ROI_ALIGN:
            return std::make_shared<ROIAlign>(operationIndex, mGraphMetadata);
        case OperationType::ROI_POOLING:
            return std::make_shared<ROIPooling>(operationIndex, mGraphMetadata);
        case OperationType::RSQRT:
            return std::make_shared<RSQRT>(operationIndex, mGraphMetadata);
        case OperationType::RESIZE_BILINEAR:
            return std::make_shared<ResizeBilinear>(operationIndex, mGraphMetadata);
        case OperationType::RESIZE_NEAREST_NEIGHBOR:
            return std::make_shared<ResizeNearestNeighbor>(operationIndex, mGraphMetadata);
        case OperationType::SELECT:
            return std::make_shared<Select>(operationIndex, mGraphMetadata);
        case OperationType::SOFTMAX:
            return std::make_shared<Softmax>(operationIndex, mGraphMetadata);
        case OperationType::SPACE_TO_BATCH_ND:
            return std::make_shared<SpaceToBatch>(operationIndex, mGraphMetadata);
        case OperationType::SPACE_TO_DEPTH:
            return std::make_shared<SpaceToDepth>(operationIndex, mGraphMetadata);
        case OperationType::SQRT:
            return std::make_shared<SQRT>(operationIndex, mGraphMetadata);
        case OperationType::SIN:
            return std::make_shared<Sin>(operationIndex, mGraphMetadata);
        case OperationType::SPLIT:
            return std::make_shared<Split>(operationIndex, mGraphMetadata);
        case OperationType::STRIDED_SLICE:
            return std::make_shared<StridedSlice>(operationIndex, mGraphMetadata);
        case OperationType::SQUEEZE:
            return std::make_shared<Squeeze>(operationIndex, mGraphMetadata);
        case OperationType::SUB:
            return std::make_shared<Sub>(operationIndex, mGraphMetadata);
        case OperationType::TANH:
            return std::make_shared<Tanh>(operationIndex, mGraphMetadata);
        case OperationType::TOPK_V2:
            return std::make_shared<TopkV2>(operationIndex, mGraphMetadata);
        case OperationType::TRANSPOSE_CONV_2D:
            return std::make_shared<TransposeConv2D>(operationIndex, mGraphMetadata);
        case OperationType::TRANSPOSE:
            return std::make_shared<Transpose>(operationIndex, mGraphMetadata);
        case OperationType::UNIDIRECTIONAL_SEQUENCE_RNN:
            return std::make_shared<UnidirectionalSequenceRNN>(operationIndex, mGraphMetadata);
        default:
            ALOGE("%s Cannot identify OperationType %d", __func__, operationType);
            break;
    }
    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
