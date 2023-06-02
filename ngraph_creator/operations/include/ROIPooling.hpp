#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class ROIPooling : public OperationsBase {
public:
    ROIPooling(int operationIndex, GraphMetadata graphMetadata);
    bool validate() override;
    std::shared_ptr<ov::Node> createNode() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
