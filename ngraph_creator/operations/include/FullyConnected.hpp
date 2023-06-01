#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class FullyConnected : public OperationsBase {
public:
    FullyConnected(int operationIndex, GraphMetadata graphMetadata);
    bool validate() override;
    std::shared_ptr<ov::Node> createNode() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android