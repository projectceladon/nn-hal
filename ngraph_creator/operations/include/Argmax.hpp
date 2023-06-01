#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Argmax : public OperationsBase {
public:
    Argmax(int operationIndex, GraphMetadata graphMetadata);
    std::shared_ptr<ov::Node> createNode() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
