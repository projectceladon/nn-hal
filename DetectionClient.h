#ifndef __DETECTION_CLIENT_H
#define __DETECTION_CLIENT_H

#include <android-base/logging.h>
#include <android/log.h>
#include <grpcpp/grpcpp.h>
#include <log/log.h>
#include <fstream>
#include <string>
#include "Driver.h"
#include "nnhal_object_detection.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientWriter;
using grpc::Status;
using objectDetection::DataTensor;
using objectDetection::Detection;
using objectDetection::ReplyDataTensors;
using objectDetection::ReplyStatus;
using objectDetection::RequestDataChunks;
using objectDetection::RequestDataTensors;
using objectDetection::RequestString;
using time_point = std::chrono::system_clock::time_point;

class DetectionClient {
public:
    DetectionClient(std::shared_ptr<Channel> channel, uint32_t token)
        : stub_(Detection::NewStub(channel)), mToken(token) {}

    std::string prepare(bool& flag);
    std::string release(bool& flag);

    Status sendFile(std::string fileName,
                    std::unique_ptr<ClientWriter<RequestDataChunks> >& writer);

    std::string sendIRs(bool& flag, const std::string& ir_xml, const std::string& ir_bin);

    bool isModelLoaded(std::string fileName, bool quantType);
    void add_input_data(std::string label, const uint8_t* buffer, std::vector<uint32_t> shape,
                        uint32_t size,
                        android::hardware::neuralnetworks::nnhal::OperandType operandType);
    void get_output_data(std::string label, uint8_t* buffer, uint32_t expectedLength);
    void clear_data();
    std::string remote_infer();
    bool get_status();

private:
    std::unique_ptr<Detection::Stub> stub_;
    RequestDataTensors request;
    ReplyDataTensors reply;
    Status status;
    uint32_t mToken;
};

#endif