#include "DetectionClient.h"

#undef LOG_TAG
#define LOG_TAG "DetectionClient"

std::string DetectionClient::prepare(bool& flag) {
    RequestString request;
    request.mutable_token()->set_data(mToken);
    ReplyStatus reply;
    ClientContext context;
    time_point deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(10000);
    context.set_deadline(deadline);

    Status status = stub_->prepare(&context, request, &reply);

    if (status.ok()) {
        flag = reply.status();
        return (flag ? "status True" : "status False");
    } else {
        flag = false;
        return std::string(status.error_message());
    }
}

std::string DetectionClient::release(bool& flag) {
    RequestString request;
    request.mutable_token()->set_data(mToken);
    ReplyStatus reply;
    ClientContext context;

    Status status = stub_->release(&context, request, &reply);

    if (status.ok()) {
        flag = reply.status();
        return (flag ? "status True" : "status False");
    } else {
        return std::string(status.error_message());
    }
}

Status DetectionClient::sendFile(std::string fileName,
                std::unique_ptr<ClientWriter<RequestDataChunks> >& writer) {
    RequestDataChunks request;
    request.mutable_token()->set_data(mToken);
    uint32_t CHUNK_SIZE = 1024 * 1024;
    std::ifstream fin(fileName, std::ifstream::binary);
    std::vector<char> buffer(CHUNK_SIZE, 0);
    ALOGV("GRPC sendFile %s", fileName.c_str());
    ALOGI("GRPC sendFile %d sized chunks", CHUNK_SIZE);

    if (!fin.is_open()) ALOGE("GRPC sendFile file Open Error ");
    while (!fin.eof()) {
        fin.read(buffer.data(), buffer.size());
        std::streamsize s = fin.gcount();
        // ALOGI("GRPC sendFile read %d", s);
        request.set_data(buffer.data(), s);
        if (!writer->Write(request)) {
            ALOGE("GRPC Broken Stream ");
            break;
        }
    }

    writer->WritesDone();
    ALOGI("GRPC sendFile completed");
    return writer->Finish();
}

bool DetectionClient::isModelLoaded(std::string fileName) {
    ReplyStatus reply;
    ClientContext context;
    RequestString request;
    request.mutable_token()->set_data(mToken);
    time_point deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(20000);
    context.set_deadline(deadline);
    status = stub_->loadModel(&context, request, &reply);
    if(status.ok()) {
        return reply.status();
    } else {
        ALOGE("Model Load failure: %s", status.error_message().c_str());
    }
    return false;
}

std::string DetectionClient::sendIRs(bool& flag, const std::string& ir_xml, const std::string& ir_bin) {
    ReplyStatus reply;
    ClientContext context;
    std::unique_ptr<ClientWriter<RequestDataChunks> > writerXml =
        std::unique_ptr<ClientWriter<RequestDataChunks> >(stub_->sendXml(&context, &reply));
    Status status = sendFile(ir_xml, writerXml);

    if (status.ok()) {
        ClientContext newContext;
        std::unique_ptr<ClientWriter<RequestDataChunks> > writerBin =
            std::unique_ptr<ClientWriter<RequestDataChunks> >(
                stub_->sendBin(&newContext, &reply));
        status = sendFile(ir_bin, writerBin);
        if (status.ok()) {
            flag = reply.status();
            //if model is sent succesfully trigger model loading
            if (flag && isModelLoaded(ir_xml) ) {
                flag = true;
                return ("status True");
            } else {
                flag = false;
                ALOGE("Model Loading Failed!!!");
                return ("status False");
            }
        } else {
            return ("status False");
        }
    }
    return std::string(status.error_message());
}

void DetectionClient::add_input_data(std::string label, const uint8_t* buffer, std::vector<size_t> shape, uint32_t size, android::hardware::neuralnetworks::nnhal::OperandType operandType) {
    const float* src;
    size_t index;

    DataTensor* input = request.add_data_tensors();
    input->set_node_name(label);
    switch(operandType) {
        case android::hardware::neuralnetworks::nnhal::OperandType::TENSOR_INT32: {
            input->set_data_type(DataTensor::i32);
            break;
        }
        case android::hardware::neuralnetworks::nnhal::OperandType::TENSOR_FLOAT16: {
            input->set_data_type(DataTensor::f16);
            break;
        }
        case android::hardware::neuralnetworks::nnhal::OperandType::TENSOR_FLOAT32: {
            input->set_data_type(DataTensor::f32);
            break;
        }
        case android::hardware::neuralnetworks::nnhal::OperandType::TENSOR_BOOL8: {
            input->set_data_type(DataTensor::boolean);
            break;
        }
        case android::hardware::neuralnetworks::nnhal::OperandType::TENSOR_QUANT8_ASYMM: {
            input->set_data_type(DataTensor::u8);
            break;
        }
        case android::hardware::neuralnetworks::nnhal::OperandType::TENSOR_QUANT8_SYMM:
        case android::hardware::neuralnetworks::nnhal::OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
        case android::hardware::neuralnetworks::nnhal::OperandType::TENSOR_QUANT8_ASYMM_SIGNED: {
            input->set_data_type(DataTensor::i8);
            break;
        }
        case android::hardware::neuralnetworks::nnhal::OperandType::TENSOR_QUANT16_SYMM: {
            input->set_data_type(DataTensor::i16);
            break;
        }
        case android::hardware::neuralnetworks::nnhal::OperandType::TENSOR_QUANT16_ASYMM: {
            input->set_data_type(DataTensor::u16);
            break;
        }
        default: {
            input->set_data_type(DataTensor::u8);
            break;
        }
    }
    for (index = 0; index < shape.size(); index++) {
        input->add_tensor_shape(shape[index]);
    }
    input->set_data(buffer, size);
}

void DetectionClient::get_output_data(std::string label, uint8_t* buffer, std::vector<size_t> shape, uint32_t expectedLength) {
    std::string src;
    size_t index;
    size_t size = 1;

    for (index = 0; index < shape.size(); index++) {
        size *= shape[index];
    }
    for (index = 0; index < reply.data_tensors_size(); index++) {
        if (label.compare(reply.data_tensors(index).node_name()) == 0) {
            src = reply.data_tensors(index).data();
            if(expectedLength != src.length()) {
                ALOGE("Length Mismatch error: expected length %d , actual length %d", expectedLength, src.length());
            }
            memcpy(buffer, src.data(), src.length());
            break;
        }
    }
}

void DetectionClient::clear_data() {
    request.clear_data_tensors();
    reply.clear_data_tensors();
}

std::string DetectionClient::remote_infer() {
    ClientContext context;
    time_point deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(5000);
    context.set_deadline(deadline);

    request.mutable_token()->set_data(mToken);
    status = stub_->getInferResult(&context, request, &reply);
    if (status.ok()) {
        if (reply.data_tensors_size() == 0) ALOGE("GRPC reply empty, ovms failure ?");
        return "Success";
    } else {
        ALOGE("GRPC Error code: %d, message: %s", status.error_code(),
                status.error_message().c_str());
        return std::string(status.error_message());
    }
}

bool DetectionClient::get_status() {
    if (status.ok() && (reply.data_tensors_size() > 0))
        return 1;
    else {
        return 0;
    }
}