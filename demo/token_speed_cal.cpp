#include "model/qwen3.h"
#include <chrono>

using namespace mllm;
using namespace mllm::tokenizer;

void set_token_id(base::Tensor &tensor, uint32_t id)
{
    auto original_device = tensor.device();
    tensor.toDevice(base::Device::CPU);
    *reinterpret_cast<uint32_t *>(tensor.data()) = id;
    tensor.toDevice(original_device);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: ./token_speed_cal text_path output_path" << std::endl;
        return -1;
    }
    std::string text_path = argv[1];
    std::ifstream ifs(text_path);
    if (!ifs.is_open())
    {
        std::cerr << "Failed to open file: " << text_path << std::endl;
        return -1;
    }
    std::string text((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
    ifs.close();

    std::string output_path = argv[2];
    std::ofstream csv(output_path);
    csv << "Sequence Length,Tokens per Second" << std::endl;

    base::Device device = base::Device::CUDA;
    std::string model_path = "/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B";
    std::cout << "Loading model..." << std::endl;
    model::Qwen3 qwen3 = model::Qwen3::from_pretrained(model_path, device, 1.0f, 100, 1.0f, 0.0f);
    tokenizer::BPETokenizerPtr tokenizer = qwen3.get_tokenizer();
    std::cout << "Loading accomplished." << std::endl;

    auto token_ids = tokenizer->encode(text);

    base::Tensor input_id({1}, device, false, qwen3.stream());
    base::Tensor next_id({1}, device, false, qwen3.stream());
    float total_time = 0.0f;
    for (int i = 0; i < token_ids.size(); i++)
    {
        set_token_id(input_id, token_ids[i]);
        auto start = std::chrono::high_resolution_clock::now();
        qwen3.forward(input_id, next_id);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<float>(end - start).count();
        std::cout << tokenizer->decode(token_ids[i]) << std::flush;
        if ((i + 1) % 10 == 0)
        {
            csv << (i + 1) << "," << (10.0f / total_time) << std::endl;
            total_time = 0.0f;
        }
    }

    csv.close();
    return 0;
}