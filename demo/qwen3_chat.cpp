#include "model/qwen3.h"

using namespace mllm;
using namespace mllm::tokenizer;

class Qwen3Chat
{
public:
    Qwen3Chat(const std::string &model_path, mllm::base::Device device, float temperature, float top_p, size_t top_k, float min_p)
        : model(mllm::model::Qwen3::from_pretrained(model_path, device, temperature, top_k, top_p, min_p)),
          tokenizer(model.get_tokenizer())
    {
    }

    std::string chat(const std::string &input, bool enable_thinking)
    {
        auto chat_token_ids = tokenizer->encode_with_chat_template(input, true, enable_thinking);
        auto input_id = tokenizer->to_tensor(chat_token_ids, model.device());
        base::Tensor next_id({1, 1}, model.device(), false, model.stream());

        bool is_end_think = false;
        std::string output;
        while (true)
        {
            if (!enable_thinking)
                is_end_think = true;
            input_id.toDevice(model.device());
            next_id.toDevice(model.device());
            model.forward(input_id, next_id);
            next_id.toDevice(base::Device::CPU);
            uint32_t id = *(reinterpret_cast<uint32_t *>(next_id.data()));

            std::string next_token = tokenizer->decode(id);
            output.append(next_token);

            if (id != BPETokenizer::QWEN3_END_OF_TEXT && id != BPETokenizer::QWEN3_IM_END)
            {
                std::cout << next_token;
                std::cout.flush();
            }
            if (id == BPETokenizer::QWEN3_END_THINK)
                is_end_think = true;
            if (is_end_think && id == BPETokenizer::QWEN3_END_OF_TEXT)
            {
                std::cout << std::endl;
                break;
            }
            input_id = next_id.clone();
        }

        return output;
    }

private:
    mllm::model::Qwen3 model;
    BPETokenizerPtr tokenizer;
};

int main()
{
    std::string model_path = "/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B";
    std::cout << "Loading model..." << std::endl;
    Qwen3Chat qwen3(model_path, base::Device::CUDA, 1.0, 1, 10000, 0.0);
    std::cout << "Loading accomplished." << std::endl;

    while (true)
    {
        std::string input;
        std::cout << "User: ";
        getline(std::cin, input);
        if (input == "exit")
            break;
        std::cout << "Qwen3: ";
        std::cout.flush();
        qwen3.chat(input, false);
    }

    return 0;
}