# mLLM 🔥

一个从零开始实现的轻量级大语言模型推理框架，使用 C++ 编写，专注于学习和理解 Transformer 架构的底层实现！

### ✨ 核心特性

- **🤖 模型支持**: 目前支持 Qwen3-0.6B 模型，直接兼容 Hugging Face 官方模型文件格式
- **⚡ 双后端支持**: 实现了 CUDA 和 CPU 算子，支持 GPU 和 CPU 推理
- **🧮 Sgemm**: 实现了 CUDA 的 Sgemm 算子，目前性能达到 cublas 的 82%
- **📝 BPE Tokenizer**: 自主实现的字节对编码分词器
- **💾 KV Cache**: 实现了键值缓存机制，提升推理效率

## 🏛️ 项目结构

```
MiniLLM/
├── include/
│   ├── base/                  # 基础组件
│   │   ├── allocator.h        # 内存分配器
│   │   ├── buffer.h           # 数据缓冲区
│   │   ├── common.h           # 通用定义
│   │   ├── json.hpp           # JSON 解析库
│   │   ├── safetensors.h      # SafeTensors 文件解析
│   │   ├── tensor.h           # 张量类定义
│   │   └── util.h             # 通用工具函数
│   ├── kernel/                # 计算内核
│   │   ├── cpu/               # CPU 算子实现
│   │   │   └── ...
│   │   ├── cuda/              # CUDA 算子实现
│   │   │   └── ...
│   │   └── kernel.h           # 内核接口
│   ├── model/                 # 模型架构
│   │   ├── qwen3.h            # Qwen3 主模型
│   │   ├── qwen3_decode_layer.h    # 解码层
│   │   ├── qwen3_mlp.h        # 多层感知机
│   │   ├── qwen3_rotary_embedding.h # 旋转位置编码
│   │   └── qwen3_self_attn.h  # 自注意力机制
│   ├── op/                    # 算子定义
│   │   ├── add.h              # 加法操作
│   │   ├── causal_mask.h      # 因果掩码
│   │   ├── ele_mul.h          # 元素乘法
│   │   ├── embedding.h        # 嵌入层
│   │   ├── layer.h            # 层基类
│   │   ├── linear.h           # 线性层
│   │   ├── mat_mul.h          # 矩阵乘法
│   │   ├── rms_norm.h         # RMS 归一化
│   │   ├── silu.h             # SiLU 激活函数
│   │   └── softmax.h          # Softmax 操作
│   └── tokenizer/             # 分词器
│       └── tokenizer.h        # 分词器接口
├── src/                       # 源文件实现
│   ├── base/
│   ├── kernel/
│   ├── model/
│   ├── op/
│   ├── tokenizer/
│   └── CMakeLists.txt
├── demo/
│   ├── qwen3_chat.cpp         # Qwen3 聊天演示
│   └── CMakeLists.txt
├── test/                      # 测试程序
├── scripts/                   # 辅助脚本
├── CMakeLists.txt
└── README.md
```

## 🧮 Sgemm 性能对比
**使用设备的是 4090**

M=N=K, M 整除64的情况
![SgemmDiv64性能对比图](https://github.com/BowTen/mLLM/raw/main/resources/gemm_performance_comparison_div64.png)

M=N=K, M 整除1的情况<br>
这种情况目前只是在 div64 的基础上将所有 float4 存取展开，导致性能大幅下降，后续优化TODO
![SgemmDiv1性能对比图](https://github.com/BowTen/mLLM/raw/main/resources/gemm_performance_comparison_div1.png)


## 🤖 Qwen3 架构图
![Qwen3架构图](https://github.com/BowTen/mLLM/raw/main/resources/qwen3_arc.png)

## ⌨️ 聊天示例

```cpp
#include "model/qwen3.h"

using namespace mllm;
using namespace mllm::tokenizer;

class Qwen3Chat
{
public:
    Qwen3Chat(const std::string &model_path, mllm::base::Device device, float temperature)
        : model(mllm::model::Qwen3::from_pretrained(model_path, device, temperature)),
          tokenizer(model.get_tokenizer())
    {
    }

    std::string chat(const std::string &input)
    {
        auto chat_token_ids = tokenizer->encode_with_chat_template(input, true, true);
        auto input_id = tokenizer->to_tensor(chat_token_ids, model.device());
        base::Tensor next_id({1, 1}, model.device(), false, model.stream());

        std::string output;
        while (true)
        {
            input_id.toDevice(model.device());
            next_id.toDevice(model.device());
            model.forward(input_id, next_id);
            next_id.toDevice(base::Device::CPU);
            uint32_t id = *(reinterpret_cast<uint32_t *>(next_id.data()));

            std::string next_token = tokenizer->decode(id);
            output.append(next_token);

            if (id == BPETokenizer::QWEN3_END_OF_TEXT || id == BPETokenizer::QWEN3_IM_END)
            {
                std::cout << std::endl;
                break;
            }
            std::cout << next_token;
            std::cout.flush();
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
    std::string model_path = "path_to_your_resources/Qwen/Qwen3-0.6B";
    std::cout << "Loading model..." << std::endl;
    Qwen3Chat qwen3(model_path, base::Device::CUDA, 0.6);
    std::cout << "Loading accomplished." << std::endl;

    while (true)
    {
        std::string input;
        std::cout << "User: ";
        std::cin >> input;
        if (input == "exit")
            break;
        std::cout << "Qwen3: ";
        std::cout.flush();
        qwen3.chat(input);
    }

    return 0;
}
```

![Qwen3 chat demo 演示动图](https://github.com/BowTen/mLLM/raw/main/resources/chat_demo.gif)


## 依赖库

- armadillo
- glog
- gtest