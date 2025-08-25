import sys
sys.path.insert(0, "/home/hznuojai/ai_infra/MiniLLM/scripts/transformers_8")
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Model
import torch
import numpy as np
# from transformers_8 import AutoTokenizer, AutoModelForCausalLM

model_path = "/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 存储每层输出的字典
layer_outputs = {}
attention_outputs = {}

def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            # 对于attention层，通常返回(hidden_states, attention_weights)
            tensor_data = output[0].detach().cpu().contiguous().numpy()
            layer_outputs[name] = tensor_data
            if len(output) > 1 and output[1] is not None:
                attention_data = output[1].detach().cpu().contiguous().numpy()
                attention_outputs[name] = attention_data
        else:
            tensor_data = output.detach().cpu().contiguous().numpy()
            layer_outputs[name] = tensor_data
        print(f"Layer {name}: shape {layer_outputs[name].shape}")
    return hook

# 注册hooks到每一层
for name, module in model.named_modules():
    if 'layers' in name or 'embed' in name or 'norm' in name or 'attn' in name or 'mlp' in name:
        module.register_forward_hook(hook_fn(name))

# 直接使用原始字符串
text = "The weather is really "
inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(model.device)

print("Input tokens:", inputs['input_ids'])
print("Input text:", tokenizer.decode(inputs['input_ids'][0]))

# 执行一次前向传播来获取所有层的输出
print("\n=== Forward Pass ===")
with torch.no_grad():
    outputs = model(**inputs)

print(f"\nFinal output shape: {outputs.logits.shape}")
print(f"Final logits (first 10): {outputs.logits[0, -1, :10].cpu().numpy()}")

# 保存关键层的输出到文件，方便C++对比
import os
debug_dir = "/home/hznuojai/ai_infra/MiniLLM/scripts/debug_outputs"
os.makedirs(debug_dir, exist_ok=True)

print(f"\n=== Saving layer outputs to {debug_dir} ===")
for name, output in layer_outputs.items():
    if any(key in name for key in ['embed', 'layers.', 'norm']):
        # 确保数据是连续的
        if not output.flags['C_CONTIGUOUS']:
            output = np.ascontiguousarray(output)
        filename = f"{debug_dir}/{name.replace('.', '_')}.npy"
        np.save(filename, output)
        print(f"Saved {name}: {output.shape} -> {filename}")
        print(f"  Memory layout: C_CONTIGUOUS={output.flags['C_CONTIGUOUS']}, F_CONTIGUOUS={output.flags['F_CONTIGUOUS']}")

# 也保存输入tokens和最终logits
input_ids_continuous = inputs['input_ids'].detach().cpu().contiguous().numpy()
final_logits_continuous = outputs.logits.detach().cpu().contiguous().numpy()

np.save(f"{debug_dir}/input_ids.npy", input_ids_continuous)
np.save(f"{debug_dir}/final_logits.npy", final_logits_continuous)

# 打印一些关键信息用于对比
print(f"\n=== Key Information for C++ Comparison ===")
print(f"Input tokens: {inputs['input_ids'][0].tolist()}")
print(f"Sequence length: {inputs['input_ids'].shape[1]}")
print(f"Vocab size: {outputs.logits.shape[-1]}")

# 打印embedding层输出（如果存在）
for name, output in layer_outputs.items():
    if 'embed' in name and 'token' in name:
        print(f"\nEmbedding output shape: {output.shape}")
        print(f"First token embedding (first 5 dims): {output[0, 0, :5]}")
        break

# 打印第一层transformer的输出
for name, output in layer_outputs.items():
    if 'layers.0' in name and name.endswith('layers.0'):
        print(f"\nFirst layer output shape: {output.shape}")
        print(f"First token first layer (first 5 dims): {output[0, 0, :5]}")
        break