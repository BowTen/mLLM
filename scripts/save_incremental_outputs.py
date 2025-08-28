import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Model
import torch
import numpy as np
import os
import pdb  # Python调试器

# 设置更详细的日志
import logging
logging.basicConfig(level=logging.DEBUG)

model_path = "/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 存储每层输出的字典
layer_outputs_step1 = {}
layer_outputs_step2 = {}

def create_hook_fn(outputs_dict, step_name):
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # 对于attention层，通常返回(hidden_states, attention_weights)
                tensor_data = output[0].detach().cpu().contiguous().numpy()
                outputs_dict[name] = tensor_data
            else:
                tensor_data = output.detach().cpu().contiguous().numpy()
                outputs_dict[name] = tensor_data
            print(f"[{step_name}] Layer {name}: shape {outputs_dict[name].shape}")
        return hook
    return hook_fn

def register_hooks(outputs_dict, step_name):
    """注册hooks到每一层"""
    hooks = []
    hook_creator = create_hook_fn(outputs_dict, step_name)
    
    for name, module in model.named_modules():
        if 'layers' in name or 'embed' in name or 'norm' in name or 'attn' in name or 'mlp' in name:
            hook = module.register_forward_hook(hook_creator(name))
            hooks.append(hook)
    return hooks

def remove_hooks(hooks):
    """移除所有hooks"""
    for hook in hooks:
        hook.remove()

# 创建debug目录
debug_dir = "/home/hznuojai/ai_infra/MiniLLM/scripts/debug_outputs"
os.makedirs(debug_dir, exist_ok=True)
os.makedirs(f"{debug_dir}/step1", exist_ok=True)
os.makedirs(f"{debug_dir}/step2", exist_ok=True)

print("=== Step 1: Processing first input ===")
# 第一步：处理 "The weather is really "
first_text = "The weather is really "
first_inputs = tokenizer(
    first_text,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(model.device)

print("First input tokens:", first_inputs['input_ids'])
print("First input text:", tokenizer.decode(first_inputs['input_ids'][0]))

# 注册hooks for step 1
hooks1 = register_hooks(layer_outputs_step1, "STEP1")

# 第一次前向传播 - 获取past_key_values
print("\n=== First Forward Pass ===")
with torch.no_grad():
    first_outputs = model(**first_inputs, use_cache=True)

# 移除第一步的hooks
remove_hooks(hooks1)

print(f"First output shape: {first_outputs.logits.shape}")
print(f"Past key values length: {len(first_outputs.past_key_values)}")
print(f"First layer past_key_values shape: key={first_outputs.past_key_values[0][0].shape}, value={first_outputs.past_key_values[0][1].shape}")

# 保存第一步的输出
print(f"\n=== Saving Step 1 outputs to {debug_dir} ===")
for name, output in layer_outputs_step1.items():
    if any(key in name for key in ['embed', 'layers.', 'norm']):
        if not output.flags['C_CONTIGUOUS']:
            output = np.ascontiguousarray(output)
        filename = f"{debug_dir}/step1/{name.replace('.', '_')}.npy"
        np.save(filename, output)
        print(f"Saved step1 {name}: {output.shape} -> {filename}")

# 保存第一步的输入和logits
first_input_ids_continuous = first_inputs['input_ids'].detach().cpu().contiguous().numpy()
first_logits_continuous = first_outputs.logits.detach().cpu().contiguous().numpy()
np.save(f"{debug_dir}/step1_input_ids.npy", first_input_ids_continuous)
np.save(f"{debug_dir}/step1_final_logits.npy", first_logits_continuous)

print("\n=== Step 2: Processing second input with KV cache ===")
# 第二步：处理 "2" 同时使用KV cache
second_text = "2"
second_inputs = tokenizer(
    second_text,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(model.device)

print("Second input tokens:", second_inputs['input_ids'])
print("Second input text:", tokenizer.decode(second_inputs['input_ids'][0]))

# 注册hooks for step 2
hooks2 = register_hooks(layer_outputs_step2, "STEP2")

# 第二次前向传播 - 使用past_key_values
print("\n=== Second Forward Pass with KV Cache ===")
print("Debug: About to call model with incremental input...")
print(f"second_inputs keys: {second_inputs.keys()}")
print(f"second_inputs['input_ids'].shape: {second_inputs['input_ids'].shape}")
print(f"past_key_values type: {type(first_outputs.past_key_values)}")
print(f"past_key_values length: {len(first_outputs.past_key_values)}")

# 添加调试断点 - 在这里可以进入transformers库
# pdb.set_trace()  # 取消注释来启用断点

with torch.no_grad():
    # 在这里可以设置断点来进入transformers库的forward方法
    second_outputs = model(**second_inputs, 
                          past_key_values=first_outputs.past_key_values,
                          use_cache=True)

# 移除第二步的hooks
remove_hooks(hooks2)

print(f"Second output shape: {second_outputs.logits.shape}")
print(f"Updated past key values length: {len(second_outputs.past_key_values)}")
print(f"Updated first layer past_key_values shape: key={second_outputs.past_key_values[0][0].shape}, value={second_outputs.past_key_values[0][1].shape}")

# 保存第二步的输出
print(f"\n=== Saving Step 2 outputs to {debug_dir} ===")
for name, output in layer_outputs_step2.items():
    if any(key in name for key in ['embed', 'layers.', 'norm']):
        if not output.flags['C_CONTIGUOUS']:
            output = np.ascontiguousarray(output)
        filename = f"{debug_dir}/step2/{name.replace('.', '_')}.npy"
        np.save(filename, output)
        print(f"Saved step2 {name}: {output.shape} -> {filename}")

# 保存第二步的输入和logits
second_input_ids_continuous = second_inputs['input_ids'].detach().cpu().contiguous().numpy()
second_logits_continuous = second_outputs.logits.detach().cpu().contiguous().numpy()
np.save(f"{debug_dir}/step2_input_ids.npy", second_input_ids_continuous)
np.save(f"{debug_dir}/step2_final_logits.npy", second_logits_continuous)

# 保存KV cache数据
print(f"\n=== Saving KV Cache data ===")
for layer_idx, (key, value) in enumerate(first_outputs.past_key_values):
    key_data = key.detach().cpu().contiguous().numpy()
    value_data = value.detach().cpu().contiguous().numpy()
    np.save(f"{debug_dir}/step1/kv_cache_layer_{layer_idx}_key.npy", key_data)
    np.save(f"{debug_dir}/step1/kv_cache_layer_{layer_idx}_value.npy", value_data)
    print(f"Saved layer {layer_idx} KV cache: key {key_data.shape}, value {value_data.shape}")

for layer_idx, (key, value) in enumerate(second_outputs.past_key_values):
    key_data = key.detach().cpu().contiguous().numpy()
    value_data = value.detach().cpu().contiguous().numpy()
    np.save(f"{debug_dir}/step2/kv_cache_layer_{layer_idx}_key.npy", key_data)
    np.save(f"{debug_dir}/step2/kv_cache_layer_{layer_idx}_value.npy", value_data)
    print(f"Saved updated layer {layer_idx} KV cache: key {key_data.shape}, value {value_data.shape}")

# 打印关键信息用于C++对比
print(f"\n=== Key Information for C++ Incremental Comparison ===")
print(f"Step 1 - Input tokens: {first_inputs['input_ids'][0].tolist()}")
print(f"Step 1 - Sequence length: {first_inputs['input_ids'].shape[1]}")
print(f"Step 2 - Input tokens: {second_inputs['input_ids'][0].tolist()}")
print(f"Step 2 - Sequence length: {second_inputs['input_ids'].shape[1]}")
print(f"Vocab size: {second_outputs.logits.shape[-1]}")

# 预测下一个token
next_token_id = torch.argmax(second_outputs.logits[0, -1, :]).item()
print(f"\nPredicted next token ID: {next_token_id}")
print(f"Predicted next token: '{tokenizer.decode(next_token_id)}'")

# 打印一些关键层的输出用于调试
print(f"\n=== Debug Information ===")
for name, output in layer_outputs_step2.items():
    if 'embed' in name and 'token' in name:
        print(f"Step 2 Embedding output shape: {output.shape}")
        print(f"Step 2 First token embedding (first 5 dims): {output[0, 0, :5]}")
        break

for name, output in layer_outputs_step2.items():
    if 'layers.0' in name and name.endswith('layers.0'):
        print(f"Step 2 First layer output shape: {output.shape}")
        print(f"Step 2 First token first layer (first 5 dims): {output[0, 0, :5]}")
        break

print(f"\nStep 2 Final logits (first 10): {second_outputs.logits[0, -1, :10].cpu().numpy()}")
