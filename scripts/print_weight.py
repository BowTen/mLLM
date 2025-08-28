import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 本地模型路径
model_path = "/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B"  # 替换为你的实际路径

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 设置模型为评估模式
model.eval()

# 遍历所有参数并打印指定区域
for name, param in model.named_parameters():
    # 检查参数名称是否包含层信息，并且层号 >= 27
    if "layers." in name:
        # 提取层号
        parts = name.split(".")
        layer_index = None
        for i, part in enumerate(parts):
            if part == "layers" and i+1 < len(parts) and parts[i+1].isdigit():
                layer_index = int(parts[i+1])
                break
        
        # 如果层号小于27，跳过
        if layer_index is not None and layer_index < 27:
            continue

    print(f"\n=== 参数名称: {name} ===")
    print(f"参数形状: {param.shape}")
    if param.dim() == 2:
        # 转换为numpy数组以便处理
        param_np = param.data.cpu().numpy()
        
        # 打印左上角 4x4 区域
        print("左上角 4x4 矩阵:")
        rows = min(4, param_np.shape[0])
        cols = min(4, param_np.shape[1])
        print(param_np[:rows, :cols])
        
        # 打印右下角 4x4 区域
        print("右下角 4x4 矩阵:")
        start_row = max(0, param_np.shape[0] - 4)
        start_col = max(0, param_np.shape[1] - 4)
        print(param_np[start_row:, start_col:])
        
    else:
        print("前10个参数值:")
        print(param.data.flatten()[:10].cpu().numpy())  # 转移到CPU并转为numpy打印
        print("后10个参数值:")
        print(param.data.flatten()[-10:].cpu().numpy())
    print("-" * 50)
