#ifndef QWEN3_CHECK_HPP
#define QWEN3_CHECK_HPP

#include <gtest/gtest.h>
#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>
#include <cmath>
#include <exception>

#define private public
#define protected public
#include "model/qwen3.h"
#undef protected
#undef private

using namespace std;
using namespace mllm;
using namespace mllm::base;
using namespace mllm::model;

class Qwen3Check
{
public:
    float check_eps = 1e-6f;
    Qwen3 model_cpu;
    Qwen3 model_cuda;

    size_t hidden_size;

    Qwen3Check(string model_path, float temperature, float check_eps)
        : check_eps(check_eps),
          model_cpu(Qwen3::from_pretrained(model_path, Device::CPU, temperature)),
          model_cuda(Qwen3::from_pretrained(model_path, Device::CUDA, temperature)),
          hidden_size(model_cpu.hidden_size)
    {
        LOG(INFO) << "Load model completed";
    }

    void check_tensor(Tensor &cpu, Tensor &cuda, const string &tensor_name = "")
    {
        CHECK(cpu.shape() == cuda.shape()) << "Shape mismatch for " << tensor_name;
        cuda.toDevice(Device::CPU);
        size_t total_size = cpu.size();

        // Check for NaN or inf values first
        size_t nan_count_cpu = 0, inf_count_cpu = 0;
        size_t nan_count_cuda = 0, inf_count_cuda = 0;

        for (size_t i = 0; i < total_size; i++)
        {
            float cpu_val = *cpu[i];
            float cuda_val = *cuda[i];

            if (std::isnan(cpu_val))
                nan_count_cpu++;
            if (std::isinf(cpu_val))
                inf_count_cpu++;
            if (std::isnan(cuda_val))
                nan_count_cuda++;
            if (std::isinf(cuda_val))
                inf_count_cuda++;
        }

        if (nan_count_cpu > 0 || inf_count_cpu > 0 || nan_count_cuda > 0 || inf_count_cuda > 0)
        {
            LOG(ERROR) << "Invalid values detected in " << tensor_name << ":";
            LOG(ERROR) << "CPU - NaN: " << nan_count_cpu << ", Inf: " << inf_count_cpu;
            LOG(ERROR) << "CUDA - NaN: " << nan_count_cuda << ", Inf: " << inf_count_cuda;
        }

        // Find first few mismatches for debugging
        size_t mismatch_count = 0;
        for (size_t i = 0; i < total_size && mismatch_count < 10; i++)
        {
            float cpu_val = *cpu[i];
            float cuda_val = *cuda[i];

            if (fabs(cpu_val - cuda_val) >= check_eps)
            {
                LOG(ERROR) << "Mismatch " << mismatch_count << " in " << tensor_name
                           << " at index " << i << ", cpu: " << cpu_val << ", cuda: " << cuda_val
                           << ", diff: " << fabs(cpu_val - cuda_val);
                mismatch_count++;
            }
        }

        CHECK(mismatch_count == 0) << "Found " << mismatch_count << " mismatches in " << tensor_name;
        cuda.toDevice(Device::CUDA);
    }

    size_t forward(vector<size_t> token_ids)
    {
        LOG(INFO) << "Starting forward pass comparison between CPU and CUDA";

        // Prepare input tensors
        Tensor token_ids_cpu = Tensor::from_vector(token_ids, {token_ids.size()}, Device::CPU);
        Tensor token_ids_cuda = Tensor::from_vector(token_ids, {token_ids.size()}, Device::CUDA);

        // 1. Embedding layer comparison
        LOG(INFO) << "Checking embedding layer...";
        model_cpu.hidden_state = Tensor({token_ids_cpu.shape(0), hidden_size}, model_cpu.device_);
        model_cuda.hidden_state = Tensor({token_ids_cuda.shape(0), hidden_size}, model_cuda.device_);
        model_cpu.embed_tokens.forward(token_ids_cpu, model_cpu.hidden_state);
        model_cuda.embed_tokens.forward(token_ids_cuda, model_cuda.hidden_state);
        check_tensor(model_cpu.hidden_state, model_cuda.hidden_state, "embedding_output");
        LOG(INFO) << "✓ Embedding layer passed";

        // 2. Rotary embedding preparation
        LOG(INFO) << "Preparing rotary embeddings...";
        auto rope_emb_shape_cpu = model_cpu.hidden_state.shape();
        auto rope_emb_shape_cuda = model_cuda.hidden_state.shape();
        rope_emb_shape_cpu.back() = model_cpu.config_["head_dim"];
        rope_emb_shape_cuda.back() = model_cuda.config_["head_dim"];

        model_cpu.cos = Tensor(rope_emb_shape_cpu, model_cpu.device_);
        model_cpu.sin = Tensor(rope_emb_shape_cpu, model_cpu.device_);
        model_cuda.cos = Tensor(rope_emb_shape_cuda, model_cuda.device_);
        model_cuda.sin = Tensor(rope_emb_shape_cuda, model_cuda.device_);

        base::PosEmb pos_emb_cpu(&model_cpu.cos, &model_cpu.sin);
        base::PosEmb pos_emb_cuda(&model_cuda.cos, &model_cuda.sin);

        size_t seq_len = model_cpu.hidden_state.shape(-2);
        model_cpu.rotary_embedding.forward(model_cpu.pos_id, model_cpu.pos_id + seq_len, pos_emb_cpu);
        model_cuda.rotary_embedding.forward(model_cuda.pos_id, model_cuda.pos_id + seq_len, pos_emb_cuda);

        check_tensor(model_cpu.cos, model_cuda.cos, "rotary_cos");
        check_tensor(model_cpu.sin, model_cuda.sin, "rotary_sin");
        LOG(INFO) << "✓ Rotary embedding preparation passed";

        // 3. Decode layers comparison
        LOG(INFO) << "Checking decode layers...";
        for (size_t i = 0; i < model_cpu.layers.size(); i++)
        {
            LOG(INFO) << "Checking layer " << i << "...";

            // Save previous hidden state for comparison
            Tensor prev_hidden_cpu = model_cpu.hidden_state.clone();
            Tensor prev_hidden_cuda = model_cuda.hidden_state.clone();

            try
            {
                // model_cpu.layers[i].forward(&model_cpu.hidden_state, &model_cpu.hidden_state, pos_emb_cpu);
                // model_cuda.layers[i].forward(&model_cuda.hidden_state, &model_cuda.hidden_state, pos_emb_cuda);
                // 合并成下面这个函数
                decode_layer_forward(&model_cpu.layers[i], &model_cpu.hidden_state, &model_cpu.hidden_state, pos_emb_cpu,
                                     &model_cuda.layers[i], &model_cuda.hidden_state, &model_cuda.hidden_state, pos_emb_cuda);

                check_tensor(model_cpu.hidden_state, model_cuda.hidden_state, "layer_" + to_string(i) + "_output");
                LOG(INFO) << "✓ Layer " << i << " passed";
            }
            catch (const exception &e)
            {
                LOG(ERROR) << "❌ Layer " << i << " failed: " << e.what();
                LOG(ERROR) << "Stopping at layer " << i;
                throw;
            }
        }

        // 4. Get last hidden state
        LOG(INFO) << "Getting last hidden state...";
        kernel::get_last_hidden_state_kernel(model_cpu.device_)(&model_cpu.hidden_state, nullptr);
        kernel::get_last_hidden_state_kernel(model_cuda.device_)(&model_cuda.hidden_state, nullptr);
        check_tensor(model_cpu.hidden_state, model_cuda.hidden_state, "last_hidden_state");
        LOG(INFO) << "✓ Last hidden state extraction passed";

        // 5. Final normalization
        LOG(INFO) << "Checking final normalization...";
        model_cpu.norm.forward(model_cpu.hidden_state, model_cpu.hidden_state);
        model_cuda.norm.forward(model_cuda.hidden_state, model_cuda.hidden_state);
        check_tensor(model_cpu.hidden_state, model_cuda.hidden_state, "final_norm_output");
        LOG(INFO) << "✓ Final normalization passed";

        // 6. Temperature scaling
        LOG(INFO) << "Checking temperature scaling...";
        model_cpu.temp_scal.forward(model_cpu.hidden_state, model_cpu.temperature_scaling, model_cpu.hidden_state);
        model_cuda.temp_scal.forward(model_cuda.hidden_state, model_cuda.temperature_scaling, model_cuda.hidden_state);
        check_tensor(model_cpu.hidden_state, model_cuda.hidden_state, "temperature_scaled_output");
        LOG(INFO) << "✓ Temperature scaling passed";

        // 7. Language model head
        LOG(INFO) << "Checking language model head...";
        model_cpu.final_probability = Tensor({model_cpu.hidden_state.shape(0), model_cpu.vocab_size}, model_cpu.device_);
        model_cuda.final_probability = Tensor({model_cuda.hidden_state.shape(0), model_cuda.vocab_size}, model_cuda.device_);
        model_cpu.lm_head.forward(model_cpu.hidden_state, model_cpu.final_probability);
        model_cuda.lm_head.forward(model_cuda.hidden_state, model_cuda.final_probability);
        check_tensor(model_cpu.final_probability, model_cuda.final_probability, "lm_head_output");
        LOG(INFO) << "✓ Language model head passed";

        // 8. Softmax
        LOG(INFO) << "Checking softmax...";
        model_cpu.softmax.forward(model_cpu.final_probability, model_cpu.final_probability);
        model_cuda.softmax.forward(model_cuda.final_probability, model_cuda.final_probability);
        check_tensor(model_cpu.final_probability, model_cuda.final_probability, "softmax_output");
        LOG(INFO) << "✓ Softmax passed";

        LOG(INFO) << "🎉 All layers passed CPU vs CUDA comparison!";

        // Move to CPU for final processing
        model_cpu.final_probability.toDevice(Device::CPU);
        model_cuda.final_probability.toDevice(Device::CPU);

        // Return next token (using CPU model result)
        Tensor next_token = Tensor({1}, Device::CPU);
        kernel::random_sampling_cpu(&model_cpu.final_probability, &next_token, model_cpu.device_);

        return *next_token[0];
    }

    void decode_layer_forward(Qwen3DecodeLayer *this_cpu, Tensor *hidden_state_cpu, Tensor *output_cpu, base::PosEmb position_embeddings_cpu,
                              Qwen3DecodeLayer *this_cuda, Tensor *hidden_state_cuda, Tensor *output_cuda, base::PosEmb position_embeddings_cuda)
    {
        LOG(INFO) << "Decode layer forward - CPU vs CUDA comparison";

        // Step 1: Save residual connection
        auto residual_cpu = hidden_state_cpu->clone();
        auto residual_cuda = hidden_state_cuda->clone();
        check_tensor(residual_cpu, residual_cuda, "decode_layer_residual_1");

        // Step 2: Input layer normalization
        LOG(INFO) << "  Checking input layer norm...";
        this_cpu->input_layernorm.forward(*hidden_state_cpu, *hidden_state_cpu);
        this_cuda->input_layernorm.forward(*hidden_state_cuda, *hidden_state_cuda);
        check_tensor(*hidden_state_cpu, *hidden_state_cuda, "decode_layer_input_layernorm");

        // Step 3: Self attention
        LOG(INFO) << "  Checking self attention...";
        self_attn_forward(&this_cpu->self_attn, hidden_state_cpu, hidden_state_cpu, position_embeddings_cpu,
                          &this_cuda->self_attn, hidden_state_cuda, hidden_state_cuda, position_embeddings_cuda);

        // Step 4: Add residual connection (first)
        LOG(INFO) << "  Checking first add operation...";
        this_cpu->add_op.forward(*hidden_state_cpu, residual_cpu, *hidden_state_cpu);
        this_cuda->add_op.forward(*hidden_state_cuda, residual_cuda, *hidden_state_cuda);
        check_tensor(*hidden_state_cpu, *hidden_state_cuda, "decode_layer_add_1");

        // Step 5: Save second residual connection
        residual_cpu = hidden_state_cpu->clone();
        residual_cuda = hidden_state_cuda->clone();
        check_tensor(residual_cpu, residual_cuda, "decode_layer_residual_2");

        // Step 6: Post attention layer normalization
        LOG(INFO) << "  Checking post attention layer norm...";
        this_cpu->post_attention_layernorm.forward(*hidden_state_cpu, *hidden_state_cpu);
        this_cuda->post_attention_layernorm.forward(*hidden_state_cuda, *hidden_state_cuda);
        check_tensor(*hidden_state_cpu, *hidden_state_cuda, "decode_layer_post_attention_layernorm");

        // Step 7: MLP
        LOG(INFO) << "  Checking MLP...";
        mlp_forward(&this_cpu->mlp, hidden_state_cpu, hidden_state_cpu,
                    &this_cuda->mlp, hidden_state_cuda, hidden_state_cuda);

        // Step 8: Add residual connection (second)
        LOG(INFO) << "  Checking second add operation...";
        this_cpu->add_op.forward(*hidden_state_cpu, residual_cpu, *output_cpu);
        this_cuda->add_op.forward(*hidden_state_cuda, residual_cuda, *output_cuda);
        check_tensor(*output_cpu, *output_cuda, "decode_layer_add_2");
    }

    void self_attn_forward(
        Qwen3SelfAttn *this_cpu, Tensor *hidden_state_cpu, Tensor *output_cpu, base::PosEmb position_embeddings_cpu,
        Qwen3SelfAttn *this_cuda, Tensor *hidden_state_cuda, Tensor *output_cuda, base::PosEmb position_embeddings_cuda)
    {
        LOG(INFO) << "    Self attention forward - CPU vs CUDA comparison";

        // Step 1: Prepare output tensors for Q, K, V projections
        std::vector<size_t> q_shape(hidden_state_cpu->shape());
        std::vector<size_t> kv_shape(hidden_state_cpu->shape());
        q_shape.back() = this_cpu->num_attention_heads * this_cpu->head_dim;
        kv_shape.back() = this_cpu->num_key_value_heads * this_cpu->head_dim;

        this_cpu->q_output = Tensor(q_shape, this_cpu->device_, false);
        this_cpu->k_output = Tensor(kv_shape, this_cpu->device_, true);
        this_cpu->v_output = Tensor(kv_shape, this_cpu->device_, true);

        this_cuda->q_output = Tensor(q_shape, this_cuda->device_, false);
        this_cuda->k_output = Tensor(kv_shape, this_cuda->device_, true);
        this_cuda->v_output = Tensor(kv_shape, this_cuda->device_, true);

        // Step 2: Q, K, V projections
        LOG(INFO) << "      Checking Q projection...";
        this_cpu->q_proj.forward(*hidden_state_cpu, this_cpu->q_output);
        this_cuda->q_proj.forward(*hidden_state_cuda, this_cuda->q_output);
        check_tensor(this_cpu->q_output, this_cuda->q_output, "q_proj_output");

        LOG(INFO) << "      Checking K projection...";
        this_cpu->k_proj.forward(*hidden_state_cpu, this_cpu->k_output);
        this_cuda->k_proj.forward(*hidden_state_cuda, this_cuda->k_output);
        check_tensor(this_cpu->k_output, this_cuda->k_output, "k_proj_output");

        LOG(INFO) << "      Checking V projection...";
        this_cpu->v_proj.forward(*hidden_state_cpu, this_cpu->v_output);
        this_cuda->v_proj.forward(*hidden_state_cuda, this_cuda->v_output);
        check_tensor(this_cpu->v_output, this_cuda->v_output, "v_proj_output");

        // Step 3: Reshape for multi-head attention
        q_shape.back() = this_cpu->num_attention_heads;
        q_shape.push_back(this_cpu->head_dim);
        kv_shape.back() = this_cpu->num_key_value_heads;
        kv_shape.push_back(this_cpu->head_dim);

        this_cpu->q_output.view(q_shape);
        this_cpu->k_output.view(kv_shape);
        this_cpu->v_output.view(kv_shape);
        this_cuda->q_output.view(q_shape);
        this_cuda->k_output.view(kv_shape);
        this_cuda->v_output.view(kv_shape);

        // Step 4: Transpose
        this_cpu->q_output.transpose(-3, -2);
        this_cpu->k_output.transpose(-3, -2);
        this_cpu->v_output.transpose(-3, -2);
        this_cuda->q_output.transpose(-3, -2);
        this_cuda->k_output.transpose(-3, -2);
        this_cuda->v_output.transpose(-3, -2);

        check_tensor(this_cpu->q_output, this_cuda->q_output, "q_after_transpose");
        check_tensor(this_cpu->k_output, this_cuda->k_output, "k_after_transpose");
        check_tensor(this_cpu->v_output, this_cuda->v_output, "v_after_transpose");

        // Step 5: Q and K normalization
        LOG(INFO) << "      Checking Q normalization...";
        this_cpu->q_norm.forward(this_cpu->q_output, this_cpu->q_output);
        this_cuda->q_norm.forward(this_cuda->q_output, this_cuda->q_output);
        check_tensor(this_cpu->q_output, this_cuda->q_output, "q_norm_output");

        LOG(INFO) << "      Checking K normalization...";
        this_cpu->k_norm.forward(this_cpu->k_output, this_cpu->k_output);
        this_cuda->k_norm.forward(this_cuda->k_output, this_cuda->k_output);
        check_tensor(this_cpu->k_output, this_cuda->k_output, "k_norm_output");

        // Step 6: Apply rotary position embeddings
        LOG(INFO) << "      Checking RoPE for Q...";
        kernel::get_rope_kernel(this_cpu->device_)(&this_cpu->q_output, position_embeddings_cpu.first, position_embeddings_cpu.second, &this_cpu->q_output, this_cpu->stream_);
        kernel::get_rope_kernel(this_cuda->device_)(&this_cuda->q_output, position_embeddings_cuda.first, position_embeddings_cuda.second, &this_cuda->q_output, this_cuda->stream_);
        check_tensor(this_cpu->q_output, this_cuda->q_output, "q_rope_output");

        LOG(INFO) << "      Checking RoPE for K...";
        kernel::get_rope_kernel(this_cpu->device_)(&this_cpu->k_output, position_embeddings_cpu.first, position_embeddings_cpu.second, &this_cpu->k_output, this_cpu->stream_);
        kernel::get_rope_kernel(this_cuda->device_)(&this_cuda->k_output, position_embeddings_cuda.first, position_embeddings_cuda.second, &this_cuda->k_output, this_cuda->stream_);
        check_tensor(this_cpu->k_output, this_cuda->k_output, "k_rope_output");

        // Step 7: Repeat K and V for multi-head attention
        LOG(INFO) << "      Checking repeat_kv...";
        this_cpu->repeat_kv(this_cpu->k_output, this_cpu->v_output);
        this_cuda->repeat_kv(this_cuda->k_output, this_cuda->v_output);
        check_tensor(this_cpu->k_output, this_cuda->k_output, "k_repeat_output");
        check_tensor(this_cpu->v_output, this_cuda->v_output, "v_repeat_output");

        // Step 8: Update KV cache
        LOG(INFO) << "      Checking KV cache update...";
        if (this_cpu->k_cache.empty())
        {
            this_cpu->k_cache = this_cpu->k_output;
            this_cpu->v_cache = this_cpu->v_output;
        }
        else
        {
            this_cpu->k_cache.cat(this_cpu->k_output, -2);
            this_cpu->v_cache.cat(this_cpu->v_output, -2);
        }

        if (this_cuda->k_cache.empty())
        {
            this_cuda->k_cache = this_cuda->k_output;
            this_cuda->v_cache = this_cuda->v_output;
        }
        else
        {
            this_cuda->k_cache.cat(this_cuda->k_output, -2);
            this_cuda->v_cache.cat(this_cuda->v_output, -2);
        }

        check_tensor(this_cpu->k_cache, this_cuda->k_cache, "k_cache");
        check_tensor(this_cpu->v_cache, this_cuda->v_cache, "v_cache");

        // Step 9: Attention computation
        LOG(INFO) << "      Checking attention weights computation...";
        Tensor attn_weights_cpu({this_cpu->num_attention_heads, this_cpu->k_cache.shape(-2), this_cpu->k_cache.shape(-2)}, this_cpu->device_);
        Tensor attn_weights_cuda({this_cuda->num_attention_heads, this_cuda->k_cache.shape(-2), this_cuda->k_cache.shape(-2)}, this_cuda->device_);

        this_cpu->k_cache.t();
        this_cuda->k_cache.t();

        this_cpu->mat_mul.forward(this_cpu->q_output, this_cpu->k_cache, attn_weights_cpu);
        this_cuda->mat_mul.forward(this_cuda->q_output, this_cuda->k_cache, attn_weights_cuda);
        check_tensor(attn_weights_cpu, attn_weights_cuda, "attn_weights_raw");

        // Step 10: Scale attention weights
        LOG(INFO) << "      Checking attention scaling...";
        this_cpu->mat_sc_mul.forward(attn_weights_cpu, this_cpu->scaling, attn_weights_cpu);
        this_cuda->mat_sc_mul.forward(attn_weights_cuda, this_cuda->scaling, attn_weights_cuda);

        // Add CUDA synchronization to ensure kernel completion
        if (this_cuda->device_ == Device::CUDA)
        {
            cudaDeviceSynchronize();
        }

        check_tensor(attn_weights_cpu, attn_weights_cuda, "attn_weights_scaled");

        // Step 11: Apply causal mask
        LOG(INFO) << "      Checking causal mask...";
        this_cpu->causal_mask.forward(attn_weights_cpu);
        this_cuda->causal_mask.forward(attn_weights_cuda);
        check_tensor(attn_weights_cpu, attn_weights_cuda, "attn_weights_masked");

        // Step 12: Apply softmax
        LOG(INFO) << "      Checking attention softmax...";
        this_cpu->softmax.forward(attn_weights_cpu, attn_weights_cpu);
        this_cuda->softmax.forward(attn_weights_cuda, attn_weights_cuda);
        check_tensor(attn_weights_cpu, attn_weights_cuda, "attn_weights_softmax");

        // Step 13: Apply attention to values
        LOG(INFO) << "      Checking attention output computation...";
        Tensor attn_output_cpu(this_cpu->q_output);
        Tensor attn_output_cuda(this_cuda->q_output);

        this_cpu->mat_mul_attn_output.forward(attn_weights_cpu, this_cpu->v_cache, attn_output_cpu);
        this_cuda->mat_mul_attn_output.forward(attn_weights_cuda, this_cuda->v_cache, attn_output_cuda);
        check_tensor(attn_output_cpu, attn_output_cuda, "attn_output_raw");

        // Step 14: Reshape and transpose back
        attn_output_cpu.transpose(-3, -2);
        attn_output_cuda.transpose(-3, -2);
        auto attn_out_shape = attn_output_cpu.shape();
        attn_out_shape.pop_back();
        attn_out_shape.back() *= this_cpu->head_dim;
        attn_output_cpu.reshape(attn_out_shape);
        attn_output_cuda.reshape(attn_out_shape);

        check_tensor(attn_output_cpu, attn_output_cuda, "attn_output_reshaped");

        // Step 15: Output projection
        LOG(INFO) << "      Checking output projection...";
        this_cpu->o_proj.forward(attn_output_cpu, *output_cpu);
        this_cuda->o_proj.forward(attn_output_cuda, *output_cuda);
        check_tensor(*output_cpu, *output_cuda, "self_attn_final_output");
    }

    void mlp_forward(
        Qwen3MLP *this_cpu, Tensor *hidden_state_cpu, Tensor *output_cpu,
        Qwen3MLP *this_cuda, Tensor *hidden_state_cuda, Tensor *output_cuda)
    {
        LOG(INFO) << "    MLP forward - CPU vs CUDA comparison";

        // Step 1: Prepare intermediate tensors
        auto intermediate_shape = hidden_state_cpu->shape();
        intermediate_shape.back() = this_cpu->intermediate_size;

        if (this_cpu->gate_state.shape() != intermediate_shape)
        {
            this_cpu->gate_state = Tensor(intermediate_shape, this_cpu->device_);
            this_cpu->up_state = Tensor(intermediate_shape, this_cpu->device_);
        }

        if (this_cuda->gate_state.shape() != intermediate_shape)
        {
            this_cuda->gate_state = Tensor(intermediate_shape, this_cuda->device_);
            this_cuda->up_state = Tensor(intermediate_shape, this_cuda->device_);
        }

        // Step 2: Up projection
        LOG(INFO) << "      Checking up projection...";
        this_cpu->up_proj.forward(*hidden_state_cpu, this_cpu->up_state);
        this_cuda->up_proj.forward(*hidden_state_cuda, this_cuda->up_state);
        check_tensor(this_cpu->up_state, this_cuda->up_state, "mlp_up_proj_output");

        // Step 3: Gate projection
        LOG(INFO) << "      Checking gate projection...";
        this_cpu->gate_proj.forward(*hidden_state_cpu, this_cpu->gate_state);
        this_cuda->gate_proj.forward(*hidden_state_cuda, this_cuda->gate_state);
        check_tensor(this_cpu->gate_state, this_cuda->gate_state, "mlp_gate_proj_output");

        // Step 4: SiLU activation on gate
        LOG(INFO) << "      Checking SiLU activation...";
        this_cpu->silu.forward(this_cpu->gate_state);
        this_cuda->silu.forward(this_cuda->gate_state);
        check_tensor(this_cpu->gate_state, this_cuda->gate_state, "mlp_silu_output");

        // Step 5: Element-wise multiplication
        LOG(INFO) << "      Checking element-wise multiplication...";
        this_cpu->ele_mul.forward(this_cpu->gate_state, this_cpu->up_state, this_cpu->gate_state);
        this_cuda->ele_mul.forward(this_cuda->gate_state, this_cuda->up_state, this_cuda->gate_state);
        check_tensor(this_cpu->gate_state, this_cuda->gate_state, "mlp_ele_mul_output");

        // Step 6: Down projection
        LOG(INFO) << "      Checking down projection...";
        this_cpu->down_proj.forward(this_cpu->gate_state, *output_cpu);
        this_cuda->down_proj.forward(this_cuda->gate_state, *output_cuda);
        check_tensor(*output_cpu, *output_cuda, "mlp_final_output");
    }
};

#endif
