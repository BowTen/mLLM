# Unified BF16 Inference Foundation Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land OpenSpec tasks `1.1`-`1.3` by making the tensor runtime explicitly dtype-aware without changing Qwen3 behavior yet.

**Architecture:** Add an explicit runtime dtype enum and byte-size helpers in the base runtime, thread dtype through `TensorMeta`/`Tensor`, and replace `sizeof(float)` sizing and implicit `float *` access with explicit typed/raw access helpers plus compatibility shims. Keep existing float-based callers working during this slice so later weight-loading and operator migrations can build on a stable runtime contract.

**Tech Stack:** C++17, CUDA runtime, GoogleTest, existing `base`/`kernel` libraries, CMake.

**Rollout note:** Phase 1 introduces the dtype infrastructure and explicit BF16 storage support, but it does **not** flip the generic runtime default to BF16 yet. The global default switch is deferred to OpenSpec task `4.3` after parity validation.

---

### Task 1: Add dtype metadata and default floating-point policy

**Files:**
- Create: `include/base/dtype.h`
- Modify: `include/base/common.h`
- Modify: `include/base/tensor.h`
- Modify: `src/base/tensor.cpp`
- Test: `test/tensor_test.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
TEST_F(TensorTest, DefaultFloatingPointTensorUsesProjectDefaultDType)
{
    Tensor tensor({2, 3}, Device::CPU, false, nullptr);

    EXPECT_EQ(tensor.dtype(), default_float_dtype());
    EXPECT_EQ(tensor.element_size(), dtype_element_size(default_float_dtype()));
    EXPECT_TRUE(is_floating_point_dtype(tensor.dtype()));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target tensor_test -j2 && ./build/test/tensor_test --gtest_filter=TensorTest.DefaultFloatingPointTensorUsesProjectDefaultDType`
Expected: FAIL because `Tensor::dtype()`, `Tensor::element_size()`, or `DType::BF16` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
enum class DType
{
    FP32,
    BF16,
    U32,
};

inline DType default_float_dtype()
{
    return DType::FP32;
}
```

Thread `dtype_` through `TensorMeta` and `Tensor`, default floating-point constructors to `default_float_dtype()`, and expose `dtype()` / `element_size()` accessors. Keep the policy centralized so task `4.3` can flip it later without reworking the tensor ABI.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target tensor_test -j2 && ./build/test/tensor_test --gtest_filter=TensorTest.DefaultFloatingPointTensorUsesProjectDefaultDType`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/base/dtype.h include/base/common.h include/base/tensor.h src/base/tensor.cpp test/tensor_test.cpp
git commit -m "feat: add tensor dtype metadata"
```

### Task 2: Make tensor storage sizing dtype-aware

**Files:**
- Modify: `include/base/tensor.h`
- Modify: `src/base/tensor.cpp`
- Modify: `include/base/buffer.h`
- Modify: `src/base/buffer.cpp`
- Modify: `src/base/util.cpp`
- Modify: `src/kernel/cpu/contiguous_kernel.cpp`
- Modify: `src/kernel/cuda/contiguous_kernel.cu`
- Test: `test/tensor_test.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
TEST_F(TensorTest, MutableTensorReserveAndResizeUseDTypeSizedStorage)
{
    Tensor tensor({2, 2}, Device::CPU, true, nullptr, DType::BF16);

    tensor.reserve(10);
    tensor.resize(6);

    auto buffer = std::dynamic_pointer_cast<VecBuffer>(tensor.buffer());
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(buffer->size(), 6 * sizeof(uint16_t));
    EXPECT_GE(buffer->capacity(), 10 * sizeof(uint16_t));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target tensor_test -j2 && ./build/test/tensor_test --gtest_filter=TensorTest.MutableTensorReserveAndResizeUseDTypeSizedStorage`
Expected: FAIL because reserve/resize still use `sizeof(float)` and the dtype-aware constructor is missing or ignored.

- [ ] **Step 3: Write minimal implementation**

```cpp
size_t dtype_size(DType dtype);

size_t Tensor::size() const
{
    return meta_->buffer_->size() / element_size();
}
```

Update tensor construction, `toDevice()`, `cat()`, `push()`, `reserve()`, and `resize()` to move bytes using `element_size()` instead of `sizeof(float)`.
Update CPU/CUDA contiguous materialization to resize destination buffers by `element_size()` and copy elements through dtype-aware access rather than assuming `float`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target tensor_test -j2 && ./build/test/tensor_test --gtest_filter=TensorTest.MutableTensorReserveAndResizeUseDTypeSizedStorage`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/base/tensor.h src/base/tensor.cpp include/base/buffer.h src/base/buffer.cpp src/base/util.cpp test/tensor_test.cpp
git commit -m "feat: size tensor storage by dtype"
```

### Task 3: Add explicit typed/raw access helpers with compatibility shims

**Files:**
- Modify: `include/base/tensor.h`
- Modify: `src/base/tensor.cpp`
- Modify: `include/base/util.h`
- Modify: `src/base/util.cpp`
- Test: `test/tensor_test.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
TEST_F(TensorTest, RawAndTypedAccessRespectTensorDType)
{
    Tensor tensor({2, 2}, Device::CPU, false, nullptr, DType::BF16);

    auto *raw = tensor.raw_data();
    auto *bf16 = tensor.data<uint16_t>();

    ASSERT_NE(raw, nullptr);
    ASSERT_EQ(raw, static_cast<void *>(bf16));
    EXPECT_EQ(tensor.compatible_float_data(), nullptr);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target tensor_test -j2 && ./build/test/tensor_test --gtest_filter=TensorTest.RawAndTypedAccessRespectTensorDType`
Expected: FAIL because explicit typed/raw accessors do not exist and legacy float access cannot distinguish non-fp32 storage.

- [ ] **Step 3: Write minimal implementation**

```cpp
void *Tensor::raw_data();

template <typename T>
T *Tensor::data()
{
    CHECK(dtype_matches<T>(meta_->dtype_));
    return static_cast<T *>(meta_->buffer_->data());
}
```

Retain a controlled compatibility path for legacy float callers, but make it explicit and guarded so later operator migrations can move away from implicit `float *` assumptions.
Also make `Tensor::from_vector<T>` choose a storage dtype compatible with `T` instead of silently tagging every vector-backed tensor as the default floating-point dtype.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target tensor_test -j2 && ./build/test/tensor_test --gtest_filter=TensorTest.RawAndTypedAccessRespectTensorDType`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/base/tensor.h src/base/tensor.cpp include/base/util.h src/base/util.cpp test/tensor_test.cpp
git commit -m "feat: add explicit tensor access helpers"
```

### Task 4: Verify the phase-1 runtime slice and update OpenSpec task status

**Files:**
- Modify: `openspec/changes/unified-bf16-inference-foundation/tasks.md`
- Test: `test/tensor_test.cpp`
- Test: `test/buffer_test.cpp`
- Test: `test/allocator_test.cpp`

- [ ] **Step 1: Run focused regression tests**

Run: `cmake --build build --target allocator_test buffer_test tensor_test -j2`
Expected: build succeeds

- [ ] **Step 2: Run the test binaries**

Run: `./build/test/allocator_test && ./build/test/buffer_test && ./build/test/tensor_test`
Expected: PASS

- [ ] **Step 3: Mark completed OpenSpec tasks**

Update `openspec/changes/unified-bf16-inference-foundation/tasks.md`:

```md
- [x] 1.1 Add explicit dtype metadata, element-size helpers, and default floating-point dtype policy to the tensor runtime.
- [x] 1.2 Make tensor allocation, clone, reshape, contiguous, cat, reserve, resize, and device-transfer paths size buffers by dtype rather than `sizeof(float)`.
- [x] 1.3 Introduce explicit typed/raw tensor access helpers and migrate core runtime utilities away from implicit `float*` assumptions.
```

- [ ] **Step 4: Commit**

```bash
git add openspec/changes/unified-bf16-inference-foundation/tasks.md
git commit -m "docs: update bf16 foundation task progress"
```
