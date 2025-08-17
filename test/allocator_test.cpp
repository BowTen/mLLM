#include <gtest/gtest.h>
#include <vector>
#include "base/allocator.h"

using namespace mllm::base;

TEST(HostAllocatorTest, HostAllocatorBasic)
{
    Allocator *allocator = HostAllocator::getInstance();
    ASSERT_NE(allocator, nullptr);

    size_t size = 1024;
    void *ptr = allocator->allocate(size);
    ASSERT_NE(ptr, nullptr);

    allocator->deallocate(ptr);
}

TEST(HostAllocatorTest, AllocateOneGB)
{
    Allocator *allocator = HostAllocator::getInstance();
    ASSERT_NE(allocator, nullptr);

    size_t size = 1024;
    std::vector<void *> ptrs(1024 * 1024);
    for (auto &ptr : ptrs)
    {
        ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr);
    }

    for (auto &ptr : ptrs)
    {
        allocator->deallocate(ptr);
    }
}

TEST(CudaAllocatorTest, CudaAllocatorBasic)
{
    Allocator *allocator = CudaAllocator::getInstance();
    ASSERT_NE(allocator, nullptr);

    size_t size = 1024;
    void *ptr = allocator->allocate(size);
    ASSERT_NE(ptr, nullptr);

    allocator->deallocate(ptr);
}

TEST(CudaAllocatorTest, AllocateOneGB)
{
    Allocator *allocator = CudaAllocator::getInstance();
    ASSERT_NE(allocator, nullptr);

    size_t size = 1024;
    std::vector<void *> ptrs(1024 * 1024);
    for (auto &ptr : ptrs)
    {
        ptr = allocator->allocate(size);
        ASSERT_NE(ptr, nullptr);
    }

    for (auto &ptr : ptrs)
    {
        allocator->deallocate(ptr);
    }
}