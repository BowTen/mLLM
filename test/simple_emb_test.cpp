#include "base/tensor.h"
#include <iostream>
#include <vector>

int main()
{
    try
    {
        std::cout << "Creating tensor..." << std::endl;
        std::vector<size_t> shape = {5};
        mllm::base::Tensor tensor(shape, mllm::base::Device::CPU);
        std::cout << "Tensor created successfully!" << std::endl;

        float *data = tensor.data();
        std::cout << "Got data pointer: " << data << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
