#include "base/util.h"
#include <fstream>

namespace mllm
{
    namespace base
    {
        json load_json(const std::string &file_path)
        {
            std::ifstream file(file_path);
            if (!file.is_open())
            {
                throw std::runtime_error("Could not open file: " + file_path);
            }
            json j;
            file >> j;
            return j;
        }
    }
}