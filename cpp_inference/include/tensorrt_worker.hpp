#pragma once

#include <string>
#include <vector>

namespace pcb_vision {

int run_tensorrt_worker(
    const std::string& engine_path,
    const std::vector<std::string>& class_names,
    const std::string& engine_label,
    int image_size,
    int device_id
);

}  // namespace pcb_vision
