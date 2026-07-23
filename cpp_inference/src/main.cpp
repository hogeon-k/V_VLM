#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

namespace {

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [--image <path>]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::cout << "PCB C++ inference environment check\n";
    std::cout << "OpenCV version: " << CV_VERSION << '\n';
    std::cout << "Argument count: " << argc << '\n';

    std::string image_path;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        std::cout << "argv[" << index << "]: " << argument << '\n';
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (argument == "--image") {
            if (index + 1 >= argc) {
                std::cerr << "Error: --image requires a path.\n";
                return 2;
            }
            image_path = argv[++index];
            std::cout << "argv[" << index << "]: " << image_path << '\n';
        }
    }

    if (image_path.empty()) {
        std::cout << "No image path provided. Environment check completed.\n";
        return 0;
    }

    const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: failed to load image: " << image_path << '\n';
        return 1;
    }

    std::cout << "Loaded image: " << image_path << '\n';
    std::cout << "Image size: " << image.cols << "x" << image.rows << '\n';
    return 0;
}
