#include <iostream>
#include <filesystem>
#include <string>

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

#include "panorama.hpp"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

int main()
{
    fs::path image_path = fs::path(RES_DIR) / "Field";
    std::vector<cv::Mat> images;
    PANORAMA::readImages(image_path, images);

    Mat base_img, dummy;
    PANORAMA::projectOntoCylinder(images[0], base_img, dummy);

    for (int i = 1; i < images.size(); ++i)
    {
        Mat stitched_img = PANORAMA::stitchImages(base_img, images[i]);
        stitched_img.copyTo(base_img);
    }
    imwrite(string(ROOT_DIR) + "/panorama.jpg", base_img);

    return 0;
}