#ifndef PANORAMA_HPP
#define PANORAMA_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <tuple>

namespace PANORAMA
{

    bool readImages(std::string image_dir_path, std::vector<cv::Mat> &images);
    void findMatches(cv::Mat base_img, cv::Mat sec_img, std::vector<cv::DMatch> &matches,
                     std::vector<cv::KeyPoint> &base_kpts, std::vector<cv::KeyPoint> &sec_kpts);
    void findHomography(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &base_kpts,
                        const std::vector<cv::KeyPoint> &sec_kpts, cv::Mat &H);

    /**
     * @brief Unrolled coordinate of cylinder coordinate.
     *
     * @param xt
     * @param yt
     * @param cx
     * @param cy
     * @param f
     * @return std::tuple<int, int>
     */
    std::tuple<int, int> convert_xy(int xt, int yt, int cx, int cy, float f);
    void projectOntoCylinder(const cv::Mat &initial_img, cv::Mat &transformed_img, cv::Mat &mask);
    void getNewFrameSizeAndMatrix(cv::Mat &H, int *sec_img_shape, int *base_img_shape, int *new_frame_size, int *correction);
    cv::Mat stitchImages(const cv::Mat &base_img, const cv::Mat &sec_img);

}

#endif