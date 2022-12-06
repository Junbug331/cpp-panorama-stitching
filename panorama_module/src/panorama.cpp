#include <filesystem>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <atomic>
#include <algorithm>

#include <opencv2/features2d.hpp>
#include <spdlog/spdlog.h>

#include "panorama.hpp"

namespace PANORAMA
{
    bool readImages(std::string image_dir_path, std::vector<cv::Mat> &images)
    {
        namespace fs = std::filesystem;
        // Check if it is a valid direcory
        if (fs::is_directory(fs::status(image_dir_path)))
        {
            std::vector<std::pair<int, std::string>> image_names;
            for (const auto &iter : fs::directory_iterator(image_dir_path))
            {
                image_names.push_back({std::stoi(iter.path().stem().u8string()), iter.path().string()});
            }

            // Sort image names
            std::sort(image_names.begin(), image_names.end());

            for (const auto &e : image_names)
            {
                const std::string &img_path = e.second;
                cv::Mat img = cv::imread(img_path);

                if (img.empty())
                    throw std::runtime_error("Image can't be opened: " + img_path);
                images.push_back(img);
            }

            if (images.size() < 2)
                throw std::runtime_error("Not enough images found. Require 2 or more images.");
        }
        else
            throw std::runtime_error("Image dir path is not valid.");

        return true;
    }

    void findMatches(cv::Mat img1, cv::Mat img2, std::vector<cv::DMatch> &matches,
                     std::vector<cv::KeyPoint> &img1_kpts, std::vector<cv::KeyPoint> &img2_kpts)
    {
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        cv::Mat img1_desc, img2_desc;
        cv::Mat img1_gray, img2_gray;
        cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
        sift->detectAndCompute(img1_gray, cv::noArray(), img1_kpts, img1_desc);
        sift->detectAndCompute(img2_gray, cv::noArray(), img2_kpts, img2_desc);

        cv::BFMatcher BF_matcher;
        std::vector<std::vector<cv::DMatch>> initial_matches;
        BF_matcher.knnMatch(img1_desc, img2_desc, initial_matches, 2);

        matches.reserve(initial_matches.size());
        for (const auto &m : initial_matches)
        {
            if (m[0].distance < 0.75 * m[1].distance)
                matches.push_back(m[0]);
        }
    }

    void findHomography(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &src_kpts,
                        const std::vector<cv::KeyPoint> &dst_kpts, cv::Mat &H)
    {
        // If less than 4 matcehs, throw exception
        if (matches.size() < 4)
            throw std::runtime_error("Not enough matches found between the images.");

        std::vector<cv::Point2f> src_pts_2d, dst_pts_2d;
        src_pts_2d.reserve(matches.size());
        dst_pts_2d.reserve(matches.size());
        for (const auto &m : matches)
        {
            src_pts_2d.emplace_back(src_kpts[m.queryIdx].pt.x, src_kpts[m.queryIdx].pt.y);
            dst_pts_2d.emplace_back(dst_kpts[m.trainIdx].pt.x, dst_kpts[m.trainIdx].pt.y);
        }

        // Find homogrphy matrix
        H = cv::findHomography(src_pts_2d, dst_pts_2d, cv::noArray(), cv::RANSAC, 4.0);
    }

    std::tuple<int, int> convert_xy(int xt, int yt, int cx, int cy, float f)
    {
        /*
            theta_x = (x - cx) / f <- approximation, f needs to be set appropriately
            theta_y = (y - cy) / f <- not used in cylinder projection

            Projection
            x_cylin = sin(theta)
            y_cylin = (y - cy)/f, y is unaffected. Just shrink in scale.
            z_cylin = cos(theta)

            Unroll <- return of this function
            x' = f * (x_cylin/z_cylin) + cx
               = f * (sin(theta_x)/cos(theta_x)) + cx
               = f * tan(theta_x) + cx
               = f * tan((x - cx)/f)
            y' = f * (y_cylin/z_cylin) + yx
               = f * ((y - cy) / f) / cos((x - cx)/f)) + cy
               = (y - cy) / cos((x - cx)/f) + cy
        */
        // By approximating the angle(theta), a unit cylinder is being simulated.
        float theta_x = static_cast<float>(xt - cx) / f;

        return {f * tan(theta_x) + cx, (yt - cy) / cos(theta_x) + cy};
    }

    void projectOntoCylinder(const cv::Mat &initial_img, cv::Mat &transformed_img, cv::Mat &mask)
    {
        int h = initial_img.rows, w = initial_img.cols;
        int cx = w >> 1, cy = h >> 1;
        /*
            fov = 70
            pi*fov/180 = cx / f;
            f = cx / (pi*(fov/180))
        */
        // int f = static_cast<float>(cx) / (M_PI * (70. / 180.));
        int f = 1100;

        // Creating a blank transformed image
        transformed_img = cv::Mat::zeros(initial_img.size(), initial_img.type());
        mask = cv::Mat::zeros(transformed_img.size(), transformed_img.type());

        std::vector<int> Xt;
        cv::Mutex mu;

        // NOTE : Looping all coordinate of the "**transformed** image"
        cv::parallel_for_(cv::Range(0, h * w), [&](const cv::Range range)
                          {
                              int min_xt = INT_MAX;
                              for (int idx = range.start; idx < range.end; ++idx)
                              {
                                  // transformed coordinate
                                  int yt = idx / w;
                                  int xt = idx - yt * w;

                                  // Unrolled points, cylinder(3D) -> rectangular image plane(2D)
                                  // Finding corresponding coordinates of the transformed image in the "initial image"
                                  // Think of {xt, yt} as (sin(theta), h, cos(theta))
                                  // result xi, yi is a coordinate of image plane before it was projected onto cylinder
                                  auto [xi, yi] = convert_xy(xt, yt, cx, cy, f);
                                  int i_xi = static_cast<int>(xi);
                                  int i_yi = static_cast<int>(yi);

                                  // Exclude point that lies outside the the original image size
                                  if (i_xi < 0 || i_xi > w - 2 || i_yi < 0 || i_yi > h - 2)
                                      continue;

                                  min_xt = (xt < min_xt) ? xt : min_xt;
                                  mask.at<cv::Vec3b>(yt, xt) = {255, 255, 255};

                                  // Bilinear interpolation
                                  float dx = xi - static_cast<float>(i_xi);
                                  float dy = yi - static_cast<float>(i_yi);
                                  // weight tl(top left), tr(top right), bl(bottom left), br(bottom right)
                                  float w_tl = (1.0 - dy) * (1.0 - dx);
                                  float w_tr = (1.0 - dy) * dx;
                                  float w_bl = dy * (1.0 - dx);
                                  float w_br = dy * dx;

                                  cv::Vec3b &transformed_img_intensity = transformed_img.at<cv::Vec3b>(yt, xt);
                                  const cv::Vec3b &initial_img_intensity_tl = initial_img.at<cv::Vec3b>(i_yi, i_xi);
                                  const cv::Vec3b &initial_img_intensity_tr = initial_img.at<cv::Vec3b>(i_yi, i_xi + 1);
                                  const cv::Vec3b &initial_img_intensity_bl = initial_img.at<cv::Vec3b>(i_yi + 1, i_xi);
                                  const cv::Vec3b &initial_img_intensity_br = initial_img.at<cv::Vec3b>(i_yi + 1, i_xi + 1);

                                  for (int k = 0; k < initial_img.channels(); ++k)
                                  {
                                      transformed_img_intensity.val[k] = (w_tl * initial_img_intensity_tl.val[k]) +
                                                                         (w_tr * initial_img_intensity_tr.val[k]) +
                                                                         (w_bl * initial_img_intensity_bl.val[k]) +
                                                                         (w_br * initial_img_intensity_br.val[k]);
                                  }
                              }
                              mu.lock();
                              Xt.push_back(min_xt);
                              mu.unlock(); });

        // Getting x coordinate to remove black region from right and left in the transformed image
        int min_x = *std::min_element(Xt.begin(), Xt.end());

        // Cropping out the block region from both sides(using symmetricity)
        transformed_img = transformed_img(cv::Rect(min_x, 0, transformed_img.cols - min_x * 2, transformed_img.rows)).clone();
        mask(cv::Rect(min_x, 0, mask.cols - min_x * 2, mask.rows)).copyTo(mask);
    }

    void getNewFrameSizeAndMatrix(cv::Mat &H, int *sec_img_shape, int *base_img_shape, int *new_frame_size, int *correction)
    {
        auto [w, h] = std::make_tuple(sec_img_shape[0], sec_img_shape[1]);

        // Taking the matrix of initial coordinates of the corners of the secondary image
        // Homogeneous coordinate
        // | x1 x2 x3 x4 |
        // | y1 y2 y3 y4 |
        // | 1  1  1  1  |
        // {xi, yi, 1} is the coordinate of the i_th corner of the image.
        cv::Mat initial_matrix = cv::Mat_<double>({3, 4},
                                                  {0.0, double(w - 1), double(w - 1), 0.0,
                                                   0.0, 0.0, double(h - 1), double(h - 1),
                                                   1.0, 1.0, 1.0, 1.0});

        // Finding final coordinates of the corners of the image after transformation
        // Note that coordinate of the corners of the frame may go out of the frame(negative value)
        // This will be corrected by upating the homographyt matrix accordingly
        cv::Mat final_matrix = H * initial_matrix;

        cv::Mat x = final_matrix(cv::Rect(0, 0, final_matrix.cols, 1));
        cv::Mat y = final_matrix(cv::Rect(0, 1, final_matrix.cols, 1));
        cv::Mat z = final_matrix(cv::Rect(0, 2, final_matrix.cols, 1));

        // Normalize by z
        cv::divide(x, z, x);
        cv::divide(y, z, y);

        double min_x, max_x, min_y, max_y;
        cv::minMaxLoc(x, &min_x, &max_x);
        cv::minMaxLoc(y, &min_y, &max_y);

        int new_w = max_x, new_h = max_y;
        correction[0] = correction[1] = 0;

        // min_y, min_y > 0 means that it is overlapped on right hand side, meaning its max is the new max
        if (min_x < 0)
        {
            // -(min_x) -> +min_x
            new_w -= min_x;
            correction[0] = abs(min_x);
        }
        if (min_y < 0)
        {
            // -min_y -> +min_y
            new_h -= min_y;
            correction[1] = abs(min_y);
        }

        // Again correcting new_w and new_h
        // sec_img might be overlapped on the left hand side of the base_img
        new_w = (new_w < base_img_shape[0] + correction[0]) ? base_img_shape[0] + correction[0] : new_w;
        new_h = (new_h < base_img_shape[1] + correction[1]) ? base_img_shape[1] + correction[1] : new_h;

        cv::add(x, correction[0], x);
        cv::add(y, correction[1], y);

        // New homography matrix
        cv::Point2f old_initial_pts[4], new_final_pts[4];
        old_initial_pts[0] = {0.0, 0.0};
        old_initial_pts[1] = {float(w - 1), 0.0};
        old_initial_pts[2] = {float(w - 1), float(h - 1)};
        old_initial_pts[3] = {0.0, float(h - 1)};
        for (int i = 0; i < 4; ++i)
            new_final_pts[i] = {float(x.at<double>(0, i)), float(y.at<double>(0, i))};

        // Updating the homography matrix so that the secondary image completely lies inside the frame
        H = cv::getPerspectiveTransform(old_initial_pts, new_final_pts);

        new_frame_size[0] = new_w;
        new_frame_size[1] = new_h;
    }

    cv::Mat stitchImages(const cv::Mat &base_img, const cv::Mat &sec_img)
    {
        // Applying cylindrical projection on on sec_img
        cv::Mat sec_img_cyl, sec_img_mask;
        projectOntoCylinder(sec_img, sec_img_cyl, sec_img_mask);

        /// stitching direction : sec_img_cyl -> base_img

        // Finding matches between the 2 images and their keypoints
        std::vector<cv::DMatch> matches;
        std::vector<cv::KeyPoint> base_img_kpts, sec_img_kpts;
        findMatches(sec_img_cyl, base_img, matches, sec_img_kpts, base_img_kpts);

        // Finding homography matrix (sec_img -> base_img)
        cv::Mat H;
        findHomography(matches, sec_img_kpts, base_img_kpts, H);

        // Finding size of a new frame of stitched images and updating homography matrix
        // {width, height}
        int sec_img_shape[2] = {sec_img_cyl.cols, sec_img_cyl.rows};
        int base_img_shape[2] = {base_img.cols, base_img.rows};
        int new_frame_size[2], correction[2];
        getNewFrameSizeAndMatrix(H, sec_img_shape, base_img_shape, new_frame_size, correction);

        // Finally placing the images upon one another
        cv::Mat sec_img_transformed, sec_img_transformed_mask;
        cv::warpPerspective(sec_img_cyl, sec_img_transformed, H, {new_frame_size[0], new_frame_size[1]});
        cv::warpPerspective(sec_img_mask, sec_img_transformed_mask, H, {new_frame_size[0], new_frame_size[1]});

        cv::Mat base_img_transformed = cv::Mat::zeros({new_frame_size[0], new_frame_size[1]}, base_img.type());
        base_img.copyTo(base_img_transformed(cv::Rect(correction[0], correction[1], base_img.cols, base_img.rows)));

        cv::Mat res, temp;
        cv::bitwise_not(sec_img_transformed_mask, sec_img_transformed_mask);
        cv::bitwise_and(base_img_transformed, sec_img_transformed_mask, temp);
        cv::bitwise_or(sec_img_transformed, temp, res);

        return res;
    }
}
