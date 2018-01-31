#pragma once

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

class Camera {
public:
    Camera(int id, std::string url);

    int getId();
    bool getFrame(cv::Mat& frame);

private:
    int id;
    cv::VideoCapture video;
    cv::Rect crop;
    bool cropping;
	int ang;
};
