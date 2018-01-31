#pragma once

#include <opencv2/video/background_segm.hpp>
#include <opencv2/features2d.hpp>
#include "objecttracker.h"

class DifferenceTracker {
public:
    DifferenceTracker();
    ~DifferenceTracker();

    void processFrame(cv::Mat& frame, const Spawns& spawns, std::vector<Contour>& contours );

private:
    cv::Ptr<cv::BackgroundSubtractor> diffEngine;
    // Skip some of the first few frames while training
    int skipped;
    cv::Mat maskImage;
};
