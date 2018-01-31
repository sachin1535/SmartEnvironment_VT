#pragma once

#include <cstdint>
#include <string>
#include <opencv2/core/mat.hpp>
#include "track.h"
#include "../yolo++/object.h"
#include "camera.h"
class Display {
public:
    Display(int cameraId);
    ~Display();

    void showFrame(cv::Mat& frame, const Tracks& tracks,const Spawns& spawns, bool paused);

private:
    class FPS {
    public:
        FPS(int size);
        ~FPS();

        void update(std::int64_t newTick);
        double getFPS();

    private:
        std::int64_t prevTick;
        int size;
        int index;
        int total;
        std::uint64_t sum;
        std::int64_t *ticks;
    };

    std::string imageWinTitle;
    //std::string blobWinTitle;
    cv::Mat buffer;
    cv::Mat rgbMaskImage;
    cv::Point mousePos;
    FPS fps;
};
