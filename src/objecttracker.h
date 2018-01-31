#pragma once

#include <unordered_set>
#include <vector>
#include <opencv2/core/mat.hpp>
#include "track.h"
#include "../yolo++/object.h"
#include "camera.h"
class ObjectTracker {
public:
    ~ObjectTracker();
    
    //virtual void processFrame(cv::Mat& frame,const Spawns& spawns) = 0;
    const Tracks& getTracks();
    // Will clear the list of deleted tracks, so subsequent calls return an empty list.
    const std::unordered_set<int> getDeletedTracks();
    // cv::Mat& getMaskImage();

public:
    Tracks tracks;
    std::unordered_set<int> deletedTracks;
    
    void startupdate(Tracks& tracks);
    void processContours(Tracks& tracks, std::vector<DetectedObject>& contours, std::vector<Contour>& spawnContours, const Spawns& spawns);
    void processContoursDiff(Tracks& tracks, std::vector<Contour>& contours,  const Spawns& spawns);
};
