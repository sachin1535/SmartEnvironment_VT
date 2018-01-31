#pragma once

#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "../yolo++/object.h"
#include <climits>
class Track;
typedef cv::Point TrackingData;
typedef std::vector<cv::Point> Contour;
typedef std::vector<std::unique_ptr<Track>> Tracks;

cv::Point calcCentroid(DetectedObject& contour);
cv::Point calcCentroidDiff(Contour& contour);

class Track {
public:
    static const int costNonassignment = 50;
    static const int ChangeAssignment = 50;
    static const int ageChangeAssignment = 100;
    static const int ageTrackDelete = 200;

    // Will modify the contour vector - whatever remains was not assigned to a track
    static void assignTracks(Tracks& tracks, std::vector<DetectedObject>& contours, std::vector<Contour>& diffContours);
    static void assignTracksDiff(Tracks& tracks, std::vector<Contour>& contours);
    static size_t getNextIndex();
    Track();
    Track(DetectedObject& contour, int id = getNextIndex());
    Track(Contour& contour, int id = getNextIndex());
    ~Track();

    int getId();
    int getAge();
    int getVisibleCount();
    int getInvisibleAge();
    bool isVisible();
    const cv::Rect& getBBox();
    const DetectedObject& getContour();
    const Contour& getContourDiff();
    const TrackingData& getPrediction();
    // This track was not assigned a contour, so increase age and mark as invisible
    void update();
    // This track was assigned a contour, so increase age and update filter
    void update(DetectedObject& new_contour);
    void updateDiff(Contour& new_contour);
    void inititializeKalman();
    static size_t instances;
private:
    

    int id;
    int age;
    int visibleCount;
    int invisibleAge;
    bool visible;
    cv::Rect bbox;
    DetectedObject contour;
    Contour contourDiff;
    TrackingData prediction;
    
};
