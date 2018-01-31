#include "track.h"
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <iostream>
size_t Track::instances = 0;
int countNoBlobFrames =0;
cv::KalmanFilter* KF = new cv::KalmanFilter(4,2,0);
bool firstTime = true;
cv::Point calcCentroid(DetectedObject& contour) {

    double xc = contour.bounding_box.x + contour.bounding_box.width/2;
    double yc = contour.bounding_box.y + contour.bounding_box.height/2;
    return {(int)std::round(xc), (int)std::round(yc)};
}
//Difference Tracker 
cv::Point calcCentroidDiff(Contour& contour) {
    cv::Moments moments = cv::moments(contour);
    double xc = moments.m10 / moments.m00;
    double yc = moments.m01 / moments.m00;
    return {(int)std::round(xc), (int)std::round(yc)};
}
void Track::inititializeKalman()
{ 

    cv::setIdentity(KF->transitionMatrix);

    // init...
    KF->statePre.at<float>(0) = 800;
    KF->statePre.at<float>(1) = 120;
    KF->statePre.at<float>(3) = 0;
    KF->statePre.at<float>(4) = 0;

    cv::setIdentity(KF->measurementMatrix);
    cv::setIdentity(KF->processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(KF->measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(KF->errorCovPost, cv::Scalar::all(.1));
    
}
double getAssignCost(std::unique_ptr<Track>& track, DetectedObject& contour) {
    TrackingData data = calcCentroid(contour);
    const TrackingData& prediction = track->getPrediction();
    return sqrt(pow(data.x - prediction.x, 2) + pow(data.y - prediction.y, 2));
}
//Difference Tracker 
double getAssignCostDiff(std::unique_ptr<Track>& track, Contour& contour) {
    TrackingData data = calcCentroidDiff(contour);
    const TrackingData& prediction = track->getPrediction();
    return sqrt(pow(data.x - prediction.x, 2) + pow(data.y - prediction.y, 2));
}

void Track::assignTracks(Tracks& tracks, std::vector<DetectedObject>& contours, std::vector<Contour>& diffContours) {
    // TODO Implement Munkres/Hungarian algorithm to solve
    // the cost assignment problem in O(n^3) rather t
    //std::cout<<"track size is : "<<tracks.size()<<std::endl;
    int trackno=0;
    for(std::unique_ptr<Track>& track : tracks) {
        double minCost = costNonassignment;
        int minIndex = -1;

        for(int i = 0; i < contours.size(); i++) {
            double cost = getAssignCost(track, contours[i]);
            if(cost < minCost) {
                minCost = cost;
                minIndex = i;
            }
        }
        for(int i = 0; i < diffContours.size(); i++) {
            double cost = getAssignCostDiff(track, diffContours[i]);   
            if(cost <= minCost && cost !=0) {
                minCost = cost;
                minIndex = contours.size() + i;
            }
        }

        if(minIndex == -1) {
            // No contour found, so track becomes invisible
            track->update();
        } else {
            if(minIndex < contours.size() )
            {
                track->update(contours[minIndex]);
                contours.erase(contours.begin() + minIndex);
            }
            else
            {
                track->updateDiff(diffContours[minIndex-contours.size()]);
                diffContours.erase(diffContours.begin() + minIndex -contours.size());   
            }
            
        }
        trackno++;
    }

}
//Difference Tracker 
void Track::assignTracksDiff(Tracks& tracks, std::vector<Contour>& contours) {
    // TODO Implement Munkres/Hungarian algorithm to solve
    // the cost assignment problem in O(n^3) rather t
    int trackno=0;
    for(std::unique_ptr<Track>& track : tracks) {
        double minCost = costNonassignment;
        int minIndex = -1;
        for(int i = 0; i < contours.size(); i++) {
            double cost = getAssignCostDiff(track, contours[i]);
            //std::cout<<"Cost"<<"\t"<<cost<<std::endl;    
            if(cost <= minCost && cost !=0) {
                minCost = cost;
                minIndex = i;
            }
          /*  if(costChangeAssignment<cost && track->getAge() > ageChangeAssignment)
            {
                minIndex = i;
            }*/
        }
        //std::cout<<"age of track"<<"\t"<<track->getAge()<<std::endl;
        //Deleteing the track with No Blobs 
       /* if(contours.size()==0 && track->getAge() > ageTrackDelete)
        {
            countNoBlobFrames++;
        }
        if(contours.size() !=0)
            countNoBlobFrames = 0;
        if(countNoBlobFrames>20)
        {
            countNoBlobFrames = 0;
            tracks.erase(tracks.begin()+trackno);
            break;
        }
*/
        if(minIndex == -1) {
            // No contour found, so track becomes invisible
            track->update();
        } else {
            track->updateDiff(contours[minIndex]);
            contours.erase(contours.begin() + minIndex);
        }
        //std::cout<<"Counter Size"<<"\t"<<contours.size()<<std::endl;
        trackno++;
    }
}
size_t Track::getNextIndex() {
    return instances + 1;
}
Track::Track()
{
    inititializeKalman();
}
Track::Track(DetectedObject& contour, int id) : id(id), age(0), visibleCount(0), invisibleAge(0), visible(true) {
    instances++;
    update(contour);
}
Track::Track(Contour& contour, int id) : id(id), age(0), visibleCount(0), invisibleAge(0), visible(true) {
    instances++;
    updateDiff(contour);
}
Track::~Track() {
}

int Track::getId() {
    return id;
}

int Track::getAge() {
    return age;
}

int Track::getVisibleCount() {
    return visibleCount;
}

int Track::getInvisibleAge() {
    return invisibleAge;
}

bool Track::isVisible() {
    return visible;
}

const cv::Rect& Track::getBBox() {
    return bbox;
}

const DetectedObject& Track::getContour() {
    return contour;
}
const Contour& Track::getContourDiff() {
    return contourDiff;
}

const TrackingData& Track::getPrediction() {
    return prediction;
}

void Track::update() {
    visible = false;
    age++;
    invisibleAge++;
}

void Track::update(DetectedObject& new_contour) {
    // Update metadata
    visible = true;
    age++;
    visibleCount++;
    invisibleAge = 0;
    // Update filter
    contour = new_contour;
    bbox = contour.bounding_box; //cv::boundingRect(contour);
    cv::Point center = calcCentroid(contour);
    // TODO use Kalman filter
    //cv::Mat pred = KF->predict();
    prediction = center; 
    contourDiff = {};
    //Adding noise to measurement 
    //std::randn( measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));
    // KF->measurementMatrix.at<float>(0) = center.x;
    // KF->measurementMatrix.at<float>(1) = center.y;

    // cv::Mat estimated = KF->correct(KF->measurementMatrix);
    // prediction = cv::Point(estimated.at<float>(0),estimated.at<float>(1));
}
void Track::updateDiff(Contour& new_contour) {
    // Update metadata
    visible = true;
    age++;
    visibleCount++;
    invisibleAge = 0;
    // Update filter
    contourDiff = new_contour;
    bbox = cv::boundingRect(contourDiff);
    cv::Point center = calcCentroidDiff(contourDiff);
    contour = {};
    // TODO use Kalman filter
  /*  if(firstTime)
    {
        KF->statePre.at<float>(0) = center.x;
        KF->statePre.at<float>(1) = center.y;
        KF->statePost.at<float>(0) = center.x;
        KF->statePost.at<float>(1) = center.y;
        KF->statePost.at<float>(2) = 0;
        KF->statePost.at<float>(3) = 0;
        firstTime = false;
    }*/
    //cv::Mat pred = KF->predict();
    prediction = center;
    // KF->statePre.at<float>(0) = center.x;
    // KF->statePre.at<float>(1) = center.y;   
    //std::cout<<"I am after predict"<<std::endl;

    // //Adding noise to measurement 
    // //std::randn( measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));
    // KF->measurementMatrix.at<float>(0) = center.x;
    // KF->measurementMatrix.at<float>(1) = center.y;
    // std::cout<<"I am after measure assign"<<std::endl;
    // cv::Mat estimated = KF->correct(KF->measurementMatrix);
    // std::cout<<"I am after correct"<<std::endl;
    // prediction = cv::Point(estimated.at<float>(0),estimated.at<float>(1));
}