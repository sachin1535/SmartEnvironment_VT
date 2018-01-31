#include "camera.h"
#include "config.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
Camera::Camera(int id, std::string url) : id(id), video(url) {
    Config& config = Config::get();
    cropping = config.isCameraDefined(id);
	
    if(cropping) {
        CameraInfo camInfo = config.getCameraInfo(id);
        crop = camInfo.crop;
        spawns = camInfo.spawns;
        ang = camInfo.angle;
    }
}

int Camera::getId() {
    return id;
}

bool Camera::getFrame(cv::Mat& frame) {
	cv::Mat frameRot;
    bool status = video.read(frameRot);

    if(!status) {
        return false;
    }
    // std::cout<<frameRot.cols <<"\t"<<frameRot.rows<<"before" <<std::endl; 
	if(ang!=0)
	{
		cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point(frameRot.cols/2, frameRot.rows/2), ang, 1);
	    cv::warpAffine(frameRot, frame, rot_mat, cv::Size(frameRot.rows, frameRot.cols)); 
	}
	else 
		frame =	frameRot;
    // std::cout<<frame.cols <<"\t"<<frame.rows <<"after"<<std::endl; 
    cv::resize(frame, frame, cv::Size(frame.cols/2, frame.rows/2));
    // cv::resize(frame, frame, cv::Size(528,480));
    if(cropping) {
        frame = frame(crop);
    }
    
    return true;
}
Spawns& Camera::getSpawns() {
    return spawns;
}
