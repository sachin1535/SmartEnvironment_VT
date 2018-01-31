#include "camera.h"
#include "config.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

Camera::Camera(int id, std::string url) : id(id), video(url) {
    Config& config = Config::get();
    cropping = config.isCameraDefined(id);
	CameraInfo camInfo = config.getCameraInfo(id);
    if(cropping) {
        
        crop = camInfo.crop;
    }
	ang = camInfo.angle;
	std::cout<<ang*2<<std::endl;
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
	if(ang!=0)
	{
		cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point(frameRot.cols/2, frameRot.rows/2), ang, 1);
	    cv::warpAffine(frameRot, frame, rot_mat, cv::Size(frameRot.rows, frameRot.cols)); 
	}
	else 
		frame =	frameRot;
    if(cropping) {
        frame = frame(crop);
    }

    return true;
}
