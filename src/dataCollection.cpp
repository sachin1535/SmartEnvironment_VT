#include "dataCollection.h"
#include <iostream>

//Collection* Collection::instance = nullptr; 
Collection::FPS::FPS(int size) : prevTick(0), size(size), index(0), sum(0), total(0) {
    ticks = new std::int64_t[size];
    std::fill(ticks, ticks + size, 0);
}

Collection::FPS::~FPS() {
    delete[] ticks;
}

void Collection::FPS::update(std::int64_t newTick) {
    if(total < size) {
        total++;
    }
    if(total == 1) {
        prevTick = newTick;
        return;
    }
    std::int64_t tickDiff = newTick - prevTick;
    prevTick = newTick;
    sum -= ticks[index];
    sum += tickDiff;
    ticks[index] = tickDiff;
    index = (index == size - 1 ? 0 : index + 1);
}

double Collection::FPS::getFPS() {
    return (cv::getTickFrequency() / (sum * 1.0 / total));
}

Collection::Collection(std::string filename) : file(filename), fps(100) {

   file << "detections" <<"\t"<< "cameraID"<< "\t"<< "xmin"<< "\t"<<"ymin"<< "\t"<<"xmax"<< "\t"<<"ymax"<< "\t"<<"area"<< "\t"<<"frame_no"<<std::endl;
   fpsfile<<"frame_no"<<"\t"<<"rate"<<std::endl;
}

Collection::~Collection() {
}

/*Collection& Collection::get() {
    if(!instance) {
        instance = new Config("trackOutput.txt");
    }
    return *instance;
}*/
void Collection::getFramerateData(int frame_no)
{
    fps.update(cv::getTickCount());
    double fpsValue = fps.getFPS();
    //std::cout<<fpsValue<<std::endl;
    fpsfile<<frame_no<<"\t"<<fpsValue<<std::endl;

}
void Collection::writeData(int id,int frame_no,ObjectTracker* tracker)
{
    Blob blobData;
    //Getting parameters from config file

    Config& config = Config::get();
    bool cropping = config.isCameraDefined(id);
    cv::Rect crop;
    if(cropping) {
        CameraInfo camInfo = config.getCameraInfo(id);
        crop = camInfo.crop;
    }
    memset(&blobData, 0, sizeof(blobData));
    blobData.cameraID = id;

    // Send deleted blobs first
    for(int id : tracker->getDeletedTracks()) {
        blobData.id = id;
        //sender.sendRemoveBlob(&blobData);
    }

    if(tracker->getTracks().size()==0)
    {
        file << 0 << "\t"<< 0 <<"\t"<< 0 <<"\t"<< 0 <<"\t"<< 0 <<"\t"<< 0 <<"\t"<< 0 <<"\t"<<frame_no <<std::endl;
    }
    else
    {
        // Then send new/updated blobs
        for(const std::unique_ptr<Track>& track : tracker->getTracks()) {
            blobData.id = track->getId();
            const cv::Rect& bbox = track->getBBox();
            blobData.bounding_x = bbox.x;
            blobData.bounding_y = bbox.y;
            blobData.bounding_width = bbox.width;
            blobData.bounding_height = bbox.height;
            blobData.origin_x = bbox.x + (bbox.width * 0.5);
            blobData.origin_y = bbox.y + (bbox.height * 0.5);
            blobData.area = (bbox.width*bbox.height);//cv::contourArea(track->getContour());
            file << tracker->getTracks().size() <<"\t"<< blobData.id << "\t"<<  int((crop.x + blobData.bounding_x)) << "\t"<< int((blobData.bounding_y + crop.y)) << "\t"<< int((crop.x +blobData.bounding_x + blobData.bounding_width)) << "\t"<< int((blobData.bounding_height + blobData.bounding_y+ crop.y)) << "\t"<<  blobData.area << "\t" << frame_no <<std::endl;
            //std::cout<<tracker->getTracks().size() <<"\t"<< blobData.id << "\t"<<  int((crop.x + blobData.bounding_x)/2) << "\t"<< int((blobData.bounding_y + crop.y)/1.655172414) << "\t"<< int((crop.x +blobData.bounding_x + blobData.bounding_width)/2) << "\t"<< int((blobData.bounding_height + blobData.bounding_y+ crop.y)/1.655172414) << "\t"<<  blobData.area << "\t" << frame_no <<std::endl;
        }    
    }
    
}