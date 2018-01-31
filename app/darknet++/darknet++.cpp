#include <fstream>
#include <cstdlib>
#include <string>
#include <memory>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "config.h"
#include "camera.h"
#include <yolo.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "objecttracker.h"
#include "track.h"
#include "blobSender.h"
#include "blob.h"
#include "display.h"
#include "difference.h"
#include "dataCollection.h"
Yolo yolo;
const double targetFPS = 30.0;
const double targetSleep = 1000.0 / targetFPS;
void sendTracks(int cameraId, cv::Size imgSize, ObjectTracker* tracker, blobSender& sender) {
    Blob blobData;
    memset(&blobData, 0, sizeof(blobData));
    blobData.cameraID = cameraId;

    // Send deleted blobs first
    for(int id : tracker->getDeletedTracks()) {
        blobData.id = id;
        sender.sendRemoveBlob(&blobData);
    }

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

        blobData.image_width = imgSize.width;
        blobData.image_height = imgSize.height;

        if(track->getAge() == 1) {
            sender.sendNewBlob(&blobData);
        } else {
            sender.sendUpdateBlob(&blobData);
        }
    }
    //std::cout<<"width"<<imgSize.width<<"\t"<<"Height"<<imgSize.height <<std::endl;
}
void printUsage(std::string progName) {
    std::cout << "Usage: " << progName << " <id> [cameraID]" <<" <VideoFilePath / IPAddress> [path]" <<" <Initialization Time > [secs]" <<" <resultsFileName> [string]" << std::endl;
    std::cout << std::endl;
    std::cout << "Where [path] is optional and can be an IP address, a URL, or a video file." << std::endl;
    std::cout << "If [path] is an IP address, the correct URL will be guessed based on the camera ID." << std::endl;
    std::cout << "If [path] is omitted, camera parameters will be read from the configuration file." << std::endl;
}

std::string parseURL(int camId, std::string arg) {
    in_addr ipaddr;
    if(inet_pton(AF_INET, arg.c_str(), &ipaddr) == 1) {
        Config& config = Config::get();
        if(config.isCameraDefined(camId)) {
            CameraInfo camInfo = config.getCameraInfo(camId);
            CameraClass classInfo = config.getCameraClassInfo(camInfo.className);
            std::cout << "Using camera " << camId << " (" << camInfo.description << ")" << std::endl;
            std::string url("http://");
            url += classInfo.username + ":" + classInfo.password + "@";
            url += arg + "/" + classInfo.path;
            return url;
        } else {
            std::cout << "Using camera at address " << arg << std::endl;
            return std::string("http://admin:admin@") + arg + std::string("/video.cgi?.mjpg");
        }
    } else {
        std::cout << "Using file/URL " << arg << std::endl;
        return arg;
    }
}

std::string getURL(int camId) {
    CameraInfo camInfo = Config::get().getCameraInfo(camId);
    return parseURL(camId, camInfo.ip);
}
void detectionFilter(std::vector<DetectedObject>& detections)
{
        for(auto object = detections.begin(); object != detections.end();)
        {
            DetectedObject o = (*object);
            const char* class_name = yolo.getNames()[o.object_class];
            
            if( strcmp(class_name,"person") != 0)
            {
                //std::cout<<class_name<<std::endl;
                detections.erase(object);
                
            }
            else
            {
                //std::cout<<class_name<<std::endl;
                ++object;
            }
         
        }
}
int main(int argc, char** argv)
{
    std::string serverURL = Config::get().getServerURL();
    int serverPort = Config::get().getServerPort();

    int id;
    std::string url;
    int initTime=0;
    std::string pathres = "/home/mw4vision/Desktop/sachin/Evaluation/EvaluationDetectionPerformance/Results/DT/Yolo/";
  /*  if(argc < 2){
        fprintf(stderr, "usage: %s <videofile>\n", argv[0]);
        return 0;
    }*/
// Read arguments and set parameters
    if(argc == 1) {
        // Defaults
        std::cout << "Using default video input" << std::endl;
        id = 30;
        url = "../sandbox4Video.mov";
    } else if((argc == 2 && std::string(argv[1]) == "--help") || argc > 5) {
        printUsage(argv[0]);
        return 0;
    } else if(argc < 5) {
        /*id = std::atoi(argv[1]);
        url = getURL(id);*/
        printUsage(argv[0]);
    } else {
        id = std::atoi(argv[1]);
        url = argv[2];  //  parseURL(id, argv[2]); //  
        initTime = 15*std::atoi(argv[3]);
    }

    // Initialize system objects
    Camera camera(id, url);
    std::cout<<id<<std::endl;
    const Spawns& spawns = camera.getSpawns();
    DifferenceTracker* diffTracker = new DifferenceTracker();
    ObjectTracker* tracker = new ObjectTracker();
    blobSender sender(serverURL.c_str(), serverPort); // set up networking
    Display display(id);
    
    Collection* collect = new Collection(std::string(pathres+ "DT_Yolo")+ std::string(argv[4]));
    

    std::ofstream file("out.csv");
    int frame_no = 0;
    char delimiter = ',';

    //Yolo yolo;
    yolo.setConfigFilePath("cfg/yolo.cfg");
    yolo.setDataFilePath("cfg/coco.data");
    yolo.setWeightFilePath("data/yolo.weights");
    yolo.setAlphabetPath("data/labels/");
    yolo.setNameListFile("data/coco.names");

    bool paused = false;
 // Start video processing
    try {
        cv::Mat img;
        bool flagBlob = false;
        
        while(true) {
            if(!paused) {
                if(!camera.getFrame(img)) {
                    break;
                }
                
                std::vector<DetectedObject> detection;
                std::vector<Contour> contours;
                // Yolo detector  Filtered output
                yolo.detect(img, detection,flagBlob);
                detectionFilter(detection);

                diffTracker->processFrame(img,spawns,contours);
                 
                if(detection.size()==0)
                {
                    tracker->processContoursDiff(tracker->tracks,contours,spawns);   
                }
                else 
                    tracker->processContours(tracker->tracks,detection, contours, spawns);
                sendTracks(id, {img.cols, img.rows}, tracker, sender);
                collect->writeData(id, frame_no , tracker);
                // collect->getFramerateData(frame_no);
            }
            if(frame_no==initTime)
                tracker->startupdate(tracker->tracks);
            display.showFrame(img, tracker->getTracks(),spawns, paused);
            frame_no++;     
            // Only the least-signficant byte is used, sometimes the rest is garbage so 0xFF is needed
            int key = cv::waitKey(targetSleep) & 0xFF;
            if(key == 27) { // Escape pressed
                break;
            } else if(key == ' ') {
                paused = !paused;
            }
        }
    } catch(const std::exception& ex) {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
        return 1;
    }
    return 0;

}
