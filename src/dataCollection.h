#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/core.hpp>
#include "camera.h"
#include "objecttracker.h"
#include "blob.h"
#include <iostream>
#include <fstream>
#include "track.h"
#include "config.h"
/*
struct CameraClass {
    std::string name;
    std::string username;
    std::string password;
    std::string path;
};

struct CameraInfo {
    int id;
    std::string name;
    std::string description;
    std::string className;
    std::string ip;
	int angle;
    cv::Rect crop;
    Spawns spawns;
};
*/

class Collection {
public:
    //static Collection& get();

    ~Collection();
    void writeData(int id,int frame_no,ObjectTracker* tracker);
    void getFramerateData(int frame_no);
    Collection(std::string filename);
private:
    static Collection* instance;

    std::ofstream file;
    std::ofstream fpsfile;
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
    FPS fps;
    // std::unordered_map<int, CameraInfo> cameras;

    
};
