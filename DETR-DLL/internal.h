#pragma once
#include<vector>

struct Detection {
    int classId;
    float probability;
    int x1;
    int y1;
    int x2;
    int y2;
};

// declare constants
const float conf_threshold = 0.75;

std::vector<Detection> _getDetections(wchar_t* model_path, char* image_path);