// DETRConsole.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <DETR-DLL/DETRLib.h>

int main()
{
    char img_path[] = "C:/Users/dell/source/repos/DETR-DLL/test.jpg";
    wchar_t model_path[] = L"C:/Users/dell/source/repos/DETR-DLL/models/DETRmodel.onnx";
    int result = doDetection(model_path, img_path);

    Detection objects[20];
    int size = getDetections(objects, 20);
    for (int i = 0; i < size; i++)
    {
        Detection d = objects[i];
        std::cout << d.classId << "(" << d.probability << "):";
        std::cout << "[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "]" << std::endl;
    }
}
