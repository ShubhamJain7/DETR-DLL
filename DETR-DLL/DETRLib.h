#pragma once
#include "framework.h"
#include "internal.h"

#ifdef DETRDLL_EXPORTS
#define DETRLIBRARY_API __declspec(dllexport)
#else
#define DETRLIBRARY_API __declspec(dllimport)
#endif

extern "C" DETRLIBRARY_API int doDetection(IN wchar_t* model_path, IN char* image_path);
extern "C" DETRLIBRARY_API int getDetections(OUT Detection * results_list, size_t size);