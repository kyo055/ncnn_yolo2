#pragma once
#ifndef __PNetV2_NCNN_H__
#define __PNetV2_NCNN_H__
#include "net.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>
#include "mtcnn.h"
using namespace std;
class PNetV2
{
public:
	PNetV2(const std::string param_files, const std::string bin_files);
	void detect(ncnn::Mat& img_, std::vector<Bbox>& faceBox, std::vector<Bbox>& regBox);
	~PNetV2();

private:
	ncnn::Mat img;
	void nms(vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname = "Union");
	Bbox extend_face_box(cv::Size2i& imgShape,Bbox& max_faceBox, float extendW_, float extendH_);
	ncnn::Net PnetV2;
	float m_extendW = 0.2;
	float m_extendH = 0.4;
	void refine(vector<Bbox> &vecBbox, const int &height, const int &width, bool square);
	const float score_threshold[1] = { 0.8f };
	const float nms_threshold[1] = { 0.8f };
};
#endif //__PNetV2_NCNN_H__
