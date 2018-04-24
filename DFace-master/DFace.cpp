#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>

// ncnn include 
#include <mat.h>
#include <net.h>

#include <stdio.h>
#include <math.h>
#include <iostream>


int main()
{
	const char* imagepath = "1.jpg";

	cv::Mat cv_img = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
	if (cv_img.empty())
	{
		fprintf(stderr, "cv::imread %s failed\n", imagepath);
		return -1;
	}
	cv::resize(cv_img, cv_img, cv::Size(24, 24));


	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR, cv_img.cols, cv_img.rows);
	const float mean_vals[3] = { .0f, .0f,  .0f };
	const float norm_vals[3] = { 1.0 / 255.0,1.0 / 255.0, 1.0 / 255.0 };
	ncnn_img.substract_mean_normalize(mean_vals, norm_vals);

	ncnn::Mat ncnn_img1 = ncnn_img.clone();

	// feature model
	std::string  model_path = "PNet_model";
	ncnn::Net PNetV2;
	std::string model_param = model_path + "/PNet_model.param";
	std::string model_bin = model_path + "/PNet_model.bin";
	PNetV2.load_param(model_param.c_str());
	PNetV2.load_model(model_bin.c_str());

	ncnn::Extractor fea_ex = PNetV2.create_extractor();
	fea_ex.set_light_mode(true);

	double time_start0 = (double)cv::getTickCount();
	fea_ex.input("data", ncnn_img);
	ncnn::Mat label;
	ncnn::Mat reg;
	int a = fea_ex.extract("Sigmoid_1", label);
	printf("the label, channel:%d\t width:%d\t height:%d\t\n", label.c, label.w, label.h);
	for (size_t i = 0; i < label.c; i++)
	{
		ncnn::Mat c0 = label.channel(i);
		std::cout << "[";
		for (size_t st = 0; st < c0.h; st++)
		{
			std::cout << "[ ";
			float* data = c0.row(st);
			for (size_t sw = 0; sw < c0.w; sw++)
			{
				std::cout << *data++ << " ";
			}
			std::cout << " ]" << std::endl;
		}
		std::cout << "]" << std::endl;
	}

	std::cout << "extract result is" << a << std::endl;
	int b = fea_ex.extract("ConvNd_6", reg);
	printf("the reg, channel:%d\t width:%d\t height:%d\t\n", reg.c, reg.w, reg.h);

	for (size_t i = 0; i < reg.c; i++)
	{
		ncnn::Mat c0 = reg.channel(i);
		std::cout << "[";
		for (size_t st = 0; st < c0.h; st++)
		{
			std::cout << "[ ";
			float* data = c0.row(st);
			for (size_t sw = 0; sw < c0.w; sw++)
			{
				std::cout << *data++ << " ";
			}
			std::cout << " ]" << std::endl;
		}
		std::cout << "]" << std::endl;
	}
	double time_end0 = (double)cv::getTickCount();
	double t0 = (time_end0 - time_start0) * 1000 / cv::getTickFrequency();
	std::cout << "ncnn 提取模型首次时间为：" << t0 << "ms" << std::endl;


	// 第二次提取

	system("pause");
}
