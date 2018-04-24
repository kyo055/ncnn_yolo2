#include "mtcnn.h"
#include "PNetV2.h"
#include <opencv2/opencv.hpp>

using namespace cv;


void test_video() {
	char *model_path = "../models";
	MTCNN mtcnn(model_path);
	mtcnn.SetMinFace(40);
	// 初始化PNetV2
	char* pnetV2_param = "../PNet_model/PNet_model.param";
	char* pnetV2_bin = "../PNet_model/PNet_model.bin";
	PNetV2 pNetV2(pnetV2_param, pnetV2_bin);

	cv::VideoCapture mVideoCapture(0);
	if (!mVideoCapture.isOpened()) {
		return;
	}
	cv::Mat frame;
	mVideoCapture >> frame;
	while (!frame.empty()) {
		mVideoCapture >> frame;
		if (frame.empty()) {
			break;
		}

		clock_t start_time = clock();

		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
		std::vector<Bbox> finalBbox;
		mtcnn.detect(ncnn_img, finalBbox);
		const int num_box = finalBbox.size();
		std::vector<cv::Rect> bbox;
		bbox.resize(num_box);
		for (int i = 0; i < num_box; i++) {
			bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);
		}
		// 获取人脸区域的bbox手机框
		std::vector<Bbox> phone_Bbox;
		pNetV2.detect(ncnn_img, finalBbox, phone_Bbox);
		std::vector<cv::Rect> phone_bbox;
		phone_bbox.resize(phone_Bbox.size());
		for (int i = 0; i < phone_Bbox.size(); i++) {
			phone_bbox[i] = cv::Rect(phone_Bbox[i].x1, phone_Bbox[i].y1, phone_Bbox[i].x2 - phone_Bbox[i].x1 + 1, phone_Bbox[i].y2 - phone_Bbox[i].y1 + 1);
		}
		for (vector<cv::Rect>::iterator it = phone_bbox.begin(); it != phone_bbox.end(); it++) {
			rectangle(frame, (*it), Scalar(0, 255, 0), 2, 8, 0);
		}

		for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
			rectangle(frame, (*it), Scalar(0, 0, 255), 2, 8, 0);
		}
		imshow("face_detection", frame);
		clock_t finish_time = clock();
		double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
		std::cout << "time" << total_time * 1000 << "ms" << std::endl;

		int q = cv::waitKey(10);
		if (q == 27) {
			break;
		}
	}
	system("pause");
	return;
}

int test_picture() {
	char *model_path = "../models";
	MTCNN mtcnn(model_path);

	clock_t start_time = clock();

	cv::Mat image;
	image = cv::imread("../sample.jpg");
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	std::vector<Bbox> finalBbox;
	mtcnn.detect(ncnn_img, finalBbox);

	const int num_box = finalBbox.size();
	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

	}
	for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
		rectangle(image, (*it), Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("face_detection", image);
	clock_t finish_time = clock();
	double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
	std::cout << "time" << total_time * 1000 << "ms" << std::endl;

	cv::waitKey(0);

}

int main(int argc, char** argv) {

	test_video();
	//test_picture();
	return 0;
}