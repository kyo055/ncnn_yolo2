#include "PNetV2.h"


extern bool cmpScore(Bbox lsh, Bbox rsh);

PNetV2::PNetV2(const std::string param_files, const std::string bin_files)
{
	PnetV2.load_param(param_files.data());
	PnetV2.load_model(bin_files.data());
}

void PNetV2::detect(ncnn::Mat & img_, std::vector<Bbox>& faceBox, std::vector<Bbox>& regBox)
{
	img = img_;
	int maxIndex = -1;
	float maxArea = 0.0;
	int count = 0;
	for (std::vector<Bbox>::iterator it = faceBox.begin(); it != faceBox.end(); it++, count++)
	{
		if (it->area > maxArea)
		{
			maxArea = it->area;
			maxIndex = count;
		}
	}
	if (maxIndex < 0)
	{
		return;
	}

	// 获取最大人脸
	Bbox maxBox = faceBox[maxIndex];

	cv::Size2i imgShape = cv::Size2i(img.w, img.h);
	// 扩展人脸10次
	m_extendW = 0.2;
	m_extendH = 0.4;
	std::vector<Bbox> extend_faceBoxes;
	for (int i = 0; i < 10; i++)
	{
		extend_faceBoxes.push_back(extend_face_box(imgShape, maxBox, m_extendW, m_extendH));
		m_extendW += 0.03;
		m_extendH += 0.04;
	}
	// 对扩展的人脸区域进行refine,1.进行square，给出square区域
	refine(extend_faceBoxes, imgShape.width, imgShape.height, true);
	// 对refine后的square进行 PNetV2的特征提取
	regBox.clear();
	for (vector<Bbox>::iterator it = extend_faceBoxes.begin(); it != extend_faceBoxes.end(); it++) {
		ncnn::Mat tempIm;
		copy_cut_border(img, tempIm, (*it).y1, imgShape.height - (*it).y2, (*it).x1, imgShape.width - (*it).x2);
		ncnn::Mat in;
		resize_bilinear(tempIm, in, 24, 24);
		ncnn::Extractor ex = PnetV2.create_extractor();
		ex.set_light_mode(true);
		ex.input("data", in);
		ncnn::Mat label, regx;
		ex.extract("Sigmoid_1", label);
		ex.extract("ConvNd_6-2", regx);
		
		// 如果当前的 label得分大于阈值，那么就保留这个reg和其对应的boundingbox 否则就不保留
		Bbox new_bbox;
		float *p = label.channel(1);//score.data + score.cstep
		if (*p> score_threshold[0]) {
			new_bbox = *it;
			for (int channel = 0; channel<4; channel++) {
				new_bbox.regreCoord[channel] = regx.channel(channel)[0];
			}
			regBox.push_back(new_bbox);
		}
	}
	// nms
	nms(regBox, nms_threshold[0], "Union");
	
	// 更新当前的regBox
	for (vector<Bbox>::iterator it = regBox.begin(); it != regBox.end(); it++) {
		int bw = it->x2 - it->x1;
		int bh = it->y2 - it->y1;

		int align_topx = it->x1 + it->regreCoord[0] * bw;
		int align_topy = it->y1 + it->regreCoord[1] * bh;
		int align_bottomx = it->x2 + it->regreCoord[2] * bw;
		int align_bottomy = it->y2 + it->regreCoord[3] * bh;

		// 更新bbox
		it->x1 = align_topx;
		it->x2 = align_bottomx;
		it->y1 = align_topy;
		it->y2 = align_bottomy;
	}
}



PNetV2::~PNetV2()
{
}

Bbox PNetV2::extend_face_box(cv::Size2i& imgShape, Bbox & max_faceBox, float extendW_, float extendH_)
{
	int deltaX = max_faceBox.x2 - max_faceBox.x1;
	int deltaY = max_faceBox.y2 - max_faceBox.y1;
	
	int new_x1 = std::max(0, int(max_faceBox.x1 - deltaX * extendW_));
	int new_y1 = std::max(0, int(max_faceBox.y1 - deltaX * extendH_));
	int new_x2 = std::min(int(new_x1 + (1 + 2.0 * extendW_) * deltaX), imgShape.width - 1);
	int new_y2 = std::min(int(new_y1 + (1 + 2.0 * extendH_) * deltaY), imgShape.height - 1);

	Bbox extend_face;
	extend_face.x1 = new_x1;
	extend_face.x2 = new_x2;
	extend_face.y1 = new_y1;
	extend_face.y2 = new_y2;
	extend_face.area = (new_y2 - new_y1) * (new_x2 - new_x1);
	return extend_face;
}

void PNetV2::refine(vector<Bbox>& vecBbox, const int & height, const int & width, bool square)
{
	if (vecBbox.empty()) {
		cout << "Bbox is empty!!" << endl;
		return;
	}
	float bbw = 0, bbh = 0, maxSide = 0;
	float h = 0, w = 0;
	float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	for (vector<Bbox>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++) {
		bbw = (*it).x2 - (*it).x1 + 1;
		bbh = (*it).y2 - (*it).y1 + 1;
		x1 = (*it).x1 + (*it).regreCoord[0] * bbw;
		y1 = (*it).y1 + (*it).regreCoord[1] * bbh;
		x2 = (*it).x2 + (*it).regreCoord[2] * bbw;
		y2 = (*it).y2 + (*it).regreCoord[3] * bbh;

		if (square) {
			w = x2 - x1 + 1;
			h = y2 - y1 + 1;
			maxSide = (h>w) ? h : w;
			x1 = x1 + w*0.5 - maxSide*0.5;
			y1 = y1 + h*0.5 - maxSide*0.5;
			(*it).x2 = round(x1 + maxSide - 1);
			(*it).y2 = round(y1 + maxSide - 1);
			(*it).x1 = round(x1);
			(*it).y1 = round(y1);
		}

		//boundary check
		if ((*it).x1<0)(*it).x1 = 0;
		if ((*it).y1<0)(*it).y1 = 0;
		if ((*it).x2>width)(*it).x2 = width - 1;
		if ((*it).y2>height)(*it).y2 = height - 1;

		it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
	}
}

void PNetV2::nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname) {
	if (boundingBox_.empty()) {
		return;
	}
	sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	std::vector<int> vPick;
	int nPick = 0;
	std::multimap<float, int> vScores;
	const int num_boxes = boundingBox_.size();
	vPick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i) {
		vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
	}
	while (vScores.size() > 0) {
		int last = vScores.rbegin()->second;
		vPick[nPick] = last;
		nPick += 1;
		for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
			int it_idx = it->second;
			maxX = std::max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
			maxY = std::max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
			minX = std::min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
			minY = std::min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
			//maxX1 and maxY1 reuse 
			maxX = ((minX - maxX + 1)>0) ? (minX - maxX + 1) : 0;
			maxY = ((minY - maxY + 1)>0) ? (minY - maxY + 1) : 0;
			//IOU reuse for the area of two bbox
			IOU = maxX * maxY;
			if (!modelname.compare("Union"))
				IOU = IOU / (boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);
			else if (!modelname.compare("Min")) {
				IOU = IOU / ((boundingBox_.at(it_idx).area < boundingBox_.at(last).area) ? boundingBox_.at(it_idx).area : boundingBox_.at(last).area);
			}
			if (IOU > overlap_threshold) {
				it = vScores.erase(it);
			}
			else {
				it++;
			}
		}
	}

	vPick.resize(nPick);
	std::vector<Bbox> tmp_;
	tmp_.resize(nPick);
	for (int i = 0; i < nPick; i++) {
		tmp_[i] = boundingBox_[vPick[i]];
	}
	boundingBox_ = tmp_;
}