#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>

// ncnn include 
#include <mat.h>
#include <net.h>
#include <numeric> 
#include <algorithm>
#include <functional>
#include <stdio.h>
#include <math.h>
#include <iostream>

struct cenbox
{
	float bcx=0.0;
	float bcy=0.0;
	float bcw=0.0;
	float bch=0.0;
};

struct ObjectBB {
	cenbox brec;
	int class_id;
	float iou_prob;
	float id_prob;
};


static int load_yolomodel(ncnn::Net &model)
{
	model.load_param("mobilenet_ssd_voc_ncnn.param");
	model.load_model("mobilenet_ssd_voc_ncnn.bin");

	ncnn::Extractor ex = model.create_extractor();
	ex.set_light_mode(true);
	//ex.set_num_threads(4);
}

void binary_multiplay(ncnn::Mat& a, ncnn::Mat& b, ncnn::Mat& out)
{
	ncnn::Layer* op = ncnn::create_layer("BinaryOp");
	ncnn::ParamDict pd;
	pd.set(0, 2);    // mul
	op->load_param(pd);

	// forward
	std::vector<ncnn::Mat> bottoms(2);
	bottoms[0] = a;
	bottoms[1] = b;

	std::vector<ncnn::Mat> tops(1);
	op->forward(bottoms, tops);

	out = tops[0];

	delete op;
}

void binary_sigmoid(ncnn::Mat& in_place)
{
	ncnn::Layer* op = ncnn::create_layer("Sigmoid");
	op->forward_inplace(in_place);

	delete op;
}

void binary_add(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& c)
{
	ncnn::Layer* op = ncnn::create_layer("BinaryOp");

	// set param
	ncnn::ParamDict pd;
	pd.set(0, 0);// op_type

	op->load_param(pd);

	// forward
	std::vector<ncnn::Mat> bottoms(2);
	bottoms[0] = a;
	bottoms[1] = b;

	std::vector<ncnn::Mat> tops(1);
	op->forward(bottoms, tops);

	c = tops[0];

	delete op;
}


void binary_exp(ncnn::Mat& in_place)
{
	ncnn::Layer* op = ncnn::create_layer("Exp");
	op->forward_inplace(in_place);
	delete op;
}

void mat_exp(ncnn::Mat& in_place)
{
	float* data_ptr = in_place;
	for (size_t st = 0; st < in_place.w; st++, data_ptr++)
	{
		float origin = *data_ptr;
		float score = expf(origin);
		*data_ptr = score;
	}
	
}


void binary_softmax(ncnn::Mat& in_place)
{
	ncnn::Layer* op = ncnn::create_layer("Softmax");
	ncnn::ParamDict pd;
	pd.set(0, 1);

	op->load_param(pd);
	op->forward_inplace(in_place);
	delete op;
}

void bianry_max(ncnn::Mat& input, ncnn::Mat& cls_max_confs, ncnn::Mat& cls_max_ids)
{
	int width = input.w;
	int height = input.h;

	float* cls_conf = static_cast<float*>(cls_max_confs.data);
	float* cls_max = static_cast<float*> (cls_max_ids.data);
	for (int i = 0; i<height; i++)
	{
		float* data_ptr = input.row(i);

		float max_conf = 0.0;
		float max_id = -1;
		for (int j = 0; j<width; j++,data_ptr++)
		{
			if (*(data_ptr) >max_conf)
			{
				max_conf =* data_ptr;
				max_id = j;
			}
		}
		*cls_conf++ = max_conf;	// copy and move pointer
		*cls_max++ = static_cast<float>(max_id);
	}
}

bool compare(int a, int b, float* data)
{
	return data[a]<data[b];
}

void sort_indexes(std::vector<float>& det_conf, std::vector<int> &result) 
{
	std::iota(result.begin(),result.end(), 0); // fill index with {0,1,2,...} This only needs to happen once
	std::sort(result.begin(), result.end(), std::bind(compare, std::placeholders::_1, std::placeholders::_2, det_conf.data()));
}

float bbox_iou(ObjectBB bbox1, ObjectBB bbox2, bool flag = false)
{
	if (flag)
	{

	}
	else
	{
		float mx = std::max(bbox1.brec.bcx - bbox1.brec.bcw / 2.0, bbox2.brec.bcx - bbox2.brec.bcw / 2.0);
		float my = std::max(bbox1.brec.bcy - bbox1.brec.bch / 2.0, bbox2.brec.bcy - bbox2.brec.bch / 2.0);
		float Mx = std::max(bbox1.brec.bcx + bbox1.brec.bcw / 2.0, bbox2.brec.bcx + bbox2.brec.bcw / 2.0);
		float My = std::max(bbox1.brec.bcy + bbox1.brec.bch / 2.0, bbox2.brec.bcy + bbox2.brec.bch / 2.0);

		float w1 = bbox1.brec.bcw;
		float h1 = bbox1.brec.bch;
		float w2 = bbox2.brec.bcw;
		float h2 = bbox2.brec.bch;
		
		float uw = Mx - mx;
		float uh = My - my;
		float cw = w1 + w2 - uw;
		float ch = h1 + h2 - uh;

		if (cw <= 0 || ch <= 0)
			return 0.0;
		float area1 = w1 * h1;
		float area2 = w2 * h2;
		float carea = cw * ch;
		float uarea = area1 + area2 - carea;
		float iou = carea / uarea;
		return iou;
	}
}

void nms(std::vector<ObjectBB>& bboxes , float nms_thresh,std::vector<ObjectBB>& nms_bbox)
{
	nms_bbox.empty();
	if (bboxes.size() == 0)
		return;
	std::vector<float> def_confs(bboxes.size());
	for (size_t st = 0; st < bboxes.size(); st++)
	{
		def_confs[st] = 1 - bboxes[st].iou_prob;
	}
	std::vector<int> sort_ids(bboxes.size());
	sort_indexes(def_confs, sort_ids);
	for (size_t st = 0; st < bboxes.size(); st++)
	{
		//std::cout << "sort :" << st << " index:" << sort_ids[st] << std::endl;
		ObjectBB bbox = bboxes[sort_ids[st]];
		if (bbox.iou_prob > 0)
		{
			nms_bbox.push_back(bbox);
			for (size_t sz = st + 1; sz < bboxes.size(); sz++)
			{
				ObjectBB bbox_nt = bboxes[sort_ids[sz]];
				if (bbox_iou(bbox, bbox_nt, false) > nms_thresh)
				{
					bboxes[sort_ids[sz]].iou_prob = 0.0;
				}
			}
		}
	}
}

ncnn::Mat tronspose01(ncnn::Mat& input_data)
{
	size_t h = input_data.h;	// 1 轴
	size_t c = input_data.c;	// 0 轴
	size_t w = input_data.w;	// 2 轴
	//int width_step = c * sizeof(input_data.elemsize)*h;
	float* out_ptr = new float[h*c*w]();
	float* in_ptr = static_cast<float*>(input_data.data);
	float* loc_ptr = in_ptr;
	float* out_rfptr = out_ptr;
	for (size_t sh = 0; sh < h; sh++)
	{
		for (size_t sc = 0; sc < c; sc++)
		{
			ncnn::Mat mat_c = input_data.channel(sc);
			float* r_data = mat_c.row(sh);
			for (size_t sw = 0; sw < w; sw++)
			{
				*out_rfptr++ = *r_data++;
			}
		}
	}
	ncnn::Mat out_data(w*c*h, out_ptr,4);
	return out_data.reshape(w, c, h);
}


std::vector<ObjectBB> forward(ncnn::Net &model, ncnn::Mat input)
{
	ncnn::Mat out;
	ncnn::Extractor ex = model.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(4);
	ex.input("data", input);
	int a = ex.extract("ConvNd_9", out);; // [channel,height,width] ----> out Mat shape [125,13,13]

									  /// get the Object Bounding box from the output
									  // output ---->c=125,w=13,h=13
									  //# output: ----> [5,5+nC,h*w] : ------> [5+nc,5,h*w] ----> [5+nC, nA*nh*nw] == [25, nB*5*13*13]
	//ncnn::Mat c0 = out.channel(0);

	/*std::cout << "[";
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
	std::cout << "]" << std::endl;*/

	float conf_thresh = 0.5;
	float nms_thresh = 0.4;
	int num_classes = 20;
	int num_anchors = 5;
	float anchors[10] = { 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52 };
	int h = out.h;
	int c = out.c;
	int w = out.w;
	ncnn::Mat out1 = out.reshape(h*w, 5 + num_classes, num_anchors); // [ w, h ,c]
	out = out.reshape(h*w, 5 + num_classes,num_anchors);
	// 将ncnn::Mat 转换为 cv::array
	out = tronspose01(out);



	out = out.reshape(h*w*num_anchors, 5 + num_classes); //[13*13*5,5+20]
	// 显示部分out

	const int grid_step = num_anchors*h*w;

	// grid_x
	float* grid_x = new float[num_anchors*h*w]();
	for (int i = 0; i < num_anchors; i++)
	{
		for (size_t j = 0; j < h; j++)
		{
			for (size_t sz = 0; sz<w; sz++)
			{
				grid_x[w*h*i+j*w + sz] = (float)sz;
			}
		}
	}
	// grid_y
	float* grid_y = new float[grid_step]();
	for (int i = 0; i < num_anchors; i++)
	{
		for (int j = 0; j < h; j++)
		{
			for (size_t sz = 0; sz<w; sz++)
			{
				grid_y[i*h*w + j*w + sz] = j;
			}
		}
	}
	//std::cout << "finish make the grid " << std::endl;
	ncnn::Mat grid_xM = ncnn::Mat(h*w*num_anchors, grid_x);
	ncnn::Mat grid_yM = ncnn::Mat(h*w*num_anchors, grid_y);
	float *out_0 = out.row(0);	//x 特征
	float *out_1 = out.row(1);	// y 特征
	float *out_2 = out.row(2);	// w 特征
	float *out_3 = out.row(3);	// h 特征
	float *out_4 = out.row(4);	// bbox_conf 特征

	ncnn::Mat x_score = ncnn::Mat(h*w*num_anchors, out_0);


	ncnn::Mat y_score = ncnn::Mat(h*w*num_anchors, out_1);

	// sigmod
	binary_sigmoid(x_score);		

	binary_sigmoid(y_score);

	binary_add(x_score, grid_xM, x_score);		// 求解 sigmod(x_feature)+grid_x
	binary_add(y_score, grid_xM, y_score);		// sigmod(y_feature) + grid_y


	// define the anchors mesh grid
	float* anchor_gridw = new float[h*w*num_anchors]();
	float* anchor_gridh = new float[h*w*num_anchors]();
	for (int i = 0; i <num_anchors; i++)
	{
		for (int j = 0; j<h*w; j++)
		{
			anchor_gridw[i*h*w + j] = anchors[i * 2];
			anchor_gridh[i*h*w + j] = anchors[i * 2 + 1];
		}
	}
	ncnn::Mat anchor_w = ncnn::Mat(h*w*num_anchors, anchor_gridw);
	ncnn::Mat anchor_h = ncnn::Mat(h*w*num_anchors, anchor_gridh);

	ncnn::Mat anchorw_score = ncnn::Mat(h*w*num_anchors, out_2);
	
	ncnn::Mat anchorh_score = ncnn::Mat(h*w*num_anchors, out_3);

	mat_exp(anchorw_score);
	mat_exp(anchorh_score);
	binary_multiplay(anchorw_score, anchor_w, anchorw_score);	// ws = torch.exp(output[2]) * anchor_w
	binary_multiplay(anchorh_score, anchor_h, anchorh_score);	// hs = torch.exp(output[3]) * anchor_h

	ncnn::Mat det_conf = ncnn::Mat(h*w*num_anchors, out_4);
	binary_sigmoid(det_conf);
	ncnn::Mat def_conf_flat = det_conf.reshape(det_conf.w * det_conf.h * det_conf.c);	// det_confs = torch.sigmoid(output[4])

	float* out_5 = out.row(5);
	ncnn::Mat cls = ncnn::Mat(h*w*num_anchors, num_classes, out_5);
	cv::Mat clsCV_mat( num_classes, h*w*num_anchors, CV_32F,cls.data);//进行转置操作
	cv::Mat cls_matT= clsCV_mat.t();
	//std::cout<<"w:" << cls_matT.cols<<" \t h: " << cls_matT.rows << std::endl;		// w:20  h:845
	ncnn::Mat cls_mat = ncnn::Mat(num_classes, h*w*num_anchors, cls_matT.data,sizeof(float));
	//std::cout << "the cls_mat_transpose mat is  width " << cls_mat.w << " height is " << cls_mat.h << std::endl;	//w:20, h:845
	
	binary_softmax(cls_mat); // softmax [nA*nH*nW,nC],  cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data

	ncnn::Mat cls_conf_mat(h*w*num_anchors);
	ncnn::Mat cls_id_mat(h*w*num_anchors);

	bianry_max(cls_mat, cls_conf_mat, cls_id_mat);	//cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
	//cv::minMaxIdx(visitor_scores, &visitor_minVal, &visitor_maxVal, &visitor_minLoc, &visitor_maxLoc);
	float* cls_conf_data = cls_conf_mat.row(0);
	float* cls_id_data = cls_id_mat.row(0);

	int sz_hw = h*w;

	std::vector<ObjectBB> boxes;
	for (int i = 0; i<h; i++)
	{
		for (int j = 0; j<w; j++)
		{
			for (int a = 0; a<num_anchors; a++)
			{
				int ind = a*sz_hw + i*w + j;
				float cls_conf = cls_conf_mat[ind];
				float pro_conf = def_conf_flat[ind];
				float conf = pro_conf * cls_conf;

				ObjectBB bbx;

				if (conf > conf_thresh)
				{
					float bcx = x_score[ind];
					float bcy = y_score[ind];
					float bcw = anchorw_score[ind];
					float bch = anchorh_score[ind];

					bbx.brec.bcx = bcx/(w*1.0);
					bbx.brec.bcy = bcy/(h*1.0);
					bbx.brec.bcw = bcw/(w*1.0);
					bbx.brec.bch = bch/(h*1.0);
					bbx.class_id = (int)cls_id_mat[ind];
					bbx.id_prob = cls_conf;
					bbx.iou_prob = pro_conf;
					boxes.push_back(bbx);
				}
			}
		}
	}
	delete[] grid_x;
	delete[] grid_y;
	delete[] anchor_gridw;
	delete[] anchor_gridh;
	return boxes;
}

std::vector<ObjectBB> do_detect(ncnn::Mat &out)
{
	float conf_thresh = 0.5;
	float nms_thresh = 0.4;
	int num_classes = 20;
	int num_anchors = 5;
	float anchors[10] = { 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52 };
	int h = out.h;
	int c = out.c;
	int w = out.w;
	ncnn::Mat out1 = out.reshape(h*w, 5 + num_classes, num_anchors); // [ w, h ,c]
	out = out.reshape(h*w, 5 + num_classes, num_anchors);
	// 将ncnn::Mat 转换为 cv::array
	out = tronspose01(out);



	out = out.reshape(h*w*num_anchors, 5 + num_classes); //[13*13*5,5+20]
														 // 显示部分out

	const int grid_step = num_anchors*h*w;

	// grid_x
	float* grid_x = new float[num_anchors*h*w]();
	for (int i = 0; i < num_anchors; i++)
	{
		for (size_t j = 0; j < h; j++)
		{
			for (size_t sz = 0; sz<w; sz++)
			{
				grid_x[w*h*i + j*w + sz] = (float)sz;
			}
		}
	}
	// grid_y
	float* grid_y = new float[grid_step]();
	for (int i = 0; i < num_anchors; i++)
	{
		for (int j = 0; j < h; j++)
		{
			for (size_t sz = 0; sz<w; sz++)
			{
				grid_y[i*h*w + j*w + sz] = j;
			}
		}
	}
	std::cout << "finish make the grid " << std::endl;
	ncnn::Mat grid_xM = ncnn::Mat(h*w*num_anchors, grid_x);
	ncnn::Mat grid_yM = ncnn::Mat(h*w*num_anchors, grid_y);
	float *out_0 = out.row(0);	//x 特征
	float *out_1 = out.row(1);	// y 特征
	float *out_2 = out.row(2);	// w 特征
	float *out_3 = out.row(3);	// h 特征
	float *out_4 = out.row(4);	// bbox_conf 特征

	ncnn::Mat x_score = ncnn::Mat(h*w*num_anchors, out_0);


	ncnn::Mat y_score = ncnn::Mat(h*w*num_anchors, out_1);

	// sigmod
	binary_sigmoid(x_score);

	binary_sigmoid(y_score);

	binary_add(x_score, grid_xM, x_score);		// 求解 sigmod(x_feature)+grid_x
	binary_add(y_score, grid_xM, y_score);		// sigmod(y_feature) + grid_y


												// define the anchors mesh grid
	float* anchor_gridw = new float[h*w*num_anchors]();
	float* anchor_gridh = new float[h*w*num_anchors]();
	for (int i = 0; i <num_anchors; i++)
	{
		for (int j = 0; j<h*w; j++)
		{
			anchor_gridw[i*h*w + j] = anchors[i * 2];
			anchor_gridh[i*h*w + j] = anchors[i * 2 + 1];
		}
	}
	ncnn::Mat anchor_w = ncnn::Mat(h*w*num_anchors, anchor_gridw);
	ncnn::Mat anchor_h = ncnn::Mat(h*w*num_anchors, anchor_gridh);

	ncnn::Mat anchorw_score = ncnn::Mat(h*w*num_anchors, out_2);

	ncnn::Mat anchorh_score = ncnn::Mat(h*w*num_anchors, out_3);

	mat_exp(anchorw_score);
	mat_exp(anchorh_score);
	binary_multiplay(anchorw_score, anchor_w, anchorw_score);	// ws = torch.exp(output[2]) * anchor_w
	binary_multiplay(anchorh_score, anchor_h, anchorh_score);	// hs = torch.exp(output[3]) * anchor_h

	ncnn::Mat det_conf = ncnn::Mat(h*w*num_anchors, out_4);
	binary_sigmoid(det_conf);
	ncnn::Mat def_conf_flat = det_conf.reshape(det_conf.w * det_conf.h * det_conf.c);	// det_confs = torch.sigmoid(output[4])

	float* out_5 = out.row(5);
	ncnn::Mat cls = ncnn::Mat(h*w*num_anchors, num_classes, out_5);
	cv::Mat clsCV_mat(num_classes, h*w*num_anchors, CV_32F, cls.data);//进行转置操作
	cv::Mat cls_matT = clsCV_mat.t();
	//std::cout<<"w:" << cls_matT.cols<<" \t h: " << cls_matT.rows << std::endl;		// w:20  h:845
	ncnn::Mat cls_mat = ncnn::Mat(num_classes, h*w*num_anchors, cls_matT.data, sizeof(float));
	//std::cout << "the cls_mat_transpose mat is  width " << cls_mat.w << " height is " << cls_mat.h << std::endl;	//w:20, h:845

	binary_softmax(cls_mat); // softmax [nA*nH*nW,nC],  cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data

	ncnn::Mat cls_conf_mat(h*w*num_anchors);
	ncnn::Mat cls_id_mat(h*w*num_anchors);

	bianry_max(cls_mat, cls_conf_mat, cls_id_mat);	//cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
													//cv::minMaxIdx(visitor_scores, &visitor_minVal, &visitor_maxVal, &visitor_minLoc, &visitor_maxLoc);
	float* cls_conf_data = cls_conf_mat.row(0);
	float* cls_id_data = cls_id_mat.row(0);

	int sz_hw = h*w;

	std::vector<ObjectBB> boxes;
	for (int i = 0; i<h; i++)
	{
		for (int j = 0; j<w; j++)
		{
			for (int a = 0; a<num_anchors; a++)
			{
				int ind = a*sz_hw + i*w + j;
				float cls_conf = cls_conf_mat[ind];
				float pro_conf = def_conf_flat[ind];
				float conf = pro_conf * cls_conf;

				ObjectBB bbx;

				if (conf > conf_thresh)
				{
					float bcx = x_score[ind];
					float bcy = y_score[ind];
					float bcw = anchorw_score[ind];
					float bch = anchorh_score[ind];

					bbx.brec.bcx = bcx / (w*1.0);
					bbx.brec.bcy = bcy / (h*1.0);
					bbx.brec.bcw = bcw / (w*1.0);
					bbx.brec.bch = bch / (h*1.0);
					bbx.class_id = (int)cls_id_mat[ind];
					bbx.id_prob = cls_conf;
					bbx.iou_prob = pro_conf;
					boxes.push_back(bbx);
				}
			}
		}
	}
	delete[] grid_x;
	delete[] grid_y;
	delete[] anchor_gridw;
	delete[] anchor_gridh;
	return boxes;
}

void plot_boxes_cv(cv::Mat& img, std::vector<ObjectBB> boxes, std::vector<std::string> class_names)
{
	std::vector<cv::Scalar>colors = { cv::Scalar(50,0,0),cv::Scalar(0,50,0),cv::Scalar(0,0,50),cv::Scalar(50,50,0),
									  cv::Scalar(100,0,0),cv::Scalar(0,100,0),cv::Scalar(0,0,100),cv::Scalar(0,100,100),
									  cv::Scalar(200,0,0),cv::Scalar(0,200,0),cv::Scalar(0,0,200),cv::Scalar(0,200,200),
									  cv::Scalar(255,0,0),cv::Scalar(0,255,0),cv::Scalar(0,0,255),cv::Scalar(255,0,255) ,
									   cv::Scalar(128,128,0),cv::Scalar(0,255,128),cv::Scalar(0,128,128),cv::Scalar(255,128,255) };
	int width = img.cols;
	int height = img.rows;
	for (size_t st = 0; st < boxes.size(); st++)
	{
		ObjectBB box = boxes[st];
		try
		{
			int x1 = int(round((box.brec.bcx - box.brec.bcw / 2.0) * width));
			int y1 = int(round((box.brec.bcy - box.brec.bch / 2.0) * height));
			int x2 = int(round((box.brec.bcx + box.brec.bcw / 2.0) * width));
			if (x2 > width)
				x2 = width;
			int y2 = int(round((box.brec.bcy + box.brec.bch / 2.0) * height));
			if (y2 > height)
				y2 = height;
			int box_width = x2 - x1;
			int box_height = y2 - y1;
			printf("--------->>> w: %d \t h:%d\n", box_width, box_height);
			float cls_conf = box.id_prob;
			int cls_id = (int)box.class_id;
			float iou_conf = box.iou_prob;
			printf("------->>>>> %s: %f\n", class_names[cls_id].c_str(), cls_conf);
			cv::putText(img, class_names[cls_id].c_str(), cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 1.2, colors[cls_id], 1);
			cv::rectangle(img, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), colors[cls_id], 1);
		}
		catch (const std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
		
	}
}

int main(int argc, char** argv)
{
	std::string option = argv[1];

	std::string  model_path = "Darknet_model2";
	const char* imagepath = "2.jpg";

	std::vector<std::string> cls_names = {
		"aeroplane","bicycle","bird","boat","bottle",
		"bus","car","cat","chair","cow",
		"diningtable","dog","horse","motorbike","person",
		"pottedplant","sheep","sofa","train","tvmonitor"
	};

	ncnn::Net darknet;
	std::string model_param = model_path + "/Darknet_model.param";
	std::string model_bin = model_path + "/Darknet_model.bin";
	darknet.load_param(model_param.c_str());
	darknet.load_model(model_bin.c_str());

	if (option == "v")
	{
		cv::VideoCapture cap(0);
		cv::Mat frame;
		while (1)
		{
			cap >> frame;
			if (frame.empty())
				continue;
			cv::Mat resized;
			cv::resize(frame, resized, cv::Size(416, 416));
			ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR, resized.cols, resized.rows);
			const float mean_vals[3] = { .0f, .0f, .0f };
			const float norm_vals[3] = { 1.0 / 255.0,1.0 / 255.0, 1.0 / 255.0 };
			ncnn_img.substract_mean_normalize(mean_vals, norm_vals);

			ncnn::Extractor fea_ex = darknet.create_extractor();
			fea_ex.set_num_threads(4);
			fea_ex.set_light_mode(true);
			double time_start0 = (double)cv::getTickCount();
			fea_ex.input("data", ncnn_img);
			ncnn::Mat result;
			int a = fea_ex.extract("ConvNd_9", result);
			double time_end0 = (double)cv::getTickCount();
			double t0 = (time_end0 - time_start0) * 1000 / cv::getTickFrequency();
			std::cout << "ncnn 提取模型首次时间为：" << t0 << "ms" << std::endl;
			if (a == 0)
			{
				std::cout << "前向传到成功" << std::endl;
				std::vector<ObjectBB> boxes = do_detect(result);
				std::vector<ObjectBB> nms_boxes;
				nms(boxes, 0.4, nms_boxes);
				std::cout << "获取boxes的大小为" << nms_boxes.size() << std::endl;
				plot_boxes_cv(frame, nms_boxes, cls_names);
			}
			cv::imshow("show", frame);
			int key = cv::waitKey(1);
			if (key == 13)
				break;
		}
	}
	else if(option == "s")
	{
		cv::Mat cv_img = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
		cv::Mat show_img = cv_img.clone();
		if (cv_img.empty())
		{
			fprintf(stderr, "cv::imread %s failed\n", imagepath);
			return -1;
		}
		cv::Mat resized;
		cv::resize(cv_img, resized, cv::Size(416, 416));
		ncnn::Mat test_img = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR, resized.cols, resized.rows);
		const float mean_vals[3] = { .0f, .0f, .0f };
		const float norm_vals[3] = { 1.0 / 255.0,1.0 / 255.0, 1.0 / 255.0 };
		test_img.substract_mean_normalize(mean_vals, norm_vals);
		ncnn::Extractor fea_ex = darknet.create_extractor();
		fea_ex.set_num_threads(4);
		fea_ex.set_light_mode(true);
		ncnn::Mat test_res;
		fea_ex.input("data", test_img);
		fea_ex.extract("ConvNd_9", test_res);
		std::vector<ObjectBB> boxes = do_detect(test_res);
		std::vector<ObjectBB> nms_boxes;
		nms(boxes, 0.4, nms_boxes);
		std::cout << "获取boxes的大小为" << nms_boxes.size() << std::endl;
		plot_boxes_cv(show_img, nms_boxes, cls_names);
		cv::imshow("test_img", show_img);
		cv::waitKey(20);
	}	
	
	system("pause");
}
