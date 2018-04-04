//
// Created by HeilsMac1 on 2018/3/19.
//

#include <net.h>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>


struct ObjectBB{
    cv::Rect rec;
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

void binary_multiplay(ncnn::Mat& a, ncnn::Mat& b,  ncnn::Mat& out)
{
    ncnn::Layer* op = ncnn::create_layer("BinaryOp");
    ncnn::ParamDict pd;
    pd.set(0,2);    // mul
    op->load_param(pd);

    // forward
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;

    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms,tops);

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

void binary_softmax(ncnn::Mat& in_place)
{
    ncnn::Layer* op = ncnn::create_layer("Softmax");
    ncnn::ParamDict pd;
    pd.set(0,1);

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
    for(int i=0;i<height;i++)
    {
        float* data_ptr = input.row(i);

        float max_conf = 0.0;
        float max_id = -1;
        for(int j=0; j<width;j++)
        {
            if(*(data_ptr) >max_conf)
            {
                max_conf = *(data_ptr++);
                max_id = j;
            }
        }
        *cls_conf++ = max_conf;
        *cls_max++ = static_cast<float>(max_id);
    }
}


std::vector<ObjectBB> forward(ncnn::Net &model, ncnn::Mat input)
{
    ncnn::Mat out;
    ncnn::Extractor ex = model.create_extractor();
    ex.set_light_mode(true);
    //ex.set_num_threads(4);
    ex.input("data", input);
    ex.extract("detection_out",out); // [channel,height,width] ----> out Mat shape [125,13,13]

    /// get the Object Bounding box from the output
    // output ---->c=125,w=13,h=13
    //# output: ----> [5,5+nC,h*w] : ------> [5+nc,5,h*w] ----> [5+nC, nA*nh*nw] == [25, nB*5*13*13]
    float conf_thresh = 0.5;
    float nms_thresh = 0.4;
    int num_classes = 20;
    int num_anchors = 5;
    float anchors[10] = {1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52};
    int h = out.h;
    int c = out.c;
	int w = out.w;
    out.reshape(h*w,5+num_classes,num_anchors); // [ w, h ,c]
    out.reshape(h*w, num_anchors,5+num_classes);
    out.reshape(h*w*num_anchors,5+num_classes); //[13*13*5,5+20]
	const int grid_step = num_anchors*h*w;

    // grid_x
    float* grid_x = new float[num_anchors*h*w];
    for(int i=0; i < num_anchors*h; i++)
    {
        for( size_t sz = 0 ; sz<w; sz++)
        {
            grid_x[num_anchors*h*i + sz] = w;
        }
    }

    // grid_y
    float* grid_y = new float[num_anchors*h*w];
    for (int i=0; i < num_anchors;i++)
    {
        for(int j=0; j < h; j++)
        {
            for(size_t sz = 0; sz<w; sz++)
            {
                grid_y[i*h*w + j*w + sz] = h;
            }
        }
    }


    ncnn::Mat grid_xM = ncnn::Mat(h*w*num_anchors,grid_x);
    ncnn::Mat grid_yM = ncnn::Mat(h*w*num_anchors, grid_y);
    float *out_0= out.row(0);
    float *out_1 = out.row(1);
    float *out_2 = out.row(2);
    float *out_3 = out.row(3);
    float *out_4 = out.row(4);

    ncnn::Mat x_score = ncnn::Mat(h*w*num_anchors, out_0);
    ncnn::Mat y_score = ncnn::Mat(h*w*num_anchors, out_1);
    // sigmod
    binary_sigmoid(x_score);
    binary_sigmoid(y_score);
    binary_add(x_score,grid_xM,x_score);
    binary_add(y_score,grid_xM,y_score);


    // define the anchors mesh grid
    float* anchor_gridw = new float[h*w*num_anchors];
    float* anchor_gridh = new float[h*w*num_anchors];
    for(int i=0; i <num_anchors ; i++)
    {
        for(int j=0 ; j<h*w; j++)
        {
            anchor_gridw[i*h*w + j] = anchors[i*2];
            anchor_gridh[i*h*w + j] = anchors[i*2 + 1];
        }
    }
    ncnn::Mat anchor_w = ncnn::Mat(h*w*num_anchors, anchor_gridw);
    ncnn::Mat anchor_h = ncnn::Mat(h*w*num_anchors, anchor_gridh);

    ncnn::Mat anchorw_score = ncnn::Mat(h*w*num_anchors, out_2);
    ncnn::Mat anchorh_score = ncnn::Mat(h*w*num_anchors, out_3);

    binary_exp(anchorw_score);
    binary_exp(anchorh_score);
    binary_add(anchorw_score,anchor_w,anchorw_score);
    binary_add(anchorh_score,anchor_h,anchorh_score);

    ncnn::Mat det_conf = ncnn::Mat(h*w*num_anchors,out_4);
    binary_sigmoid(det_conf);
    ncnn::Mat def_conf_flat = det_conf.reshape(det_conf.w * det_conf.h * det_conf.c);

    float* out_5 = out.row(5);
    ncnn::Mat cls = ncnn::Mat(h*w*num_anchors,num_classes,out_5);
    ncnn::Mat cls_mat;
    cvTranspose(cls,cls_mat);   //进行转置操作
    std::cout << "the cls_mat_transpose mat is  width "<<cls_mat.w << "height is "<<cls_mat.h <<std::endl;

    binary_softmax(cls_mat); // softmax [nA*nH*nW,nC]

    ncnn::Mat cls_conf_mat(h*w*num_anchors);
    ncnn::Mat cls_id_mat(h*w*num_anchors);

    bianry_max(cls_mat, cls_conf_mat, cls_id_mat);

    int sz_hw = h*w;

    std::vector<ObjectBB> boxes;
    for(int i=0; i<h; i++)
    {
        for(int j=0; j<w; j++)
        {
            for(int a=0; a<num_anchors; a++)
            {
                int ind = i*sz_hw + i*w + j;
                float cls_conf = cls_conf_mat[ind];
                float pro_conf = def_conf_flat[ind];
                float conf = pro_conf * cls_conf;

                ObjectBB bbx;

                if(conf > conf_thresh)
                {
                    int x1 = (int)(x_score[ind]/w - anchorw_score[ind]/(w*2.0)) * input.w;
                    int y1 = (int)(y_score[ind]/h - anchorw_score[ind]/(h*2.0)) * input.h;

                    int rectx = (int)anchorw_score[ind]/w * input.w;
                    int recty = (int)anchorh_score[ind]/h * input.h;

                    bbx.rec.x = x1;
                    bbx.rec.y = y1;
                    bbx.rec.width = rectx;
                    bbx.rec.height = recty;
                    bbx.class_id = (int)cls_id_mat[ind];
                    bbx.id_prob = cls_conf;
                    bbx.iou_prob = pro_conf;
                    boxes.push_back(bbx);
                }
            }
        }
    }
    return boxes;
}

int main()
{
    
}

