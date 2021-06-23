/*
 * yolov3 module
 * */
// tvm 
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
// opencv 
#include <opencv4/opencv2/opencv.hpp>
// system
#include <cstdio>
#include <fstream>
#include <sys/time.h>

double GetCurTime(void)
{
    struct timeval tm;
    gettimeofday(&tm, 0);
    return tm.tv_usec + tm.tv_sec * 1000000;
}

void hwc_to_chw(cv::Mat src, cv::Mat dst)
{
    const int src_h = src.rows;
    const int src_w = src.cols;
    const int src_c = src.channels();

    cv::Mat hw_c = src.reshape(1, src_h * src_w);
    LOG(INFO) << "[yolov3 tvm]:hw_c:" << hw_c.channels() << ":" << hw_c.rows << ":" << hw_c.cols;

    cv::Mat c_hw = cv::Mat();
    cv::transpose(hw_c, c_hw);
    LOG(INFO) << "[yolov3 tvm]:c_hw:" << c_hw.channels() << ":" << c_hw.rows << ":" << c_hw.cols;

    //dst = c_hw.reshape(3, src_h);
    dst = c_hw;
    //const std::array<int,3> dims = {src_c, src_h, src_w};
    //dst.create(3,  &dims[0], CV_MAKETYPE(src.depth(), 1));
    
    LOG(INFO) << "[yolov3 tvm]:dst:" << dst.channels() << " : " << dst.rows << " : " << dst.cols;
    //LOG(INFO) << "[yolov3 tvm]:dst:" << dst.getMat().channels() << " : " << dst.getMat().rows << " : " << dst.getMat().cols;

    //cv::Mat dst_1d = dst.getMat().reshape(1, {src_c, src_h, src_w});

    //LOG(INFO) << "[yolov3 tvm]:dst_1d:" << dst_1d.channels() << " : " << dst_1d.rows << " : " << dst_1d.cols;
    //cv::transpose(hw_c, dst_1d);
    //LOG(INFO) << "[yolov3 tvm]:dst_1d:" << dst_1d.channels() << " : " << dst_1d.rows << " : " << dst_1d.cols;
    //LOG(INFO) << "[yolov3 tvm]:dst:" << dst.getMat().channels() << " : " << dst.getMat().rows << " : " << dst.getMat().cols;

    
}
/*
 *
 *
 */
cv::Mat ConvertImage(cv::Mat img)
{
    int img_h = img.rows;
    int img_w = img.cols;

    cv::Mat img_ex = cv::Mat(img_h, img_w, CV_8UC3);
    //cv::Mat img_chw = cv::Mat(img_h, img_w, CV_8UC3);

    LOG(INFO) << "[yolov3 tvm]:bgr to rgb"<< img_ex.dims << img_ex.rows << img_ex.cols;
    cv::cvtColor(img, img_ex, cv::COLOR_BGR2RGB);
    //img_ex.transpose(2, 0, 1);

//    LOG(INFO) << "[yolov3 tvm]:hwc-data" << img_ex.at<cv::Vec3b>(0, 0)[0];

  //  LOG(INFO) << "[yolov3 tvm]:hwc to chw" << img_ex.dims << img_ex.rows << img_ex.cols;
   // hwc_to_chw(img_ex, img_chw);
    //LOG(INFO) << "[yolov3 tvm]:divide" << img_chw.dims << img_chw.rows << img_chw.cols;

    img_ex.convertTo(img_ex, CV_32FC3, 1.0/255, 0);

    //LOG(INFO) << "[yolov3 tvm]:flip" << img_ex.channels() << img_ex.rows << img_ex.cols;

    //LOG(INFO) << "[yolov3 tvm]:flip-data" << img_ex.at<cv::Vec3f>(0, 0)[0];
    //LOG(INFO) << "[yolov3 tvm]:flip-data" << img_ex.at<cv::Vec3f>(0, 0)[1];
    //LOG(INFO) << "[yolov3 tvm]:flip-data" << img_ex.at<cv::Vec3f>(0, 0)[2];
    /*
    for (int j = 0; j < img_h; j++)
    {
        for (int i = 0; i < img_w; i++)
        {
            //LOG(INFO) << "[yolov3 tvm]:flip" << i << ":" << j;
            float tmp = img_ex.at<cv::Vec3b>(j, i)[0];
            img_ex.at<cv::Vec3b>(j, i)[0] = img_ex.at<cv::Vec3b>(j, i)[2];
            img_ex.at<cv::Vec3b>(j, i)[2] = tmp;
        }
    }*/
    return img_ex;
}

cv::Mat LetterBox(cv::Mat img, int in_w, int in_h)
{
    int img_w = img.cols;
    int img_h = img.rows;

    int new_w = 0;
    int new_h = 0;

    cv::Mat resize_img = cv::Mat();

    if ((in_w * 1.0 / img_w) < (in_h*1.0 / img_h))
    {
       new_w = in_w;
       new_h = img_h * in_w / img_w;
    }
    else
    {
        new_h = in_h;
        new_w = img_w * in_h / img_h;
    }

    cv::resize(img, resize_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);
    // h x w
    LOG(INFO) << "[yolov3 tvm]:Resize Shape: " << resize_img.rows << "X" << resize_img.cols;
    resize_img = ConvertImage(resize_img);

    LOG(INFO) << "[yolov3 tvm]:Remap Image size";
    cv::Mat boxed(in_h, in_w, CV_32FC3, cv::Scalar(0.5, 0.5, 0.5));
    LOG(INFO) << "[yolov3 tvm]:boxed-data:" << boxed.at<cv::Vec3f>(0, 0)[0];

    LOG(INFO) << "[yolov3 tvm]:boxed:" << boxed.channels() << " : " << boxed.rows << " : " << boxed.cols;
    int offset_w = (in_w - new_w) / 2; 
    int offset_h = (in_h - new_h) / 2; 

    LOG(INFO) << "[yolov3 tvm]:offset " << offset_w << ":" << offset_h;
    for (int j = 0; j < new_h; j++)
    {
        for (int i = 0; i < new_w; i++)
        {
            boxed.at<cv::Vec3f>(j + offset_h,i + offset_w)[0] = resize_img.at<cv::Vec3f>(j,i)[2];
            boxed.at<cv::Vec3f>(j + offset_h,i + offset_w)[1] = resize_img.at<cv::Vec3f>(j,i)[1];
            boxed.at<cv::Vec3f>(j + offset_h,i + offset_w)[2] = resize_img.at<cv::Vec3f>(j,i)[0];
        }
    }
    // transpose
    boxed = boxed.reshape(0, 1);
    LOG(INFO) << "[yolov3 tvm]:transpose:" << boxed.channels() << ":" << boxed.rows << ":" << boxed.cols;

    cv::Mat ex_boxed(in_h, in_w, CV_32FC3);
    cv::transpose(boxed, ex_boxed);
    LOG(INFO) << "[yolov3 tvm]:ex_boxed:" << ex_boxed.channels() << ":" << ex_boxed.rows << ":" << ex_boxed.cols;

    return boxed;
}

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        LOG(INFO) << "[yolov3 tvm]:Image Path: " << argv[1];
        LOG(INFO) << "[yolov3 tvm]:Dynamic Lib Path: " << argv[2];
        LOG(INFO) << "[yolov3 tvm]:Parameter Path: " << argv[3];
        LOG(INFO) << "[yolov3 tvm]:COCO Path: " << argv[3];
    }
    else
    {
        LOG(INFO) << "executor [img] [mod lib] [mod param] [coco]";
        return -1;
    }
    LOG(INFO) << "[yolov3 tvm]:Soft Version: V" << MNIST_VERSION;

    // read the image
    cv::Mat image, resize_image;
    image = cv::imread(argv[1]);
    if(image.data == nullptr){
        LOG(INFO) << "[yolov3 tvm]:Image don't exist!";
        return 0;
    }
    else{
        // image preprocess
        resize_image = LetterBox(image, 416, 416);
        //cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        //gray_image.convertTo(gray_image, CV_32FC3);

        LOG(INFO) << "[yolov3 tvm]:---Load Image--";
        LOG(INFO) << "[yolov3 tvm]:Image size: " << resize_image.rows << " X " << resize_image.cols;
        // cv::imshow("mnist image", gray_image);
        // cv::waitKey(0);
    }

    // create tensor
    DLTensor *x;
    //DLTensor *y;
    DLTensor *attr;
    DLTensor *biases;
    DLTensor *mask;
    DLTensor *data[3];
    int input_ndim  = 3;
    int output_ndim = 4;
    int64_t input_shape[3]  = {3, resize_image.rows, resize_image.cols};
    int64_t output_shape_attr[1] = {6};
    int64_t output_shape_biases[1] = {18};
    int64_t output_shape_mask[1] = {3};
    int64_t output_shape_data1[4] = {1, 255, 52, 52};
    int64_t output_shape_data2[4] = {1, 255, 26, 26};
    int64_t output_shape_data3[4] = {1, 255, 13, 13};

    int dtype_code  = kDLFloat;
    int dtype_bits  = 32;
    int dtype_lanes = 1;
    int device_type = kDLOpenCL;
    int device_id   = 0;

    TVMByteArray params_arr;
    DLDevice dev{kDLOpenCL, 0};

    // allocate the array space
    TVMArrayAlloc(input_shape, input_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

    TVMArrayAlloc(output_shape_attr, 1, kDLUInt, dtype_bits, dtype_lanes, device_type, device_id, &attr);
    TVMArrayAlloc(output_shape_mask, 1, kDLUInt, dtype_bits, dtype_lanes, device_type, device_id, &mask);
    TVMArrayAlloc(output_shape_biases, 1, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &biases);
    TVMArrayAlloc(output_shape_data1, output_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &data[0]);
    TVMArrayAlloc(output_shape_data2, output_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &data[1]);
    TVMArrayAlloc(output_shape_data3, output_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &data[2]);

    // the memory space allocate
    std::vector<uint32_t> attr_out(6);
    std::vector<float> data_out[3];
        
    // load the mnist dynamic lib
    LOG(INFO) << "[yolov3 tvm]:---Load Dynamic Lib---";
    tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(argv[2]);
    // get the mnist module
    tvm::runtime::Module mod = mod_dylib.GetFunction("yolov3")(dev);

    // load the mnist module parameters
    LOG(INFO) << "[yolov3 tvm]:---Load Parameters---";
    std::ifstream params_in(argv[3], std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();
    // get load parameters function
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    // get set input data function
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    for (int t = 0; t < atoi(argv[4]); t++)
    {
    LOG(INFO) << "[yolov3 tvm]:---Set Input---";
    // from cpu memory space copy data to gpu memory space
    TVMArrayCopyFromBytes(x, resize_image.data, 3 * resize_image.rows * resize_image.cols * sizeof(float));
    // using function set_input to configure
    set_input("data", x);

    LOG(INFO) << "[yolov3 tvm]:---Run---";
    // get run function
    double s1 = GetCurTime();
    run();
    double s2 = GetCurTime();

    // get output data function
    for (int i = 0; i < 3; i++)
    {
        get_output(4 * i + 3, attr);
        TVMArrayCopyToBytes(attr, attr_out.data(), 6 * sizeof(float));
        // 0   1     2     3     4     5
        // n out_c out_h out_w class total
        get_output(4 * i, data[i]);
        TVMArrayCopyToBytes(data[i], data_out[i].data(), attr_out[1] * attr_out[2] * attr_out[3] * sizeof(float));
    }
    double s3 = GetCurTime();

    LOG(INFO) << "[yolov3 tvm]:---Get Output---";
    LOG(INFO) << "[yolov3 tvm]:Run Time(run) " << (s2 - s1);
    LOG(INFO) << "[yolov3 tvm]:Run Time(get_out) " << (s3 - s2);
    }


    double s4 = GetCurTime();
    TVMArrayFree(x);
    TVMArrayFree(attr);
    TVMArrayFree(biases);
    TVMArrayFree(mask);
    TVMArrayFree(data[0]);
    TVMArrayFree(data[1]);
    TVMArrayFree(data[2]);
    double s5 = GetCurTime();
    LOG(INFO) << "[yolov3 tvm]:Run Time(Free) " << (s5 - s4);

    return 0;
}
