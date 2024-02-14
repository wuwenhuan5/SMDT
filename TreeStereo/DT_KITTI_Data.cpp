//#include "NLCCA.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "evaluate.h"
#include "StereoDisparity.h"
#include <iostream>
#include <fstream>
//#include "qx_nonlocal_cost_aggregation.h"

using namespace std;
using namespace cv;


int  main(int argc, char** argv)   // main_kitti
{
	printf("MUL-Tree Cost Aggregation on KITTI stereo pairs\n");

	string left_img_dir = "KITTI_training/2015/image_2";
	string right_img_dir = "KITTI_training/2015/image_3";
	string disparity_dir = "KITTI_training/results/2015/disp_0";

	string left_img_file, right_img_file, disparity_map_file;

	int maxLevel=256; //视差总数
	int scale=256;    //尺度
	float sigma=0.03;

	double duration;
	duration = static_cast<double>(getTickCount());

	int num = 10;
	double avg_time = 0;
	// for all test files do
	for (int32_t i = 0; i < num; i++) {

		// file name
		char prefix[256];
		sprintf(prefix, "%06d_10", i);

		printf("Processing: %s.png", prefix);

		left_img_file = left_img_dir + "/" + prefix + ".png";
		right_img_file = right_img_dir + "/" + prefix + ".png";
		disparity_map_file = disparity_dir + "/" + prefix + ".png";

		double duration;
		duration = static_cast<double>(cv::getTickCount());
		stereo_routine_kitti(left_img_file, right_img_file, disparity_map_file, maxLevel, scale, sigma, DT_REFINED); //DT_RAW
		duration = static_cast<double>(cv::getTickCount()) - duration;
		duration /= cv::getTickFrequency();
		avg_time = avg_time + duration;
	}

	avg_time = avg_time / num;

	printf("平均时间=%f\n", avg_time);
	ofstream fout("KITTI_training/results/_10avg_time.txt", ios::out | ios::app);
	fout << "the average time of 200 images : " << avg_time << endl;
	fout.close();



	return 0;
}

