#include "evaluate.h"
#include <fstream>
#include <iostream>
#include <windows.h>
#include "StereoDisparity.h"

using namespace std;
using namespace cv;

/*wwh test the 27 image pairs*/
/*
"Data_extended/Midd1/disp1.png",   "Data_extended/Midd1/disp5.png", "_Midd1.png", "Data_extended/Midd1/view1.png",  "Data_extended/Midd1/view5.png", "_Midd1_bad.png",
"Data_extended/Midd2/disp1.png",   "Data_extended/Midd2/disp5.png", "_Midd2.png", "Data_extended/Midd2/view1.png",  "Data_extended/Midd2/view5.png", "_Midd2_bad.png",
"Data_extended/Monopoly/disp1.png",   "Data_extended/Monopoly/disp5.png", "_Monopoly.png", "Data_extended/Monopoly/view1.png",  "Data_extended/Monopoly/view5.png", "_Monopoly_bad.png",
"Data_extended/Plastic/disp1.png",   "Data_extended/Plastic/disp5.png", "_Plastic.png", "Data_extended/Plastic/view1.png",  "Data_extended/Plastic/view5.png", "_Plastic_bad.png",
*/
/*,"Data_extended/Midd1/disp1.png", "Data_extended/Midd1/disp5.png", "_Midd1.png", "Data_extended/Midd1/view1.png", "Data_extended/Midd1/view5.png", "_Midd1_bad.png",
"Data_extended/Midd2/disp1.png", "Data_extended/Midd2/disp5.png", "_Midd2.png", "Data_extended/Midd2/view1.png", "Data_extended/Midd2/view5.png", "_Midd2_bad.png",
"Data_extended/Monopoly/disp1.png", "Data_extended/Monopoly/disp5.png", "_Monopoly.png", "Data_extended/Monopoly/view1.png", "Data_extended/Monopoly/view5.png", "_Monopoly_bad.png",
"Data_extended/Plastic/disp1.png", "Data_extended/Plastic/disp5.png", "_Plastic.png", "Data_extended/Plastic/view1.png", "Data_extended/Plastic/view5.png", "_Plastic_bad.png" */

char cTestFilePool[198][256] =
{

	"Data_extended/barn1/disp2.pgm",   "Data_extended/barn1/disp6.pgm", "_barn1.png",  "Data_extended/barn1/im2.ppm",  "Data_extended/barn1/im6.ppm", "_barn1_bad.png",
	"Data_extended/barn2/disp2.pgm",   "Data_extended/barn2/disp6.pgm", "_barn2.png",  "Data_extended/barn2/im2.ppm",  "Data_extended/barn2/im6.ppm", "_barn2_bad.png",
	"Data_extended/bull/disp2.pgm",   "Data_extended/bull/disp6.pgm", "_bull.png",  "Data_extended/bull/im2.ppm",  "Data_extended/bull/im6.ppm", "_bull_bad.png",
	"Data_extended/Map/disp0.pgm",   "Data_extended/Map/disp1.pgm", "_Map.png",  "Data_extended/Map/im0.pgm",  "Data_extended/Map/im1.pgm", "_Map_bad.png",
	"Data_extended/poster/disp2.pgm",   "Data_extended/poster/disp6.pgm", "_poster.png",  "Data_extended/poster/im2.ppm",  "Data_extended/poster/im6.ppm", "_poster_bad.png",
	"Data_extended/Sawtooth/disp2.pgm",   "Data_extended/Sawtooth/disp6.pgm", "_Sawtooth.png",  "Data_extended/Sawtooth/im2.ppm",  "Data_extended/Sawtooth/im6.ppm", "_Sawtooth_bad.png",

	"Data_extended/Aloe/disp1.png",   "Data_extended/Aloe/disp5.png", "_Aloe.png",  "Data_extended/Aloe/view1.png",  "Data_extended/Aloe/view5.png", "_Aloe_bad.png",
	"Data_extended/Art/disp1.png",    "Data_extended/Art/disp5.png", "_Art.png",    "Data_extended/Art/view1.png", "Data_extended/Art/view5.png", "_Art_bad.png",
	"Data_extended/Baby1/disp1.png",  "Data_extended/Baby1/disp5.png", "_Baby1.png", "Data_extended/Baby1/view1.png", "Data_extended/Baby1/view5.png", "_Baby1_bad.png",
	"Data_extended/Baby2/disp1.png",  "Data_extended/Baby2/disp5.png", "_Baby2.png", "Data_extended/Baby2/view1.png",  "Data_extended/Baby2/view5.png", "_Baby2_bad.png",
	"Data_extended/Baby3/disp1.png",  "Data_extended/Baby3/disp5.png", "_Baby3.png",  "Data_extended/Baby3/view1.png",  "Data_extended/Baby3/view5.png", "_Baby3_bad.png",
	"Data_extended/Books/disp1.png",   "Data_extended/Books/disp5.png", "_Books.png",  "Data_extended/Books/view1.png",  "Data_extended/Books/view5.png", "_Books_bad.png",
	"Data_extended/Bowling1/disp1.png",   "Data_extended/Bowling1/disp5.png", "_Bowling1.png",  "Data_extended/Bowling1/view1.png",  "Data_extended/Bowling1/view5.png", "_Bowling1_bad.png",
	"Data_extended/Bowling2/disp1.png",   "Data_extended/Bowling2/disp5.png", "_Bowling2.png",  "Data_extended/Bowling2/view1.png",  "Data_extended/Bowling2/view5.png", "_Bowling2_bad.png",
	"Data_extended/Cloth1/disp1.png",  "Data_extended/Cloth1/disp5.png", "_Cloth1.png",  "Data_extended/Cloth1/view1.png",  "Data_extended/Cloth1/view5.png", "_Cloth1_bad.png",
	"Data_extended/Cloth2/disp1.png",  "Data_extended/Cloth2/disp5.png", "_Cloth2.png",  "Data_extended/Cloth2/view1.png",  "Data_extended/Cloth2/view5.png", "_Cloth2_bad.png",
	"Data_extended/Cloth3/disp1.png",  "Data_extended/Cloth3/disp5.png", "_Cloth3.png",  "Data_extended/Cloth3/view1.png",  "Data_extended/Cloth3/view5.png", "_Cloth3_bad.png",
	"Data_extended/Cloth4/disp1.png",  "Data_extended/Cloth4/disp5.png", "_Cloth4.png",  "Data_extended/Cloth4/view1.png",  "Data_extended/Cloth4/view5.png", "_Cloth4_bad.png",
	"Data_extended/Dolls/disp1.png",   "Data_extended/Dolls/disp5.png", "_Dolls.png",  "Data_extended/Dolls/view1.png",   "Data_extended/Dolls/view5.png", "_Dolls_bad.png",
	"Data_extended/Flowerpots/disp1.png", "Data_extended/Flowerpots/disp5.png", "_Flowerpots.png", "Data_extended/Flowerpots/view1.png", "Data_extended/Flowerpots/view5.png", "_Flowerpots_bad.png",
	"Data_extended/Lampshade1/disp1.png", "Data_extended/Lampshade1/disp5.png", "_Lampshade1.png", "Data_extended/Lampshade1/view1.png", "Data_extended/Lampshade1/view5.png", "_Lampshade1_bad.png",
	"Data_extended/Lampshade2/disp1.png", "Data_extended/Lampshade2/disp5.png", "_Lampshade2.png", "Data_extended/Lampshade2/view1.png", "Data_extended/Lampshade2/view5.png", "_Lampshade2_bad.png",
	"Data_extended/Laundry/disp1.png",   "Data_extended/Laundry/disp5.png", "_Laundry.png", "Data_extended/Laundry/view1.png", "Data_extended/Laundry/view5.png", "_Laundry_bad.png",
	"Data_extended/Moebius/disp1.png",  "Data_extended/Moebius/disp5.png", "_Moebius.png", "Data_extended/Moebius/view1.png", "Data_extended/Moebius/view5.png", "_Moebius_bad.png",
	"Data_extended/Reindeer/disp1.png",  "Data_extended/Reindeer/disp5.png", "_Reindeer.png", "Data_extended/Reindeer/view1.png",  "Data_extended/Reindeer/view5.png", "_Reindeer_bad.png",
	"Data_extended/Rocks1/disp1.png",  "Data_extended/Rocks1/disp5.png", "_Rocks1.png", "Data_extended/Rocks1/view1.png",  "Data_extended/Rocks1/view5.png", "_Rocks1_bad.png",
	"Data_extended/Rocks2/disp1.png",  "Data_extended/Rocks2/disp5.png", "_Rocks2.png", "Data_extended/Rocks2/view1.png",  "Data_extended/Rocks2/view5.png", "_Rocks2_bad.png",
	"Data_extended/Wood1/disp1.png",  "Data_extended/Wood1/disp5.png", "_Wood1.png", "Data_extended/Wood1/view1.png",  "Data_extended/Wood1/view5.png", "_Wood1_bad.png",
	"Data_extended/Wood2/disp1.png",  "Data_extended/Wood2/disp5.png", "_Wood2.png", "Data_extended/Wood2/view1.png",  "Data_extended/Wood2/view5.png", "_Wood2_bad.png",

	"Data_extended/Midd1/disp1.png", "Data_extended/Midd1/disp5.png", "_Midd1.png", "Data_extended/Midd1/view1.png", "Data_extended/Midd1/view5.png", "_Midd1_bad.png",
	"Data_extended/Midd2/disp1.png", "Data_extended/Midd2/disp5.png", "_Midd2.png", "Data_extended/Midd2/view1.png", "Data_extended/Midd2/view5.png", "_Midd2_bad.png",
	"Data_extended/Monopoly/disp1.png", "Data_extended/Monopoly/disp5.png", "_Monopoly.png", "Data_extended/Monopoly/view1.png", "Data_extended/Monopoly/view5.png", "_Monopoly_bad.png",
	"Data_extended/Plastic/disp1.png", "Data_extended/Plastic/disp5.png", "_Plastic.png", "Data_extended/Plastic/view1.png", "Data_extended/Plastic/view5.png", "_Plastic_bad.png"
};

char extendNames[33][256] = {
	"barn1","barn2","bull","Map","poster","Sawtooth","Aloe", "Art", "Baby1", "Baby2", "Baby3", "Books","Bowling1","Bowling2", "Cloth1", "Cloth2", "Cloth3","Cloth4",
	"Dolls", "Flowerpots", "Lampshade1", "Lampshade2", "Laundry","Moebius", "Reindeer","Rocks1","Rocks2","Wood1","Wood2",
	"Midd1", "Midd2","Monopoly","Plastic"
}; //,

string filename = "27_dataset_sigma.txt";


float TestRoutine_Batch_MiddleburyExtended(float&pixel_avg_err, float sigma, METHOD method)//test non-occ regions only
{
	float err_ave = 0;
	float pixel_err = 0;
	double time_ave = 0;
	pixel_avg_err = 0;
	ofstream fout(filename, ios::out | ios::app);

	int scale = 8;
	int numImg1 = 6;

	string dispFile = "maxdisps of 33 stereo matching in Data2.txt"; //	
	ofstream dfout(dispFile, ios::out | ios::app);
	
	/*
	for (int i = 0; i < numImg1; i++)
	{
		int idx = i * 6;
		char* gt_name_l = cTestFilePool[idx];
		char* gt_name_r = cTestFilePool[idx + 1];
		char* filename_disparity_map = cTestFilePool[idx + 2];
		char* filename_left_image = cTestFilePool[idx + 3];
		char* filename_right_image = cTestFilePool[idx + 4];
		char* filename_bad_image = cTestFilePool[idx + 5];
		int max_disparity = CompDepthRange(gt_name_l, scale) + 1;
		dfout << endl << max_disparity << "  ";

		Mat lImg = imread(filename_left_image, CV_LOAD_IMAGE_COLOR);
		Mat rImg = imread(filename_right_image, CV_LOAD_IMAGE_COLOR);
		if (!lImg.data || !rImg.data) {
			printf("Error: can not open image\n");
			printf("\nPress any key to continue...\n");
			getchar();
			return -1;
		}
	
		cv::Mat dispImg;
		stereo_routine(filename_left_image, filename_right_image, filename_disparity_map, max_disparity, scale, sigma, method); //聚合滤波 

		float err_non_occ = TestResult_OnePixel_NonOcc(filename_disparity_map, gt_name_l, gt_name_r, scale, pixel_err, filename_bad_image);
		//printf("  %s: %.02f%%\n", extendNames[i], 100 * err_non_occ);
		fout << extendNames[i] << "   " << 100 * err_non_occ << "   " << pixel_err << endl;
		err_ave += err_non_occ * 100;
		pixel_avg_err += pixel_err;
		cout << "images  " << i + 1 << "  is over!" << endl;

	}
	*/

	scale = 3;
	int numImg = 29; //29
	for (int i = numImg1; i < numImg; i++)
	{
		int idx = i * 6;
		char* gt_name_l = cTestFilePool[idx];
		char* gt_name_r = cTestFilePool[idx + 1];
		char* filename_disparity_map = cTestFilePool[idx + 2];
		char* filename_left_image = cTestFilePool[idx + 3];
		char* filename_right_image = cTestFilePool[idx + 4];
		char* filename_bad_image = cTestFilePool[idx + 5];
		int max_disparity = CompDepthRange(gt_name_l, scale) + 1;
		dfout << endl << max_disparity << "  ";

		Mat lImg = imread(filename_left_image, CV_LOAD_IMAGE_COLOR);
		Mat rImg = imread(filename_right_image, CV_LOAD_IMAGE_COLOR);
		if (!lImg.data || !rImg.data) {
			printf("Error: can not open image\n");
			printf("\nPress any key to continue...\n");
			getchar();
			return -1;
		}

		cv::Mat dispImg;
		stereo_routine(filename_left_image, filename_right_image, filename_disparity_map, max_disparity, scale, sigma, method); //聚合滤波

		float err_non_occ = TestResult_OnePixel_NonOcc(filename_disparity_map, gt_name_l, gt_name_r, scale, pixel_err, filename_bad_image);
		//printf("  %s: %.02f%%\n", extendNames[i], 100 * err_non_occ);
		fout << extendNames[i] << "   " << 100 * err_non_occ << "   " << pixel_err << endl;
		err_ave += err_non_occ * 100;
		pixel_avg_err += pixel_err;
		cout << "images  " << i + 1 << "  is over!" << endl;

	}

	dfout.close();

	err_ave /= (numImg - numImg1);
	pixel_avg_err /= (numImg - numImg1);

	fout << "err_ave of all 23 images: ratio_err_ave=" << err_ave << "%, pixel_err_ave=" << pixel_avg_err << endl << endl;
	fout.close();
	return err_ave;
}

float Para_TestRoutine_Batch_MiddleburyExtended(float para, float&pixel_avg_err, float sigma, METHOD method)//test non-occ regions only
{
	float err_ave = 0;
	float pixel_err = 0;
	double time_ave = 0;
	pixel_avg_err = 0;
	ofstream fout(filename, ios::out | ios::app);


	int scale = 3;
	int numImg = 23;

	//定义一个数组，进行训练参数
	int set[23] = {6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28};//{ 20,21,6,7,9,11,12,18,19,22,23,24 }; //for (int k = 0; k < numImg; k++)

	for (int k = 0; k < numImg; k++)
	{
		int i = set[k];
		int idx = i * 6;
		char* gt_name_l = cTestFilePool[idx];
		char* gt_name_r = cTestFilePool[idx + 1];
		char* filename_disparity_map = cTestFilePool[idx + 2];
		char* filename_left_image = cTestFilePool[idx + 3];
		char* filename_right_image = cTestFilePool[idx + 4];
		char* filename_bad_image = cTestFilePool[idx + 5];
		int max_disparity = CompDepthRange(gt_name_l, scale) + 1;

		Mat lImg = imread(filename_left_image, CV_LOAD_IMAGE_COLOR);
		Mat rImg = imread(filename_right_image, CV_LOAD_IMAGE_COLOR);
		if (!lImg.data || !rImg.data) {
			printf("Error: can not open image\n");
			printf("\nPress any key to continue...\n");
			getchar();
			return -1;
		}

		cv::Mat dispImg;
		stereo_routine(filename_left_image, filename_right_image, filename_disparity_map, max_disparity, scale, sigma, method); //聚合滤波

		float err_non_occ = TestResult_OnePixel_NonOcc(filename_disparity_map, gt_name_l, gt_name_r, scale, pixel_err, filename_bad_image);
		//printf("  %s: %.02f%%\n", extendNames[i], 100 * err_non_occ);
		fout << extendNames[i] << "   " << 100 * err_non_occ << "   " << pixel_err << endl;
		err_ave += err_non_occ * 100;
		pixel_avg_err += pixel_err;
		cout << "images  " << i + 1 << "  is over!" << endl;

	}

	err_ave /= numImg;
	pixel_avg_err /= numImg;
	fout << "err_ave of all 23 images: ratio_err_ave=" << err_ave << "%, pixel_err_ave=" << pixel_avg_err << endl << endl;
	fout.close();
	return err_ave;
}

float TestRoutine_Batch_MiddleburyStandard(float&pixel_avg_err, float sigma, METHOD method) {

	cv::Mat dispImg1, dispImg2, dispImg3, dispImg4;
	double duration;
	duration = static_cast<double>(getTickCount());
	stereo_routine("Data/tsukuba/imL.png", "Data/tsukuba/imR.png", "tsukuba_dis.png", 16, 16, sigma, method);//匹配
	stereo_routine("Data/venus/imL.png", "Data/venus/imR.png", "venus_dis.png", 20, 8, sigma, method);//匹配
	stereo_routine("Data/teddy/imL.png", "Data/teddy/imR.png", "teddy_dis.png", 60, 4, sigma, method);//匹配
	stereo_routine("Data/cones/imL.png", "Data/cones/imR.png", "cones_dis.png", 60, 4, sigma, method);//匹配
	duration = static_cast<double>(getTickCount()) - duration;
	duration /= cv::getTickFrequency(); // the elapsed time in sec
	double time_ave = duration / 4;
	ofstream fout(filename, ios::out | ios::app); //open("a.txt",ios::out|ios::app);

	float err_average = 0, err_average_non_occ = 0, err_non_occ = 0, pixel_err = 0; pixel_avg_err = 0;
	err_average += TestResult_OnePixel_Val("tsukuba_dis.png", "Data/tsukuba/groundtruth.bmp", "Data/tsukuba/mask.bmp", 16, pixel_err);
	err_average += TestResult_OnePixel_Val("tsukuba_dis.png", "Data/tsukuba/groundtruth.bmp", "Data/tsukuba/tsukuba_mask_disc.bmp", 16, pixel_err);
	err_average += err_non_occ = TestResult_OnePixel_Val("tsukuba_dis.png", "Data/tsukuba/groundtruth.bmp", "Data/tsukuba/tsukuba_mask_nonocc.bmp", 16, pixel_err, "tsukuba_bad.png");
	printf("  Tsukuba: %.02f%%\n", 100 * err_non_occ);
	fout << "tsukuba" << "  " << 100 * err_non_occ << "   " << pixel_err << endl;
	err_average_non_occ += err_non_occ;
	pixel_avg_err += pixel_err;

	err_average += TestResult_OnePixel_Val("venus_dis.png", "Data/venus/groundtruth.png", "Data/venus/mask.bmp", 8, pixel_err);
	err_average += TestResult_OnePixel_Val("venus_dis.png", "Data/venus/groundtruth.png", "Data/venus/venus_mask_disc.bmp", 8, pixel_err);
	err_average += err_non_occ = TestResult_OnePixel_Val("venus_dis.png", "Data/venus/groundtruth.png", "Data/venus/venus_mask_nonocc.bmp", 8, pixel_err, "venus_bad.png");
	printf("  Venus: %.02f%%\n", 100 * err_non_occ);
	fout << "venus" << "  " << 100 * err_non_occ << "   " << pixel_err << endl;
	err_average_non_occ += err_non_occ;
	pixel_avg_err += pixel_err;

	err_average += TestResult_OnePixel_Val("teddy_dis.png", "Data/teddy/groundtruth.png", "Data/teddy/mask.bmp", 4, pixel_err);
	err_average += TestResult_OnePixel_Val("teddy_dis.png", "Data/teddy/groundtruth.png", "Data/teddy/teddy_mask_disc.bmp", 4, pixel_err);
	err_average += err_non_occ = TestResult_OnePixel_Val("teddy_dis.png", "Data/teddy/groundtruth.png", "Data/teddy/teddy_mask_nonocc.bmp", 4, pixel_err, "teddy_bad.png");
	printf("  Teddy: %.02f%%\n", 100 * err_non_occ);
	fout << "Teddy" << "  " << 100 * err_non_occ << "    " << pixel_err << endl;
	err_average_non_occ += err_non_occ;
	pixel_avg_err += pixel_err;

	err_average += TestResult_OnePixel_Val("cones_dis.png", "Data/cones/groundtruth.png", "Data/cones/mask.bmp", 4, pixel_err);
	err_average += TestResult_OnePixel_Val("cones_dis.png", "Data/cones/groundtruth.png", "Data/cones/cones_mask_disc.bmp", 4, pixel_err);
	err_average += err_non_occ = TestResult_OnePixel_Val("cones_dis.png", "Data/cones/groundtruth.png", "Data/cones/cones_mask_nonocc.bmp", 4, pixel_err, "cones_bad.png");
	printf("  Cones: %.02f%%\n", 100 * err_non_occ);
	fout << "Cones" << "  " << 100 * err_non_occ << "   " << pixel_err << endl;
	err_average_non_occ += err_non_occ;
	pixel_avg_err += pixel_err;

	err_average_non_occ = 100 * err_average_non_occ / 4;
	pixel_avg_err /= 4;

	fout << "err_ave of std 4 images: ratio_err_ave=" << err_average_non_occ << "%, pixel_err_ave=" << pixel_avg_err << endl;
	fout << endl << "the average time of std images : " << time_ave << endl;
	fout.close();

	return err_average_non_occ;
	//平均误差
}

float TestRoutine_Batch_MiddleburyStandard2(float para, float&pixel_avg_err, float sigma, METHOD method) {

	cv::Mat dispImg1, dispImg2, dispImg3, dispImg4;
	double duration;
	duration = static_cast<double>(getTickCount());
	stereo_routine("Data/tsukuba/imL.png", "Data/tsukuba/imR.png", "tsukuba_dis.png", 16, 16, sigma, method);//匹配
	stereo_routine("Data/venus/imL.png", "Data/venus/imR.png", "venus_dis.png", 20, 8, sigma, method);//匹配
	stereo_routine("Data/teddy/imL.png", "Data/teddy/imR.png", "teddy_dis.png", 60, 4, sigma, method);//匹配
	stereo_routine("Data/cones/imL.png", "Data/cones/imR.png", "cones_dis.png", 60, 4, sigma, method);//匹配
	duration = static_cast<double>(getTickCount()) - duration;
	duration /= cv::getTickFrequency(); // the elapsed time in sec
	double time_ave = duration / 4;
	ofstream fout(filename, ios::out | ios::app); //open("a.txt",ios::out|ios::app);
	float err_average = 0, err_average_non_occ = 0, err_non_occ = 0, pixel_err = 0; pixel_avg_err = 0;
	float err_average11 = TestResult_OnePixel_Val("tsukuba_dis.png", "Data/tsukuba/groundtruth.bmp", "Data/tsukuba/mask.bmp", 16, pixel_err);
	float err_average12 = TestResult_OnePixel_Val("tsukuba_dis.png", "Data/tsukuba/groundtruth.bmp", "Data/tsukuba/tsukuba_mask_disc.bmp", 16, pixel_err);
	float err_average13 = err_non_occ = TestResult_OnePixel_Val("tsukuba_dis.png", "Data/tsukuba/groundtruth.bmp", "Data/tsukuba/tsukuba_mask_nonocc.bmp", 16, pixel_err, "tsukuba_bad.png");
	printf("  Tsukuba: %.02f%%\n", 100 * err_non_occ);
	//fout << endl << "tsukuba" << "  " << 100 * err_average13 << "  " << 100 * err_average11 << "    " << 100 * err_average12 << endl;
	fout << "tsukuba" << "  " << 100 * err_non_occ << "   " << pixel_err << endl;
	err_average_non_occ += err_non_occ;
	pixel_avg_err += pixel_err;


	float err_average21 = TestResult_OnePixel_Val("venus_dis.png", "Data/venus/groundtruth.png", "Data/venus/mask.bmp", 8, pixel_err);
	float err_average22 = TestResult_OnePixel_Val("venus_dis.png", "Data/venus/groundtruth.png", "Data/venus/venus_mask_disc.bmp", 8, pixel_err);
	float err_average23 = err_non_occ = TestResult_OnePixel_Val("venus_dis.png", "Data/venus/groundtruth.png", "Data/venus/venus_mask_nonocc.bmp", 8, pixel_err, "venus_bad.png");
	printf("  Venus: %.02f%%\n", 100 * err_non_occ);
	//fout << "venus" << "  " << 100 * err_average23 << "  " << 100 * err_average21 << "    " << 100 * err_average22 << endl;
	fout << "venus" << "  " << 100 * err_non_occ << "   " << pixel_err << endl;
	err_average_non_occ += err_non_occ;
	pixel_avg_err += pixel_err;


	float err_average31 = TestResult_OnePixel_Val("teddy_dis.png", "Data/teddy/groundtruth.png", "Data/teddy/mask.bmp", 4, pixel_err);
	float err_average32 = TestResult_OnePixel_Val("teddy_dis.png", "Data/teddy/groundtruth.png", "Data/teddy/teddy_mask_disc.bmp", 4, pixel_err);
	float err_average33 = err_non_occ = TestResult_OnePixel_Val("teddy_dis.png", "Data/teddy/groundtruth.png", "Data/teddy/teddy_mask_nonocc.bmp", 4, pixel_err, "teddy_bad.png");
	printf("  Teddy: %.02f%%\n", 100 * err_non_occ);
	//fout << "Teddy" << "  " << 100 * err_average33 << "  " << 100 * err_average31 << "    " << 100 * err_average32 << endl;
	fout << "Teddy" << "  " << 100 * err_non_occ << "    " << pixel_err << endl;
	err_average_non_occ += err_non_occ;
	pixel_avg_err += pixel_err;


	float err_average41 = TestResult_OnePixel_Val("cones_dis.png", "Data/cones/groundtruth.png", "Data/cones/mask.bmp", 4, pixel_err);
	float err_average42 = TestResult_OnePixel_Val("cones_dis.png", "Data/cones/groundtruth.png", "Data/cones/cones_mask_disc.bmp", 4, pixel_err);
	float err_average43 = err_non_occ = TestResult_OnePixel_Val("cones_dis.png", "Data/cones/groundtruth.png", "Data/cones/cones_mask_nonocc.bmp", 4, pixel_err, "cones_bad.png");
	printf("  Cones: %.02f%%\n", 100 * err_non_occ);
	//fout << "Cones" << "  " << 100 * err_average43 << "  " << 100 * err_average41 << "    " << 100 * err_average42 << endl;
	fout << " Cones" << "  " << 100 * err_non_occ << "   " << pixel_err << endl;
	err_average_non_occ += err_non_occ;
	pixel_avg_err += pixel_err;
	

	float err_average3 = 100 * (err_average13 + err_average23 + err_average33 + err_average43) / 4;
	float err_average1 = 100 * (err_average11 + err_average21 + err_average31 + err_average41) / 4;
	float err_average2 = 100 * (err_average12 + err_average22 + err_average32 + err_average42) / 4;

	err_average_non_occ = 100 * err_average_non_occ / 4;
	pixel_avg_err /= 4;

	//fout << "err_ave of std 4 images: non, all, dis=%  " << err_average3 << "     " << err_average1 << "   " << err_average2 << endl;
	//fout << "avg of non,all,dis  " << (err_average3 + err_average1 + err_average2) / 3 << endl;
	//fout << " pixel_err_ave=" << pixel_avg_err << endl;
	//fout << endl << "the average time of std images : " << time_ave << endl;
	//fout.close();

	fout << "err_ave of std 4 images: ratio_err_ave=" << err_average_non_occ << "%, pixel_err_ave=" << pixel_avg_err << endl;
	fout << endl << "the average time of std images : " << time_ave << endl;
	fout.close();

	Generate_error_map("tsukuba_dis.png", "Data/tsukuba/groundtruth.bmp", "Data/tsukuba/mask.bmp", "Data/tsukuba/tsukuba_mask_nonocc.bmp", "Data/tsukuba/tsukuba_mask_disc.bmp", 16, "tsukuba_error_map.bmp");
	Generate_error_map("venus_dis.png", "Data/venus/groundtruth.png", "Data/venus/mask.bmp", "Data/venus/venus_mask_nonocc.bmp", "Data/venus/venus_mask_disc.bmp", 8, "venus_error_map.bmp");
	Generate_error_map("teddy_dis.png", "Data/teddy/groundtruth.png", "Data/teddy/mask.bmp", "Data/teddy/teddy_mask_nonocc.bmp", "Data/teddy/teddy_mask_disc.bmp", 4, "teddy_error_map.bmp");
	Generate_error_map("cones_dis.png", "Data/cones/groundtruth.png", "Data/cones/mask.bmp", "Data/cones/cones_mask_nonocc.bmp", "Data/cones/cones_mask_disc.bmp", 4, "cones_error_map.bmp");
	return err_average_non_occ;//平均非遮挡误差
	//return (err_average3 + err_average1 + err_average2) / 3;
}

int main_Data2(int argc, char** argv)   // main_Data2
{

	printf("GF Cost Aggregation on 33 image stereo pairs\n");

	float sigma = 0.03;
	float  pixel_avg_err1 = 0, pixel_avg_err2 = 0;

	float error_rate2 = TestRoutine_Batch_MiddleburyExtended(pixel_avg_err2, sigma, DT_REFINED);  //先打开, OLT_RAW, OLT_REFINED
	float error_rate1 = TestRoutine_Batch_MiddleburyStandard(pixel_avg_err1, sigma, DT_REFINED);  //后追加, OLT_RAW, OLT_REFINED

	float error_rate = (4 * error_rate1 + 23 * error_rate2) / 27;
	float pixel_error = (4 * pixel_avg_err1 + 23 * pixel_avg_err2) / 27;
	printf("Avg. non-occ err in 27 pairs: %.02f%%\n", error_rate);

	ofstream fout(filename, ios::out | ios::app);

	fout << endl << "std+ext error rate:" << error_rate << endl;
	fout << endl << "std+ext pixel_error:" << pixel_error << endl << endl;

	fout.close();
	return 1;
}

int main_Data2paratest() // 调参
{
	ofstream fout(filename, ios::out | ios::app);

	float sigma = 0.05;
	float pixel_avg_err1 = 0, pixel_avg_err2 = 0;

	float para = 0;
	for (para = 0.19; para <= 0.21; para += 0.01) {
		fout << endl << "sigma=" << para << endl;

		sigma = para;
		pixel_avg_err1 = 0, pixel_avg_err2 = 0;
		float error_rate2 = TestRoutine_Batch_MiddleburyExtended(pixel_avg_err2, sigma, DT_REFINED); //Para_TestRoutine_Batch_MiddleburyExtended(para, pixel_avg_err2, sigma, OLT_REFINED); //先打开
		float error_rate1 = TestRoutine_Batch_MiddleburyStandard(pixel_avg_err1, sigma, DT_REFINED); //TestRoutine_Batch_MiddleburyStandard2(para, pixel_avg_err1, sigma, OLT_REFINED); //后追加

		float error_rate = (4 * error_rate1 + 23 * error_rate2) / 27;
		float pixel_error = (4 * pixel_avg_err1 + 23 * pixel_avg_err2) / 27;
		//printf("Avg. non-occ err in 4+7 pairs: %.02f%%\n", error_rate);
		fout << "err_ave of std 4 images:rate and error " << error_rate1 << "   " << pixel_avg_err1 << endl;
		fout << "err_ave of ext 23 images:rate and error " << error_rate2 << "   " << pixel_avg_err2 << endl;

		fout << "std+ext error rate:" << error_rate << endl;
		fout << "std+ext pixel_error:" << pixel_error << endl;

	}

	fout.close();
	return 0;
}


//----------------------------------------- data2 end!

char Data3_trainingQ[120][256] = {
	"Data3_trainingQ/Adirondack/disp0GT.pfm","Data3_trainingQ/Adirondack/im0.png", "Data3_trainingQ/Adirondack/im1.png","Data3_trainingQ/Adirondack/mask0nocc.png","_Adirondack.pfm", "_Adirondack_falseColor.png","_Adirondack_scale.png", "_Adirondack_bad.png",
	"Data3_trainingQ/ArtL/disp0GT.pfm","Data3_trainingQ/ArtL/im0.png", "Data3_trainingQ/ArtL/im1.png","Data3_trainingQ/ArtL/mask0nocc.png","_ArtL.pfm", "_ArtL_falseColor.png","_ArtL_scale.png","_ArtL_bad.png",
	"Data3_trainingQ/Jadeplant/disp0GT.pfm","Data3_trainingQ/Jadeplant/im0.png", "Data3_trainingQ/Jadeplant/im1.png","Data3_trainingQ/Jadeplant/mask0nocc.png","_Jadeplant.pfm", "_Jadeplant_falseColor.png", "_Jadeplant_scale.png", "_Jadeplant_bad.png",
	"Data3_trainingQ/Motorcycle/disp0GT.pfm","Data3_trainingQ/Motorcycle/im0.png", "Data3_trainingQ/Motorcycle/im1.png","Data3_trainingQ/Motorcycle/mask0nocc.png","_Motorcycle.pfm", "_Motorcycle_falseColor.png", "_Motorcycle_scale.png", "_Motorcycle_bad.png",
	"Data3_trainingQ/MotorcycleE/disp0GT.pfm","Data3_trainingQ/MotorcycleE/im0.png", "Data3_trainingQ/MotorcycleE/im1.png","Data3_trainingQ/MotorcycleE/mask0nocc.png","_MotorcycleE.pfm", "_MotorcycleE_falseColor.png", "_MotorcycleE_scale.png", "_MotorcycleE_bad.png",
	"Data3_trainingQ/Piano/disp0GT.pfm","Data3_trainingQ/Piano/im0.png", "Data3_trainingQ/Piano/im1.png","Data3_trainingQ/Piano/mask0nocc.png","_Piano.pfm", "_Piano_falseColor.png", "_Piano_scale.png", "_Piano_bad.png",
	"Data3_trainingQ/PianoL/disp0GT.pfm","Data3_trainingQ/PianoL/im0.png", "Data3_trainingQ/PianoL/im1.png","Data3_trainingQ/PianoL/mask0nocc.png","_PianoL.pfm", "_PianoL_falseColor.png", "_PianoL_scale.png", "_PianoL_bad.png",
	"Data3_trainingQ/Pipes/disp0GT.pfm","Data3_trainingQ/Pipes/im0.png", "Data3_trainingQ/Pipes/im1.png","Data3_trainingQ/Pipes/mask0nocc.png","_Pipes.pfm", "_Pipes_falseColor.png", "_Pipes_scale.png", "_Pipes_bad.png",
	"Data3_trainingQ/Playroom/disp0GT.pfm","Data3_trainingQ/Playroom/im0.png", "Data3_trainingQ/Playroom/im1.png","Data3_trainingQ/Playroom/mask0nocc.png","_Playroom.pfm", "_Playroom_falseColor.png", "_Playroom_scale.png", "_Playroom_bad.png",
	"Data3_trainingQ/Playtable/disp0GT.pfm","Data3_trainingQ/Playtable/im0.png", "Data3_trainingQ/Playtable/im1.png","Data3_trainingQ/Playtable/mask0nocc.png","_Playtable.pfm", "_Playtable_falseColor.png", "_Playtable_scale.png", "_Playtable_bad.png",
	"Data3_trainingQ/PlaytableP/disp0GT.pfm","Data3_trainingQ/PlaytableP/im0.png", "Data3_trainingQ/PlaytableP/im1.png","Data3_trainingQ/PlaytableP/mask0nocc.png","_PlaytableP.pfm", "_PlaytableP_falseColor.png", "_PlaytableP_scale.png", "_PlaytableP_bad.png",
	"Data3_trainingQ/Recycle/disp0GT.pfm","Data3_trainingQ/Recycle/im0.png", "Data3_trainingQ/Recycle/im1.png","Data3_trainingQ/Recycle/mask0nocc.png","_Recycle.pfm", "_Recycle_falseColor.png", "_Recycle_scale.png", "_Recycle_bad.png",
	"Data3_trainingQ/Shelves/disp0GT.pfm","Data3_trainingQ/Shelves/im0.png", "Data3_trainingQ/Shelves/im1.png","Data3_trainingQ/Shelves/mask0nocc.png","_Shelves.pfm", "_Shelves_falseColor.png", "_Shelves_scale.png", "_Shelves_bad.png",
	"Data3_trainingQ/Teddy/disp0GT.pfm","Data3_trainingQ/Teddy/im0.png", "Data3_trainingQ/Teddy/im1.png","Data3_trainingQ/Teddy/mask0nocc.png","_Teddy.pfm", "_Teddy_falseColor.png", "_Teddy_scale.png", "_Teddy_bad.png",
	"Data3_trainingQ/Vintage/disp0GT.pfm","Data3_trainingQ/Vintage/im0.png", "Data3_trainingQ/Vintage/im1.png","Data3_trainingQ/Vintage/mask0nocc.png","_Vintage.pfm", "_Vintag_falseColor.png", "_Vintage_scale.png", "_Vintag_bad.png"
};

char Data3_trainingQ_imgs[15][256] = {
	"Adirondack", "ArtL", "Jadeplant", "Motorcycle", "MotorcycleE","Piano", "PianoL", "Pipes","Playroom","Playtable", "PlaytableP", "Recycle", "Shelves","Teddy", "Vintage"
};

// 72.5000   64.0000  160.0000   70.0000   70.0000   65.0000   65.0000   75.0000   82.5000   72.5000   72.5000   65.0000   60.0000   64.0000  190.0000
//int Data3_trainingQ_maxdisp[15] = { 73+1,64+1,160+1,70+1,70+1,65+1,65+1,75+1,83+1,73+1,73+1,65+1,60+1,64+1,190+1 };
int Data3_trainingQ_maxdisp[15] = { 73,64,160,70,70,65,65,75,83,73,73,65,60,64,190 }; //差别不大
//尺度 scales=255/maxdisp=[3.5068    4.0000    1.6000    3.6571    3.6571    3.9385    3.9385    3.4133    3.0843    3.5068    3.5068    3.9385    4.2667    4.0000    1.3474]
int scales[15] = { 4,        4,        1,        4,        4,        4,        4,        3,        3,        3,        3,        4,        4,        4,         2 };


char data3__training_resultDir[50] = "_Data3_TrainingResults/";

float Para_TestRoutine_Batch_MiddleburyData3_trainingQ(float para, float sigma, METHOD method) {

	int scale = 1;
	float ratio_err_ave = 0;
	float ratio_err_ave2 = 0;
	float pixel_err = 0, pixel_err_ave = 0, pixel_err_ave2 = 0;
	double time_ave = 0;

	char resultfile[256] = "\0";
	strcat(resultfile, data3__training_resultDir);
	strcat(resultfile, "data3_training_err.txt");
	ofstream fout(resultfile, ios::out | ios::app);

	//ofstream fout("sigma of mst aggregation_data3_mul_non_occ_err.txt", ios::out | ios::app);
	//fout << "para=" << para << endl;

	//int numImg = 8;
	//int set[15] = {0,1,2,4,5,7,8,9};

	int numImg = 15;
	int set[15] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 };

	float totaltime = 0;
	for (int k = 0; k < numImg; k++)
	{
		int i = set[k];
		int idx = i * 8;
		char* filename_gt_imge = Data3_trainingQ[idx];
		char* filename_left_image = Data3_trainingQ[idx + 1];
		char* filename_right_image = Data3_trainingQ[idx + 2];
		char* filename_mask0nocc = Data3_trainingQ[idx + 3];
		//char* filename_disparity_map = Data3_trainingQ[idx + 4];
		//char* filename_falseColor_image = Data3_trainingQ[idx + 5];
		//char* filename_disparity_scaleimage = Data3_trainingQ[idx + 6];
		//char* filename_disparity_badimage = Data3_trainingQ[idx + 7];

		char filename_disparity_map[256]; strcpy(filename_disparity_map, data3__training_resultDir); strcat(filename_disparity_map, Data3_trainingQ[idx + 4]);
		char filename_falseColor_image[256]; strcpy(filename_falseColor_image, data3__training_resultDir); strcat(filename_falseColor_image, Data3_trainingQ[idx + 5]);
		char filename_disparity_scaleimage[256]; strcpy(filename_disparity_scaleimage, data3__training_resultDir); strcat(filename_disparity_scaleimage, Data3_trainingQ[idx + 6]);
		char filename_disparity_badimage[256]; strcpy(filename_disparity_badimage, data3__training_resultDir); strcat(filename_disparity_badimage, Data3_trainingQ[idx + 7]);

		int max_disparity = Data3_trainingQ_maxdisp[i];


		Mat lImg = imread(filename_left_image, CV_LOAD_IMAGE_COLOR);
		Mat rImg = imread(filename_right_image, CV_LOAD_IMAGE_COLOR);
		if (!lImg.data || !rImg.data) {
			printf("Error: can not open image\n");
			printf("\nPress any key to continue...\n");
			getchar();
			return -1;
		}

		cv::Mat dispImg;
		double duration = 0;
		duration = static_cast<double>(getTickCount());
		stereo_routine(filename_left_image, filename_right_image, filename_disparity_map, max_disparity, scale, sigma, method); //聚合滤波
		duration = static_cast<double>(getTickCount()) - duration;
		duration /= cv::getTickFrequency(); // the elapsed time in sec
		totaltime += duration;
		//float err_non_occ = TestResult_OnePixel_NonOcc(filename_disparity_map, gt_name_l, gt_name_r, scale, filename_bad_image);

		float err_non_occ = TestResult_OnePixel_Val_Data3SDK(filename_disparity_map, filename_gt_imge, filename_mask0nocc, max_disparity, 1.0, pixel_err, filename_falseColor_image);
		TestResult_OnePixel_val_Data3Scale(filename_disparity_map, filename_gt_imge, filename_mask0nocc, max_disparity, 1.5, pixel_err, filename_disparity_scaleimage, filename_disparity_badimage);
		//printf("  %s: %.02f%%\n", extendNames[i], 100 * err_non_occ);
		fout << Data3_trainingQ_imgs[i] << "   " << 100 * err_non_occ << "   " << pixel_err << endl;
		ratio_err_ave += err_non_occ * 100;
		pixel_err_ave += pixel_err;
		if (i != 6 && i != 8 && i != 9 && i != 12 && i != 14) {
			ratio_err_ave2 += err_non_occ * 100;
			pixel_err_ave2 += pixel_err;
		}
		cout << "images  " << i + 1 << "  is over!" << endl;

	}

	//加权平均：ErrorRateMean = (sum(a([1,2,3,4,5,6,8,11,12,14])) + 0.5*sum(a([7,9,10,13,15])))/12.5
	ratio_err_ave2 = (ratio_err_ave2 + 0.5*(ratio_err_ave - ratio_err_ave2)) / 12.5;
	pixel_err_ave2 = (pixel_err_ave2 + 0.5*(pixel_err_ave - pixel_err_ave2)) / 12.5;

	ratio_err_ave /= numImg;
	pixel_err_ave /= numImg;
	totaltime /= numImg;

	fout << "err_ave of all 15 images: ratio_err_ave=" << ratio_err_ave << "%, pixel_err_ave=" << pixel_err_ave << endl;
	fout << "weighted err_ave of all 15 images: : ratio_err_ave=" << ratio_err_ave2 << "%, pixel_err_ave=" << pixel_err_ave2 << endl;
	fout << "avg_time" << totaltime << endl;

	fout << endl << endl << endl;

	fout.close();
	return ratio_err_ave;

}

float TestRoutine_Batch_MiddleburyData3_trainingQ_submit(float sigma, METHOD method) {

	int scale = 1;
	float ratio_err_ave = 0;
	float ratio_err_ave2 = 0;
	float pixel_err = 0, pixel_err_ave = 0, pixel_err_ave2 = 0;
	double time_ave = 0;
	int numImg = 15;

	string folderPath = "Data3_trainingQ\\trainingQ";  //"Data3_trainingQ\\_results";
	ofstream fout(folderPath + "/data3_mul_non_occ_err.txt", ios::out | ios::app);


	for (int i = 0; i < numImg; i++) //numImg
	{
		int idx = i * 8;
		char* filename_gt_imge = Data3_trainingQ[idx];
		char* filename_left_image = Data3_trainingQ[idx + 1];
		char* filename_right_image = Data3_trainingQ[idx + 2];
		char* filename_mask0nocc = Data3_trainingQ[idx + 3];
		//char* filename_disparity_map = Data3_trainingQ[idx + 4];
		char* filename_bad_image = Data3_trainingQ[idx + 5];
		int max_disparity = Data3_trainingQ_maxdisp[i];


		Mat lImg = imread(filename_left_image, CV_LOAD_IMAGE_COLOR);
		Mat rImg = imread(filename_right_image, CV_LOAD_IMAGE_COLOR);
		if (!lImg.data || !rImg.data) {
			printf("Error: can not open image\n");
			printf("\nPress any key to continue...\n");
			getchar();
			return -1;
		}

		//创建文件夹，和文件，按照MiddD3的格式		
		string stereoPath = folderPath + "/" + Data3_trainingQ_imgs[i];
		bool flag = CreateDirectory(stereoPath.c_str(), NULL);


		string disp_file = stereoPath + "/disp0DT.pfm"; //Criss-Cross
		const char* filename_disparity_map = disp_file.data();
		ofstream timefout(stereoPath + "/timeDT.txt", ios::out);

		cv::Mat dispImg;
		double duration;
		duration = static_cast<double>(getTickCount());
		stereo_routine(filename_left_image, filename_right_image, filename_disparity_map, max_disparity, scale, sigma, method); //聚合滤波
		duration = static_cast<double>(getTickCount()) - duration;
		duration /= cv::getTickFrequency(); // the elapsed time in sec
		timefout << duration << endl;
		timefout.close();
		time_ave += duration;

		float err_non_occ = TestResult_OnePixel_Val_Data3SDK(filename_disparity_map, filename_gt_imge, filename_mask0nocc, max_disparity, 1.0, pixel_err, filename_bad_image);
		//printf("  %s: %.02f%%\n", extendNames[i], 100 * err_non_occ);
		fout << Data3_trainingQ_imgs[i] << "   " << 100 * err_non_occ << "   " << pixel_err << "    time=" << duration << endl;
		ratio_err_ave += err_non_occ * 100;
		pixel_err_ave += pixel_err;
		if (i != 2 && i != 6 && i != 9 && i != 12 && i != 14) {
			ratio_err_ave2 += err_non_occ * 100;
			pixel_err_ave2 += pixel_err;
		}
		cout << "images  " << i + 1 << "  is over!" << endl;

	}


	//加权平均：ErrorRateMean = (sum(a([1,2,3,4,5,6,8,11,12,14])) + 0.5*sum(a([7,9,10,13,15])))/12.5
	ratio_err_ave2 = (ratio_err_ave2 + 0.5*(ratio_err_ave - ratio_err_ave2)) / 12.5;
	pixel_err_ave2 = (pixel_err_ave2 + 0.5*(pixel_err_ave - pixel_err_ave2)) / 12.5;

	ratio_err_ave /= numImg;
	pixel_err_ave /= numImg;
	time_ave /= numImg;

	fout << "err_ave of all 15 images: ratio_err_ave=" << ratio_err_ave << "%, pixel_err_ave=" << pixel_err_ave << endl;
	fout << "weighted err_ave of all 15 images: : ratio_err_ave=" << ratio_err_ave2 << "%, pixel_err_ave=" << pixel_err_ave2 << endl;
	fout << "avg_time" << time_ave << endl;

	fout << endl << endl << endl;

	fout.close();
	return ratio_err_ave;

}

int main_training_submit(int argc, char** argv)  //main_training_submit
{

	printf("GF Cost Aggregation on 15 image stereo pairs of Data 3\n");

	float sigma = 0.03;
	float error3 = TestRoutine_Batch_MiddleburyData3_trainingQ_submit(sigma, DT_REFINED);
	return 0;
}


int main_training(int argc, char** argv) //main_training
{
	printf("GF Cost Aggregation on 15 image stereo pairs of Data 3\n");

	float sigma = 0.03;
	float para = 0.01;
	//for (sigma = 0.01; sigma < 0.08; sigma += 0.01)  //OLT_REFINED,OLT_RAW
	{
		float error3 = Para_TestRoutine_Batch_MiddleburyData3_trainingQ(para, sigma, DT_REFINED);  //Para_TestRoutine_Batch_MiddleburyData3_trainingQ(para, sigma, ST_REFINED);
	}
	return 0;
}

//////////////////////////////////testQ
char Data3_testQ[75][256] = {
	"Data3_testQ/Australia/im0.png", "Data3_testQ/Australia/im1.png","_Australia.pfm", "_Australia_falseColor.png", "_Australia_scale.png",
	"Data3_testQ/AustraliaP/im0.png", "Data3_testQ/AustraliaP/im1.png","_AustraliaP.pfm", "_AustraliaP_falseColor.png", "_AustraliaP_scale.png",
	"Data3_testQ/Bicycle2/im0.png", "Data3_testQ/Bicycle2/im1.png","_Bicycle2.pfm", "_Bicycle2_falseColor.png", "_Bicycle2_scale.png",
	"Data3_testQ/Classroom2/im0.png", "Data3_testQ/Classroom2/im1.png","_Classroom2.pfm", "_Classroom2_falseColor.png", "_Classroom2_scale.png",
	"Data3_testQ/Classroom2E/im0.png", "Data3_testQ/Classroom2E/im1.png","_Classroom2E.pfm", "_Classroom2E_falseColor.png", "_Classroom2E_scale.png",
	"Data3_testQ/Computer/im0.png", "Data3_testQ/Computer/im1.png","_Computer.pfm", "_Computer_falseColor.png",  "_Computer_scale.png",
	"Data3_testQ/Crusade/im0.png", "Data3_testQ/Crusade/im1.png","_Crusade.pfm", "_Crusade_falseColor.png", "_Crusade_scale.png",
	"Data3_testQ/CrusadeP/im0.png", "Data3_testQ/CrusadeP/im1.png","_CrusadeP.pfm", "_CrusadeP_falseColor.png", "_CrusadeP_scale.png",
	"Data3_testQ/Djembe/im0.png", "Data3_testQ/Djembe/im1.png","_Djembe.pfm", "_Djembe_falseColor.png", "_Djembe_scale.png",
	"Data3_testQ/DjembeL/im0.png", "Data3_testQ/DjembeL/im1.png","_DjembeL.pfm", "_DjembeL_falseColor.png", "_DjembeL_scale.png",
	"Data3_testQ/Hoops/im0.png", "Data3_testQ/Hoops/im1.png","_Hoops.pfm", "_Hoops_falseColor.png", "_Hoops_scale.png",
	"Data3_testQ/Livingroom/im0.png", "Data3_testQ/Livingroom/im1.png","_Livingroom.pfm", "_Livingroom_falseColor.png", "_Livingroom_scale.png",
	"Data3_testQ/Newkuba/im0.png", "Data3_testQ/Newkuba/im1.png","_Newkuba.pfm", "_Newkuba_falseColor.png", "_Newkuba_scale.png",
	"Data3_testQ/Plants/im0.png", "Data3_testQ/Plants/im1.png","_Plants.pfm", "_Plants_falseColr.png",  "_Plants_scale.png",
	"Data3_testQ/Staircase/im0.png", "Data3_testQ/Staircase/im1.png","_Staircase.pfm", "_Staircase_falseColor.png", "_Staircase_scale.png"
};

char Data3_testQ_imgs[15][256] = {
	"Australia", "AustraliaP", "Bicycle2", "Classroom2", "Classroom2E","Computer", "Crusade", "CrusadeP","Djembe","DjembeL", "Hoops", "Livingroom", "Newkuba","Plants", "Staircase"
};

//ndisp = [290, 290, 250, 610, 610, 256, 800, 800, 320, 320, 410, 320, 570, 320, 450];
int Data3_testQ_maxdisp[15] = { 73,    73,    63,   153,   153,    64,   200,   200,    80,    80,   103,    80,   143,    80,   113 };

char data3__test_resultDir[50] = "_Data3_testResults/";

//用于保存pfm视差图和png视差图
float TestRoutine_Batch_MiddleburyData3_testQ(float sigma, METHOD method) {

	int scale = 1;
	float ratio_err_ave = 0;
	float ratio_err_ave2 = 0;
	float pixel_err = 0, pixel_err_ave = 0, pixel_err_ave2 = 0;
	double time_ave = 0;
	int numImg = 15;

	char resultfile[256] = "\0";
	strcat(resultfile, data3__test_resultDir);
	strcat(resultfile, "data3_test_err.txt");
	ofstream fout(resultfile, ios::out | ios::app);


	for (int i = 0; i < numImg; i++) //numImg
	{
		int idx = i * 5;
		char* filename_left_image = Data3_testQ[idx + 0];
		char* filename_right_image = Data3_testQ[idx + 1];
		//char* filename_disparity_map = Data3_testQ[idx + 2];
		//char* filename_disparity_falseimage = Data3_testQ[idx + 3];
		int max_disparity = Data3_testQ_maxdisp[i];

		char filename_disparity_map[256]; strcpy(filename_disparity_map, data3__test_resultDir); strcat(filename_disparity_map, Data3_testQ[idx + 2]);
		char filename_falseColor_image[256]; strcpy(filename_falseColor_image, data3__test_resultDir); strcat(filename_falseColor_image, Data3_testQ[idx + 3]);
		char filename_disparity_scaleimage[256]; strcpy(filename_disparity_scaleimage, data3__test_resultDir); strcat(filename_disparity_scaleimage, Data3_testQ[idx + 4]);

		Mat lImg = imread(filename_left_image, CV_LOAD_IMAGE_COLOR);
		Mat rImg = imread(filename_right_image, CV_LOAD_IMAGE_COLOR);
		if (!lImg.data || !rImg.data) {
			printf("Error: can not open image\n");
			printf("\nPress any key to continue...\n");
			getchar();
			return -1;
		}

		cv::Mat dispImg;
		double duration;
		duration = static_cast<double>(getTickCount());
		stereo_routine(filename_left_image, filename_right_image, filename_disparity_map, max_disparity, scale, sigma, method); //聚合滤波
		duration = static_cast<double>(getTickCount()) - duration;
		duration /= cv::getTickFrequency(); // the elapsed time in sec
		time_ave += duration;

		//以下将pfm视差图，转换成png视差图
		TestResult_Data3Scale(filename_disparity_map, Data3_testQ_maxdisp[i], filename_falseColor_image, filename_disparity_scaleimage);

		cout << "images  " << i + 1 << "  is over!" << endl;
	}

	time_ave /= numImg;
	fout << time_ave << endl;

	return 0;

}

float TestRoutine_Batch_MiddleburyData3_testQ_submit(float sigma, METHOD method) {

	int scale = 1;
	float ratio_err_ave = 0;
	float ratio_err_ave2 = 0;
	float pixel_err = 0, pixel_err_ave = 0, pixel_err_ave2 = 0;
	double time_ave = 0;
	int numImg = 15;

	string folderPath = "Data3_testQ\\testQ";
	ofstream fout(folderPath + "/data3_testQ_mul_non_occ_err.txt", ios::out | ios::app);


	for (int i = 0; i < numImg; i++) //numImg
	{
		int idx = i * 5;
		char* filename_left_image = Data3_testQ[idx + 0];
		char* filename_right_image = Data3_testQ[idx + 1];
		//char* filename_disparity_map = Data3_testQ[idx + 2];
		//char* filename_bad_image = Data3_testQ[idx + 3];
		int max_disparity = Data3_testQ_maxdisp[i];


		Mat lImg = imread(filename_left_image, CV_LOAD_IMAGE_COLOR);
		Mat rImg = imread(filename_right_image, CV_LOAD_IMAGE_COLOR);
		if (!lImg.data || !rImg.data) {
			printf("Error: can not open image\n");
			printf("\nPress any key to continue...\n");
			getchar();
			return -1;
		}

		//创建文件夹，和文件，按照MiddD3的格式

		string stereoPath = folderPath + "/" + Data3_testQ_imgs[i];
		bool flag = CreateDirectory(stereoPath.c_str(), NULL);


		string disp_file = stereoPath + "/disp0DT.pfm";
		const char* filename_disparity_map = disp_file.data();
		ofstream timefout(stereoPath + "/timeDT.txt", ios::out);

		cv::Mat dispImg;
		double duration;
		duration = static_cast<double>(getTickCount());
		stereo_routine(filename_left_image, filename_right_image, filename_disparity_map, max_disparity, scale, sigma, method); //聚合滤波
		duration = static_cast<double>(getTickCount()) - duration;
		duration /= cv::getTickFrequency(); // the elapsed time in sec

		time_ave += duration;

		timefout << duration << endl;
		timefout.close();

	}

	time_ave /= numImg;
	fout << time_ave << endl;

	return 0;

}



int main_test(int argc, char** argv)  // main_test
{

	printf("GF Cost Aggregation on 15 image stereo pairs of Data 3 testQ\n");
	float sigma = 0.03;

	float error3 = TestRoutine_Batch_MiddleburyData3_testQ(sigma, DT_REFINED); //OLT_REFINED, OLT_RAW
	return 0;
}


int main_test_submit(int argc, char** argv)    // main_test_submit
{

	printf("GF Cost Aggregation on 15 image stereo pairs of Data 3 testQ\n");
	float sigma = 0.03;

	float error3 = TestRoutine_Batch_MiddleburyData3_testQ_submit(sigma, DT_REFINED); //OLT_REFINED, OLT_RAW
	return 0;
}

/////////////////////////////////testQ


























int main_test_2(int argc, char** argv)
{
	printf("Test stereo pairs of Data 3\n");

	char* filename_left_image = "imgs/Half_right_J32.png";
	char* filename_right_image = "imgs/Half_left_J32.png";
	char* filename_disparity_map = "imgs/Half_J32_disparity.png";
	int max_disparity = 192;
	int scale = 1;

	Mat lImg = imread(filename_left_image, CV_LOAD_IMAGE_COLOR);
	Mat rImg = imread(filename_right_image, CV_LOAD_IMAGE_COLOR);


	float sigma = 0.08;
	stereo_routine(filename_left_image, filename_right_image, filename_disparity_map, max_disparity, scale, sigma, ST_REFINED);

	return 0;
}

////将Data2中的视差图真值变成伪彩色显示
int main_generateFalseColorDismapsOfData2()
{
	int scale = 8;
	int numImg1 = 6;


	for (int i = 0; i < numImg1; i++)
	{
		int idx = i * 6;
		char* gt_name_l = cTestFilePool[idx];
		char* gt_name_r = cTestFilePool[idx + 1];

		int max_disparity_0 = scale * CompDepthRange(gt_name_l, scale);
		Mat ldispMap = imread(gt_name_l, cv::IMREAD_GRAYSCALE);

		cv::Mat falseDispairty = Char2ColorJet(ldispMap, 0, max_disparity_0);

		cv::imwrite(gt_name_l, falseDispairty);

	}


	scale = 3;
	int numImg = 33;
	for (int i = numImg1; i < numImg; i++)
	{
		int idx = i * 6;
		char* gt_name_l = cTestFilePool[idx];
		char* gt_name_r = cTestFilePool[idx + 1];

		int max_disparity_0 = scale * CompDepthRange(gt_name_l, scale);
		Mat ldispMap = imread(gt_name_l, cv::IMREAD_GRAYSCALE);

		cv::Mat falseDispairty = Char2ColorJet(ldispMap, 0, max_disparity_0);

		cv::imwrite(gt_name_l, falseDispairty);

	}

	return 0;
}

//将Data3中的视差真值float类型转换成png类型(四分之一分辨率)
int main_gtd3()  // 
{
	int numImg = 15;
	for (int i = 0; i < numImg; i++) {
		int idx = i * 6;
		char *fgtdispname = Data3_trainingQ[idx]; // float视差图
		char *Bytedispname = Data3_trainingQ[idx + 5];  // png视差图

		int verbose = 0;
		CFloatImage fgtdisp;
		ReadImageVerb(fgtdisp, fgtdispname, verbose);

		CShape sh = fgtdisp.Shape();

		int width = sh.width, height = sh.height;

		cv::Mat Bytedisp = cv::Mat::zeros(height, width, CV_8U);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float gt = fgtdisp.Pixel(x, y, 0);
				if (gt == INFINITY) // unknown
					Bytedisp.at<uchar>(y, x) = 0;
				else
					Bytedisp.at<uchar>(y, x) = (uchar)(fgtdisp.Pixel(x, y, 0))*scales[i]; //(uchar)((fgtdisp.Pixel(x, y, 0))*(255.0 / Data3_trainingQ_maxdisp[i]));

			}
		}

		cv::imwrite(Bytedispname, Bytedisp);
	}

	return 0;
}

//将Data3中的视差真值float类型转换成伪彩色png类型(四分之一分辨率)
int main_gtd3tofalsecolor()  //
{
	int numImg = 15;
	for (int i = 0; i < numImg; i++) {
		int idx = i * 6;
		char *fgtdispname = Data3_trainingQ[idx]; // float视差图
		char *Bytedispname = Data3_trainingQ[idx + 5];  // png视差图

		int verbose = 0;
		CFloatImage fgtdisp;
		ReadImageVerb(fgtdisp, fgtdispname, verbose);

		CShape sh = fgtdisp.Shape();

		int width = sh.width, height = sh.height;

		cv::Mat falseDispairty;
		falseDispairty = float2color(fgtdisp, 0, Data3_trainingQ_maxdisp[i]);
		cv::imwrite(Bytedispname, falseDispairty);

	}
	return 0;
}

/*
cam0 = [1500 0 199; 0 1500 187; 0 0 1]
cam1 = [1500 0 251; 0 1500 187; 0 0 1]
doffs = 52
baseline = 80
width = 450
height = 375
ndisp = 64
isint = 0
vmin = 12
vmax = 55
dyavg = 0
dymax = 0

cam0,1:        camera matrices for the rectified views, in the form [f 0 cx; 0 f cy; 0 0 1], where
  f:           focal length in pixels
  cx, cy:      principal point  (note that cx differs between view 0 and 1)

doffs:         x-difference of principal points, doffs = cx1 - cx0

baseline:      camera baseline in mm

width, height: image size

ndisp:         a conservative bound on the number of disparity levels;
			   the stereo algorithm MAY utilize this bound and search from d = 0 .. ndisp-1

isint:         whether the GT disparites only have integer precision (true for the older datasets;
			   in this case submitted floating-point disparities are rounded to ints before evaluating)

vmin, vmax:    a tight bound on minimum and maximum disparities, used for color visualization;
			   the stereo algorithm MAY NOT utilize this information

dyavg, dymax:  average and maximum absolute y-disparities, providing an indication of
			   the calibration error present in the imperfect datasets.
To convert from the floating-point disparity value d [pixels] in the .pfm file to depth Z [mm] the following equation can be used:
Z = baseline * f / (d + doffs)
*/

int main_Depth_of_teddy() // main_Depth_of_teddy()
{
	int ndisp = 64;    //按照 Middlebury Dataset2 中的 Standard data 设置 ndisp = 60
	int scale_std = 4; //按照 Middlebury Dataset2 中的 Standard data 设置 scale = 4
	double scale_disp = (double)(255.0 / (double)ndisp); // (int)(255.0 / (double)ndisp);

	double baseline = 80;
	double focal = 1500;
	double min_disp = 12;
	double doffs = 52;

	double max_z = baseline * focal / (min_disp + doffs); //计算最大深度范围
	double scale_depth = (double)(255.0 / (double)max_z);

	int i = 13;
	int idx = i * 8;
	char *fgtdispname = Data3_trainingQ[idx]; // float视差图
	char *Bytedispname = Data3_trainingQ[idx + 5];  // png视差图

	int verbose = 0;
	CFloatImage fgtdisp;
	ReadImageVerb(fgtdisp, fgtdispname, verbose);

	CShape sh = fgtdisp.Shape();

	int width = sh.width, height = sh.height;

	cv::Mat ScaleGTDispairty(height, width, CV_8U);
	cv::Mat ScaleGTDepth(height, width, CV_8U);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float d = fgtdisp.Pixel(x, y, 0);
			ScaleGTDispairty.at<uchar>(y, x) = (uchar)(d * scale_disp); //(uchar)(f * scale+0.5);  //放大4倍显示

			float z = baseline * focal / (d + doffs);
			ScaleGTDepth.at<uchar>(y, x) = (uchar)(z * scale_depth);

			//生成深度
		}
	}

	cv::imwrite("ScaleGTDisparity_Teddy.png", ScaleGTDispairty);
	cv::imwrite("ScaleGTDepth_Teddy.png", ScaleGTDepth);

	return 0;
}