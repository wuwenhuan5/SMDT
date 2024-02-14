/*
This project implements the methods in the following paper. Please cite this paper, depending on the use.

Xing Mei, Xun Sun, Weiming Dong, Haitao Wang and Xiaopeng Zhang. Segment-Tree based Cost Aggregation for Stereo Matching, in CVPR 2013.

The code is written by Yan Kong, <kongyanwork@gmail.com>, 2013.

LICENSE
Copyright (C) 2012-2013 by Yan Kong
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../imageLib/imageLib.h"
//#include "config.h"

//四对标准图像生成的错误视差图
static void Generate_error_map(const char* result_name, const char* gt_name, const char* mask_all, const char* mask_nonocc, const char* mask_disc, int scale, const char* error_map = NULL)
{
	cv::Mat img_result = cv::imread(result_name, cv::IMREAD_GRAYSCALE);
	cv::Mat img_groundTruth = cv::imread(gt_name, cv::IMREAD_GRAYSCALE);
	cv::Mat img_mask_all = cv::imread(mask_all, cv::IMREAD_GRAYSCALE);
	cv::Mat img_mask_nonocc = cv::imread(mask_nonocc, cv::IMREAD_GRAYSCALE);
	cv::Mat img_mask_disc = cv::imread(mask_disc, cv::IMREAD_GRAYSCALE);

	cv::Size size = img_result.size();
	cv::Mat img_bad = cv::Mat::ones(size, CV_8U);
	img_bad = 255 * img_bad;

	cv::Mat1b maskPtr_all = img_mask_all;
	cv::Mat1b maskPtr_nonocc = img_mask_nonocc;
	cv::Mat1b maskPtr_disc = img_mask_disc;

	int val_result, val_g;
	for (int y = 0; y < size.height; y++) {
		for (int x = 0; x < size.width; x++) {
			/*
			if (maskPtr_all(y, x) == 255) {
				val_result = img_result.at<uchar>(y, x);
				val_g = img_groundTruth.at<uchar>(y, x);
				if (abs(val_result - val_g) >(scale)) {
					img_bad.at<uchar>(y, x) = 0;					
				}
			}*/
			if (maskPtr_nonocc(y, x) == 255) {
				val_result = img_result.at<uchar>(y, x);
				val_g = img_groundTruth.at<uchar>(y, x);
				if (abs(val_result - val_g) >(scale)) {
					img_bad.at<uchar>(y, x) = 0;
				}
			}
			if (maskPtr_disc(y, x) == 255) {
				val_result = img_result.at<uchar>(y, x);
				val_g = img_groundTruth.at<uchar>(y, x);
				if (abs(val_result - val_g) >(scale)) {
					img_bad.at<uchar>(y, x) = 0;
				}
			}
		}
	}

	if (error_map) {
		cv::imwrite(error_map, img_bad);
	}

}

//用于四对标准图像
static float TestResult_OnePixel_Val(const char* result_name, const char* gt_name, const char* mask_name, int scale,float&pixel_err, const char* result_bad = NULL) {
	cv::Mat img_result = cv::imread(result_name, cv::IMREAD_GRAYSCALE);
	cv::Mat img_groundTruth = cv::imread(gt_name, cv::IMREAD_GRAYSCALE);
	cv::Mat img_mask = cv::imread(mask_name, cv::IMREAD_GRAYSCALE);

	cv::Size size = img_result.size();
	cv::Mat img_bad;
	cv::cvtColor(img_result, img_bad, CV_GRAY2BGR);

	cv::Mat1b maskPtr = img_mask;
	int val_result, val_g;
	int num_err = 0, num_all = 0;
	pixel_err = 0;
	for(int y = 0;y < size.height;y++) {
		for(int x = 0;x < size.width;x++) {
			if(maskPtr(y, x) == 255) {
				val_result = img_result.at<uchar>(y, x);
				val_g = img_groundTruth.at<uchar>(y, x);

				if(abs(val_result - val_g) > (scale)) {
					img_bad.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
					num_err++;
				}
				pixel_err+=fabs(val_result - val_g)/scale;
				num_all++;
			}
		}
	}

	if(result_bad) {
		cv::imwrite(result_bad, img_bad);
	}
	pixel_err /=  num_all;
	return float(num_err) / num_all;
}

//用于Data2中的立体对，估计视差范围
static int CompDepthRange(const char* gt_name, int scale) {
	int max_disparity = 0;
	cv::Mat img_groundTruth = cv::imread(gt_name, cv::IMREAD_GRAYSCALE);
	cv::Size size = img_groundTruth.size();

	for (int y = 0; y < size.height; y++) {
		for (int x = 0; x < size.width; x++) {
			int val = img_groundTruth.at<uchar>(y, x);
			val /= scale;
			max_disparity = MAX(max_disparity, val);
		}
	}

	return max_disparity;
}

//用于Data2中的立体对，估计错误率
static float TestResult_OnePixel_NonOcc(const char* result_name, const char* gt_name_l, const char* gt_name_r, int scale, float &pixel_err, const char* result_bad = NULL) {
	cv::Mat img_result = cv::imread(result_name, cv::IMREAD_GRAYSCALE);
	cv::Mat img_groundTruthL = cv::imread(gt_name_l, cv::IMREAD_GRAYSCALE);
	cv::Mat img_groundTruthR = cv::imread(gt_name_r, cv::IMREAD_GRAYSCALE);

	cv::Size size = img_result.size();
	cv::Mat img_bad;
	cv::cvtColor(img_result, img_bad, CV_GRAY2BGR);

	pixel_err = 0;

	int val_g, val_result, val_g_cor;
	int num_err = 0, num_all = 0;
	for(int y = 0;y < size.height;y++) {
		for(int x = 0;x < size.width;x++) {
			val_g = img_groundTruthL.at<uchar>(y, x) / scale;
			if(val_g > 0 && x - val_g >= 0) {
				val_g_cor = img_groundTruthR.at<uchar>(y, x - val_g) / scale;
				if(val_g == val_g_cor) { //非遮挡区域
					val_result = img_result.at<uchar>(y, x) / scale;
					pixel_err += fabs(val_result - val_g);
					if(abs(val_result - val_g) > 1) {
						img_bad.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
						num_err++;
					}
					num_all++;
				}
			}
		}
	}

	pixel_err = pixel_err / num_all;
	if(result_bad) {
		cv::imwrite(result_bad, img_bad);
	}

	return float(num_err) / num_all;
}



static void writeFalseColorImage_KITTI(const std::string result_name, const std::string file_name, float max_val) {

	cv::Mat img_result = cv::imread(result_name, cv::IMREAD_GRAYSCALE);

	int height_ = img_result.size().height;
	int width_ = img_result.size().width;

	// color map
	float map[8][4] = { { 0,0,0,114 },{ 0,0,1,185 },{ 1,0,0,114 },{ 1,0,1,174 },
	{ 0,1,0,114 },{ 0,1,1,185 },{ 1,1,0,114 },{ 1,1,1,0 } };
	float sum = 0;
	for (int32_t i = 0; i<8; i++)
		sum += map[i][3];

	float weights[8]; // relative weights
	float cumsum[8];  // cumulative weights
	cumsum[0] = 0;
	for (int32_t i = 0; i<7; i++) {
		weights[i] = sum / map[i][3];
		cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
	}

	// create color png image
	//png::image< png::rgb_pixel > image(width_, height_);
	//cv::Mat falseimage(height_,width_, CV_8UC3);
	cv::Mat falseimage = cv::Mat::zeros(height_, width_, CV_8UC3);
	//cv::cvtColor(img_result, falseimage, CV_GRAY2BGR);

	// for all pixels do
	for (int32_t v = 0; v<height_; v++) {
		for (int32_t u = 0; u<width_; u++) {

			// get normalized value
			float val = min(max(img_result.at<uchar>(v, u) / max_val, 0.0f), 1.0f);

			// find bin
			int32_t i;
			for (i = 0; i<7; i++)
				if (val<cumsum[i + 1])
					break;

			// compute red/green/blue values
			float   w = 1.0 - (val - cumsum[i])*weights[i];
			uchar r = (uchar)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.0);
			uchar g = (uchar)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.0);
			uchar b = (uchar)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.0);

			// set pixel
			falseimage.at<cv::Vec3b>(v, u) = cv::Vec3b(b, g, r);
		}
	}

	// write to file
	cv::imwrite(file_name, falseimage);
}

// translate value x in [0..1] into color triplet using "jet" color map
// if out of range, use darker colors
// variation of an idea by http://www.metastine.com/?p=7
static void jet_(float x, int& r, int& g, int& b)
{
	if (x < 0) x = -0.05;
	if (x > 1) x = 1.05;
	x = x / 1.15 + 0.1; // use slightly asymmetric range to avoid darkest shades of blue.
	r = __max(0, __min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .75))))));
	g = __max(0, __min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .5))))));
	b = __max(0, __min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .25))))));
}

// convert char disparity image into a color image using jet colormap
static cv::Mat Char2ColorJet(cv::Mat &fimg, float dmin, float dmax)
{

	int width = fimg.cols, height = fimg.rows;
	cv::Mat img(height, width, CV_8UC3);

	float scale = 1.0 / (dmax - dmin);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float f = fimg.at<uchar>(y, x);
			int r = 0;
			int g = 0;
			int b = 0;

			if (f != INFINITY) {
				float val = scale * (f - dmin);
				jet_(val, r, g, b);
			}

			img.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
		}
	}

	return img;
}

static cv::Mat float2color(CFloatImage fimg,  float dmin, float dmax)
{
	CShape sh = fimg.Shape();
	int width = sh.width, height = sh.height;
	sh.nBands = 3;

	cv::Mat img(height, width, CV_8UC3);

	float scale = 1.0 / (dmax - dmin);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float f = fimg.Pixel(x, y, 0);

			int r = 0;
			int g = 0;
			int b = 0;

			if (f != INFINITY) {
				float val = scale * (f - dmin);
				jet_(val, r, g, b);
			}

			img.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
		}
	}

	return img;
}


//用于Data3中的立体对， 读取png的视差图结果， 并输出错误视图结果， 视差没有进行放大到255
static float TestResult_OnePixel_val_Data3(const char* result_name, const char* gt_name, const char* mask_name, int maxdisp, float thre, float &pixel_err, const char* result_bad = NULL) { //读取pfm格式真实视差图像文件
	cv::Mat img_result = cv::imread(result_name, cv::IMREAD_GRAYSCALE);
	//cv::Mat img_groundTruth = cv::imread(gt_name, cv::IMREAD_GRAYSCALE);
	CFloatImage gtdisp;  int verbose = 0;
	ReadImageVerb(gtdisp, gt_name, verbose);
	cv::Mat img_mask = cv::imread(mask_name, cv::IMREAD_GRAYSCALE);

	cv::Size size = img_result.size();
	cv::Mat img_bad;
	cv::cvtColor(img_result, img_bad, CV_GRAY2BGR);

	cv::Mat1b maskPtr = img_mask;
	int val_result; float val_g;
	int num_err = 0, num_all = 0;
	float  ratio_err = 0;
	for (int y = 0; y < size.height; y++) {
		for (int x = 0; x < size.width; x++) {
			if (maskPtr(y, x) == 255) {
				//val_g = img_groundTruth.at<uchar>(y, x);
				val_g = gtdisp.Pixel(x, y, 0);
				if (val_g != INFINITY) {
					val_result = img_result.at<uchar>(y, x);
					if (abs(val_result - val_g) > (thre)) {
						img_bad.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
						num_err++;
					}
					num_all++;
				}
			}
		}
	}
	ratio_err = 1.0*(num_err) / num_all;

	pixel_err = 0.0;
	for (int y = 0; y < size.height; y++) {
		for (int x = 0; x < size.width; x++) {
			if (maskPtr(y, x) == 255) {
				//val_g = img_groundTruth.at<uchar>(y, x);
				val_g = gtdisp.Pixel(x, y, 0);
				if (val_g != INFINITY) {
					val_result = img_result.at<uchar>(y, x);
					pixel_err += fabs(val_g - val_result);
				}
			}
		}
	}

	pixel_err /= num_all;

	if (result_bad) {
		cv::imwrite(result_bad, img_bad);

		//使用OpenCV的方法
		//cv::Mat falseimg;
		//applyColorMap(img_result, falseimg, 4); //COLORMAP_RAINBOW
		//cv::imwrite(result_bad, falseimg);

		//使用KITTI的方法
		//writeFalseColorImage_KITTI(result_name, result_bad, maxdisp-1);

		//使用Middlury的方法		
		//cv::Mat falseDispairty;
		//falseDispairty = Char2ColorJet(img_result, 0, maxdisp);
		//cv::imwrite(result_bad, falseDispairty);
	}

	return ratio_err;
}

////用于Data3中的训练集立体对， 读取pfm的视差图结果， 并将视差图放大到255，保存为png图像，而且输出了错误视图结果
static float TestResult_OnePixel_val_Data3Scale(const char* disp_name, const char* gt_name, const char* mask_name, int maxdisp, float thre, float &pixel_err, const char* result_png = NULL, const char* result_bad = NULL) { //读取pfm格式真实视差图像文件
	int verbose = 0;
	CFloatImage pfmdisp;
	CFloatImage gtdisp;
	ReadImageVerb(pfmdisp, disp_name, verbose); //读取pfm
	ReadImageVerb(gtdisp, gt_name, verbose);

	cv::Mat img_mask = cv::imread(mask_name, cv::IMREAD_GRAYSCALE);

	CShape sh = pfmdisp.Shape();
	int width = sh.width, height = sh.height;

	cv::Mat pngdisp(height,width, CV_8U);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			pngdisp.at<uchar>(y,x)= (uchar)(pfmdisp.Pixel(x, y, 0)*255.0/ maxdisp); //将pfm的dispmap转换成png格式, 并进行放大到255
		}
	}
	
	cv::Mat baddisp;
	cv::cvtColor(pngdisp, baddisp, CV_GRAY2BGR);

	cv::Mat1b maskPtr = img_mask;
	int val_result; float val_g;
	int num_err = 0, num_all = 0;
	float  ratio_err = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (maskPtr(y, x) == 255) {
				val_g = gtdisp.Pixel(x, y, 0);
				if (val_g != INFINITY) {
					val_result = pfmdisp.Pixel(x, y, 0);
					if (abs(val_result - val_g) > (thre)) {
						baddisp.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
						num_err++;
					}
					num_all++;
				}
			}
		}
	}

	/*
	ratio_err = 1.0*(num_err) / num_all;
	pixel_err = 0.0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (maskPtr(y, x) == 255) {
				//val_g = img_groundTruth.at<uchar>(y, x);
				val_g = gtdisp.Pixel(x, y, 0);
				if (val_g != INFINITY) {
					val_result = pfmdisp.Pixel(x, y, 0);
					pixel_err += fabs(val_g - val_result);
				}
			}
		}
	}
	pixel_err /= num_all;
	*/

	if (result_bad) {
		cv::imwrite(result_png, pngdisp);
		cv::imwrite(result_bad, baddisp);
	}

	return ratio_err;
}

////用于Data3中的测试集立体对， 读取pfm的视差图结果， 并将视差图放大到255，保存为png图像，而且输出了错误视图结果
static void TestResult_Data3Scale(const char* disp_name,  int maxdisp, const char* result_falseColor = NULL, const char* result_png = NULL) { //读取pfm格式真实视差图像文件
	int verbose = 0;
	CFloatImage pfmdisp;
	CFloatImage gtdisp;
	ReadImageVerb(pfmdisp, disp_name, verbose); //读取pfm

	CShape sh = pfmdisp.Shape();
	int width = sh.width, height = sh.height;

	cv::Mat pngdisp(height, width, CV_8U);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			pngdisp.at<uchar>(y, x) = (uchar)(pfmdisp.Pixel(x, y, 0)*255.0 / maxdisp); //将pfm的dispmap转换成png格式, 并进行放大到255
		}
	}

	if (result_falseColor) {
		cv::Mat falseDispairty;
		falseDispairty = float2color(pfmdisp, 0, maxdisp);
		cv::imwrite(result_falseColor, falseDispairty);
	}
	
	if(result_png)
	   cv::imwrite(result_png, pngdisp);
	

}

//以下是根据明度学院数据集Data3提供的SDK中的评价，重新统计
static float TestResult_OnePixel_Val_Data3SDK(const char* disp_name, const char* gt_name, const char* mask_name, int maxdisp, float badthresh, float &pixel_err, const char* falseColorImg = NULL) { //读取pfm格式真实视差图像文件
	//cv::Mat disp = cv::imread(result_name, cv::IMREAD_GRAYSCALE);
	//cv::Mat img_groundTruth = cv::imread(gt_name, cv::IMREAD_GRAYSCALE);
	int verbose = 0;
	CFloatImage disp;  
	CFloatImage gtdisp;
	ReadImageVerb(disp, disp_name, verbose); //读取pfm
	ReadImageVerb(gtdisp, gt_name, verbose);
	cv::Mat img_mask = cv::imread(mask_name, cv::IMREAD_GRAYSCALE); //CByteImage mask; ReadImageVerb(mask, maskname, verbose); 貌似存在问题
	cv::Mat1b maskPtr = img_mask;
	cv::Size msh = img_mask.size();

	CShape sh = gtdisp.Shape();
	CShape sh2 = disp.Shape();

	int width = sh.width, height = sh.height;
	int width2 = sh2.width, height2 = sh2.height;
	int scale = width / width2;

	int usemask = (msh.width > 0 && msh.height > 0);
	if (usemask && (msh.height!= sh.height)&& (msh.width != sh.width))
		throw CError("mask image must have same size as GT\n");

	int n = 0;
	int bad = 0;
	int invalid = 0;
	float serr = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float gt = gtdisp.Pixel(x, y, 0);
			if (gt == INFINITY) // unknown
				continue;
			float d = scale * disp.Pixel(x / scale, y / scale, 0);
			int valid = (d != INFINITY);
			if (valid) {
				float maxd = scale * maxdisp; // max disp range
				d = __max(0, __min(maxd, d)); // clip disps to max disp range
			}

			int rounddisp = 0;
			if (valid && rounddisp)
				d = round(d);

			float err = fabs(d - gt);
			if (usemask && img_mask.at<uchar>(y,x)!=255) { // don't evaluate pixel, mask.Pixel(x, y, 0) != 255
			}
			else {
				n++;
				if (valid) {
					serr += err;
					if (err > badthresh) {
						bad++;
					}
				}
				else {// invalid (i.e. hole in sparse disp map)
					invalid++;
				}
			}
		}
	}
	float badpercent = 100.0*bad / n;
	float invalidpercent = 100.0*invalid / n;  //由于估计得到的是densemap, 故一般invalid=0, invalidpercent=0
	float totalbadpercent = 1.0*(bad + invalid) / n;
	float avgErr = serr / (n - invalid); // CHANGED 10/14/2014 -- was: serr / n
										 //printf("mask  bad%.1f  invalid  totbad   avgErr\n", badthresh);

	pixel_err = avgErr;

	//printf("mask  bad%.1f  invalid  totbad   avgErr\n", badthresh);
	//printf("%4.1f  %6.2f  %6.2f   %6.2f  %6.2f\n", 100.0*n / (width * height),badpercent, invalidpercent, totalbadpercent, avgErr);

	if (falseColorImg) {
		//cv::imwrite(result_bad, img_bad);
		//cv::Mat falseimg;
		//applyColorMap(img_result, falseimg, 4); //COLORMAP_RAINBOW
		//cv::imwrite(result_bad, falseimg);

		//使用KITTI的方法
		//writeFalseColorImage_KITTI(result_name, result_bad, maxdisp-1);

		//使用Middlury的方法		
		cv::Mat falseDispairty;
		falseDispairty = float2color(disp, 0, maxdisp);
		cv::imwrite(falseColorImg, falseDispairty);

	}

	return totalbadpercent;
}

/*
// get min and max (non-INF) values
void getMinMax(CFloatImage fimg, float& vmin, float& vmax)
{
CShape sh = fimg.Shape();
int width = sh.width, height = sh.height;

vmin = INFINITY;
vmax = -INFINITY;

for (int y = 0; y < height; y++) {
for (int x = 0; x < width; x++) {
float f = fimg.Pixel(x, y, 0);
if (f == INFINITY)
continue;
vmin = min(f, vmin);
vmax = max(f, vmax);
}
}
}
*/

