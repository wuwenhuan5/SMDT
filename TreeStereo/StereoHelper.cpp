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
#include <algorithm> 

#include "StereoHelper.h"
#include "Toolkit.h"
//#include "../MeshStereo/MatchingCost.h"
//#include "../CrossScaleStereo/GrdCC.h"
//#include "../CrossScaleStereo/CenCC.h"
//#include "../CrossScaleStereo/CGCC.h"
//#include "../PatchMatchStereo/Utilities.h"
//#include "../PatchMatchStereo/PatchMatchStereo.h"
//#include "../ADCensusStereo/adcensuscv.h"


inline unsigned char rgb_2_gray(unsigned char * in) { return((unsigned char)(0.299*in[2]+0.587*in[1]+0.114*in[0]+0.5));} //BGR格式

#define CENSUS_H 5
#define CENSUS_W 5
#define CENSUS_BIT (CENSUS_H*CENSUS_W-1)

//-----------------Sobel图像
cv::Mat sobel_grad(const Mat img) //img是rgb图像
{
	CV_Assert(img.type() == CV_8UC3);
	Mat src = img.clone();
	cvtColor(src, src, CV_BGR2RGB);
	// MeanFilter(src, src, 1);
	Mat src_gray;
	//高斯模糊  
	//GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	//转成灰度图  
	cvtColor(src, src_gray, CV_RGB2GRAY);
	Mat grad_xy, grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	//x方向梯度计算  
	Sobel(src_gray, grad_x, ddepth, 1, 0, 5, scale, delta, BORDER_DEFAULT); // 窗口3改成1呢？
	convertScaleAbs(grad_x, abs_grad_x); //an unsigned 8-bit type:


	//y方向梯度计算  
	Sobel(src_gray, grad_y, ddepth, 0, 1, 5, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	//加权和  

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_xy);


	return grad_xy;


}

cv::Mat sobel_gradX(const Mat img) //img是rgb图像
{
	CV_Assert(img.type() == CV_8UC3);
	Mat src = img.clone();
	cvtColor(src, src, CV_BGR2RGB);
	// MeanFilter(src, src, 1);
	Mat src_gray;
	//高斯模糊  
	//GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	//转成灰度图  
	cvtColor(src, src_gray, CV_RGB2GRAY);
	Mat grad_xy, grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	//x方向梯度计算  
	Sobel(src_gray, grad_x, ddepth, 1, 0, 5, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x); //an unsigned 8-bit type:



	return abs_grad_x;
}


cv::Mat sobel_gradY(const Mat img) //img是rgb图像
{
	CV_Assert(img.type() == CV_8UC3);
	Mat src = img.clone();
	cvtColor(src, src, CV_BGR2RGB);
	// MeanFilter(src, src, 1);
	Mat src_gray;
	//高斯模糊  
	//GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	//转成灰度图  
	cvtColor(src, src_gray, CV_RGB2GRAY);
	Mat grad_xy, grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	//y方向梯度计算  
	Sobel(src_gray, grad_y, ddepth, 0, 1, 5, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	//加权和  


	return abs_grad_y;
}



//----------------------------Census变换
cv::Mat CDisparityHelper::GetCensusMatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel)
{
#define CENSUS_H 5
#define CENSUS_W 7
#define CENSUS_BIT (CENSUS_H*CENSUS_W-1)

	int height = imL.rows;//高度
	int width = imL.cols;//宽度


	Mat lGray, rGray;
	if (imL.channels() == 3 && imR.channels() == 3) {
		Mat limg = imL.clone();
		Mat rimg = imR.clone();
		cv::cvtColor(limg, lGray, CV_BGR2GRAY);
		cv::cvtColor(rimg, rGray, CV_BGR2GRAY);
	}
	else if (imL.channels() == 1 && imR.channels() == 1)
	{
		lGray = imL.clone();
		rGray = imR.clone();
	}

	cv::Mat costVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, height, width, maxLevel);

	// prepare binary code 
	//int H_WD = CENSUS_WIND / 2; // 4
	int H_WD = CENSUS_H / 2;
	int W_WD = CENSUS_W / 2;
	bitset<CENSUS_BIT>* lCode = new bitset<CENSUS_BIT>[height * width];
	bitset<CENSUS_BIT>* rCode = new bitset<CENSUS_BIT>[height * width];//每个像素有80bit位序列
	bitset<CENSUS_BIT>* pLCode = lCode;//lCode存着数组首地址
	bitset<CENSUS_BIT>* pRCode = rCode;
	for (int y = 0; y < height; y++) {
		uchar* pLData = (uchar*)(lGray.ptr<uchar>(y));//第y行
		uchar* pRData = (uchar*)(rGray.ptr<uchar>(y));
		for (int x = 0; x < width; x++) { //第x列
			int bitCnt = 0;
			for (int wy = -H_WD; wy <= H_WD; wy++) {
				//int qy = (y + wy + height) % height;//这里+hei是为了实现循环
				int qy = y + wy;
				if (qy < 0) qy = 0;
				if (qy >= height) qy = height - 1;
				uchar* qLData = (uchar*)(lGray.ptr<uchar>(qy));
				uchar* qRData = (uchar*)(rGray.ptr<uchar>(qy));
				for (int wx = -W_WD; wx <= W_WD; wx++) {
					if (wy != 0 || wx != 0) { //中心点
						//int qx = (x + wx + width) % width;
						int qx = x + wx;
						if (qx < 0) qx = 0;
						if (qx >= width) qx = width - 1;
						(*pLCode)[bitCnt] = (pLData[x] > qLData[qx]); // pLData[ x ]是中心点
						(*pRCode)[bitCnt] = (pRData[x] > qRData[qx]); // pRData[ x ]是中心点												
						bitCnt++;
						//0到79
					}
				}
			}

			/*
			//增加上下对角的比较
			int left = max(0, x - 1);
			int right = min(x + 1, width - 1);
			int up = max(0, y - 1);
			int down = min(y + 1, height - 1);

			(*pLCode)[bitCnt] = lGray.at<uchar>(up, left) > lGray.at<uchar>(down, right);
			bitCnt++;
			(*pLCode)[bitCnt] = lGray.at<uchar>(up, x) > lGray.at<uchar>(down, x);
			bitCnt++;
			(*pLCode)[bitCnt] = lGray.at<uchar>(up, right) > lGray.at<uchar>(down, left);
			bitCnt++;
			(*pLCode)[bitCnt] = lGray.at<uchar>(y, left) > lGray.at<uchar>(down, right);
			bitCnt++;
			*/

			pLCode++;
			pRCode++;
		}
	}
	// build cost volume
	bitset<CENSUS_BIT> lB;
	bitset<CENSUS_BIT> rB;
	pLCode = lCode;
	for (int y = 0; y < height; y++) {
		int index = y * width;
		for (int x = 0; x < width; x++) {
			lB = *pLCode;
			for (int d = 0; d < maxLevel; d++) {
				costVolPtr(y, x, d) = CENSUS_BIT + 1; // 
				if (x - d >= 0) {
					rB = rCode[index + x - d];//x-d是右视图中index行的第x-d个像素
					costVolPtr(y, x, d) = (lB ^ rB).count();
				}

			}
			pLCode++;
		}
	}
	delete[] lCode;
	delete[] rCode;

	return costVol;
}


cv::Mat CDisparityHelper::GetRightCensusMatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel) //直接计算右代价体
{
#define CENSUS_H 5
#define CENSUS_W 7
#define CENSUS_BIT (CENSUS_H*CENSUS_W-1)

	int height = imL.rows;//高度
	int width = imL.cols;//宽度


	Mat lGray, rGray;
	if (imL.channels() == 3 && imR.channels() == 3) {
		Mat limg = imL.clone();
		Mat rimg = imR.clone();
		cv::cvtColor(limg, lGray, CV_BGR2GRAY);
		cv::cvtColor(rimg, rGray, CV_BGR2GRAY);
	}
	else if (imL.channels() == 1 && imR.channels() == 1)
	{
		lGray = imL.clone();
		rGray = imR.clone();
	}

	cv::Mat RightcostVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> rcostVolPtr((double *)RightcostVol.data, height, width, maxLevel);

	// prepare binary code 
	//int H_WD = CENSUS_WIND / 2; // 4
	int H_WD = CENSUS_H / 2;
	int W_WD = CENSUS_W / 2;
	bitset<CENSUS_BIT>* lCode = new bitset<CENSUS_BIT>[height * width];
	bitset<CENSUS_BIT>* rCode = new bitset<CENSUS_BIT>[height * width];//每个像素有80bit位序列
	bitset<CENSUS_BIT>* pLCode = lCode;//lCode存着数组首地址
	bitset<CENSUS_BIT>* pRCode = rCode;
	for (int y = 0; y < height; y++) {
		uchar* pLData = (uchar*)(lGray.ptr<uchar>(y));//第y行
		uchar* pRData = (uchar*)(rGray.ptr<uchar>(y));
		for (int x = 0; x < width; x++) { //第x列
			int bitCnt = 0;
			for (int wy = -H_WD; wy <= H_WD; wy++) {
				//int qy = (y + wy + height) % height;//这里+hei是为了实现循环
				int qy = y + wy;
				if (qy < 0) qy = 0;
				if (qy >= height) qy = height - 1;

				uchar* qLData = (uchar*)(lGray.ptr<uchar>(qy));
				uchar* qRData = (uchar*)(rGray.ptr<uchar>(qy));
				for (int wx = -W_WD; wx <= W_WD; wx++) {
					if (wy != 0 || wx != 0) { //中心点
						//int qx = (x + wx + width) % width;
						int qx = x + wx;
						if (qx < 0) qx = 0;
						if (qx >= width) qx = width - 1;

						(*pLCode)[bitCnt] = (pLData[x] > qLData[qx]);// pLData[ x ]是中心点
						(*pRCode)[bitCnt] = (pRData[x] > qRData[qx]);// pRData[ x ]是中心点
						bitCnt++; //0到79
					}

				}
			}

			/*
			//增加上下对角的比较
			int left = max(0, x - 1);
			int right = min(x + 1, width - 1);
			int up = max(0, y - 1);
			int down = min(y + 1, height - 1);
			//
			(*pLCode)[bitCnt] = lGray.at<uchar>(up, left) > lGray.at<uchar>(down, right);
			bitCnt++;
			(*pLCode)[bitCnt] = lGray.at<uchar>(up, x) > lGray.at<uchar>(down, x);
			bitCnt++;
			(*pLCode)[bitCnt] = lGray.at<uchar>(up, right) > lGray.at<uchar>(down, left);
			bitCnt++;
			(*pLCode)[bitCnt] = lGray.at<uchar>(y, left) > lGray.at<uchar>(down, right);
			bitCnt++;
			*/

			pLCode++;
			pRCode++;
		}
	}

	// build cost volume
	bitset<CENSUS_BIT> lB;
	bitset<CENSUS_BIT> rB;
	pRCode = rCode;
	for (int y = 0; y < height; y++) {
		int index = y * width;
		for (int x = 0; x < width; x++) {
			rB = *pRCode;
			for (int d = 0; d < maxLevel; d++) {
				rcostVolPtr(y, x, d) = CENSUS_BIT + 1;
				if (x + d < width) {
					lB = lCode[index + x + d];
					rcostVolPtr(y, x, d) = (rB ^ lB).count();
				}

			}
			pRCode++;
		}
	}
	delete[] lCode;
	delete[] rCode;


	return RightcostVol;
}
//-----------------------------Census结束

// 以left image为参考图像，将gradX的bit string和gradY的bit string拼接起来，然后计算这个拼接后的长bitstring的汉明距离。
cv::Mat CDisparityHelper::Get_GradXGradY_Census_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel)
{
    #define CENSUS_H 5
    #define CENSUS_W 7
    #define CENSUS_BIT (2*CENSUS_H*CENSUS_W-2)  //覆盖前面的CENSUS_BIT定义
	Mat limg = imL.clone();
	Mat rimg = imR.clone();

	int height = limg.rows;
	int width = limg.cols;


	cv::Mat costVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, height, width, maxLevel);


	Mat lSobelX, rSobelX; //左右视图的Sobel_X梯度图像
	lSobelX = sobel_gradX(limg);
	rSobelX = sobel_gradX(rimg);

	Mat lSobelY, rSobelY; //左右视图的Sobel_Y梯度图像
	lSobelY = sobel_gradY(limg);
	rSobelY = sobel_gradY(rimg);

	// prepare binary code 
	//int H_WD = CENSUS_WIND / 2; // 4
	int H_WD = CENSUS_H / 2;
	int W_WD = CENSUS_W / 2;

	int mm = CENSUS_W;
	int nn = CENSUS_BIT;

	bitset<CENSUS_BIT>* lCode = new bitset<CENSUS_BIT>[height * width];
	bitset<CENSUS_BIT>* rCode = new bitset<CENSUS_BIT>[height * width];//每个像素有80bit位序列
	bitset<CENSUS_BIT>* pLCode = lCode;//lCode存着数组首地址
	bitset<CENSUS_BIT>* pRCode = rCode;
	for (int y = 0; y < height; y++) {
		uchar* pLData = (uchar*)(lSobelX.ptr<uchar>(y));//第y行
		uchar* pRData = (uchar*)(rSobelX.ptr<uchar>(y));

		uchar* gpLData = (uchar*)(lSobelY.ptr<uchar>(y));//第y行
		uchar* gpRData = (uchar*)(rSobelY.ptr<uchar>(y));
		for (int x = 0; x < width; x++) { //第x列
			int bitCnt = 0;
			for (int wy = -H_WD; wy <= H_WD; wy++) {
				//int qy = (y + wy + height) % height;//这里+hei是为了实现循环
				int qy = y + wy;
				if (qy < 0) qy = 0;
				if (qy >= height) qy = height - 1;
				uchar* qLData = (uchar*)(lSobelX.ptr<uchar>(qy));
				uchar* qRData = (uchar*)(rSobelX.ptr<uchar>(qy));

				uchar* gqLData = (uchar*)(lSobelY.ptr<uchar>(qy));
				uchar* gqRData = (uchar*)(rSobelY.ptr<uchar>(qy));
				for (int wx = -W_WD; wx <= W_WD; wx++) {
					if (wy != 0 || wx != 0) { //中心点
											  //int qx = (x + wx + width) % width;
						int qx = x + wx;
						if (qx < 0) qx = 0;
						if (qx >= width) qx = width - 1;
						//比较灰度
						(*pLCode)[bitCnt] = (pLData[x] > qLData[qx]);// pLData[ x ]是中心点
						(*pRCode)[bitCnt] = (pRData[x] > qRData[qx]);// pRData[ x ]是中心点	
						bitCnt++;

						//比较梯度，在灰度后面进行拼接
						(*pLCode)[bitCnt] = (gpLData[x] > gqLData[qx]);// gpLData[ x ]是中心点
						(*pRCode)[bitCnt] = (gpRData[x] > gqRData[qx]);// gpRData[ x ]是中心点	
						bitCnt++;
						//
					}

				}
			}
			pLCode++;
			pRCode++;
		}
	}
	// build cost volume
	bitset<CENSUS_BIT> lB;
	bitset<CENSUS_BIT> rB;
	pLCode = lCode;
	for (int y = 0; y < height; y++) {
		int index = y * width;
		for (int x = 0; x < width; x++) {
			lB = *pLCode;
			for (int d = 0; d < maxLevel; d++) {
				costVolPtr(y, x, d) = CENSUS_BIT + 1;
				if (x - d >= 0) {
					rB = rCode[index + x - d];//x-d是右视图中index行的第x-d个像素
					costVolPtr(y, x, d) = (lB ^ rB).count();
				}

			}
			pLCode++;
		}
	}
	delete[] lCode;
	delete[] rCode;

	return costVol;

}

/////

/////

cv::Mat CDisparityHelper::GetRight_GradXGradY_Census_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel) //直接计算右代价体
{
#define CENSUS_H 5
#define CENSUS_W 7
#define CENSUS_BIT (2*CENSUS_H*CENSUS_W-2)  //覆盖前面的CENSUS_BIT定义
	Mat limg = imL.clone();
	Mat rimg = imR.clone();

	int height = imL.rows;//高度
	int width = imL.cols;//宽度

	cv::Mat RightcostVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> rcostVolPtr((double *)RightcostVol.data, height, width, maxLevel);


	Mat lSobelX, rSobelX; //左右视图的Sobel_X梯度图像
	lSobelX = sobel_gradX(limg);
	rSobelX = sobel_gradX(rimg);

	Mat lSobelY, rSobelY; //左右视图的Sobel_Y梯度图像
	lSobelY = sobel_gradY(limg);
	rSobelY = sobel_gradY(rimg);

	// prepare binary code 
	//int H_WD = CENSUS_WIND / 2; // 4
	int H_WD = CENSUS_H / 2;
	int W_WD = CENSUS_W / 2;

#define CENSUS_BIT (2*CENSUS_H*CENSUS_W-2)  //覆盖前面的CENSUS_BIT定义

	bitset<CENSUS_BIT>* lCode = new bitset<CENSUS_BIT>[height * width];
	bitset<CENSUS_BIT>* rCode = new bitset<CENSUS_BIT>[height * width];//每个像素有80bit位序列
	bitset<CENSUS_BIT>* pLCode = lCode;//lCode存着数组首地址
	bitset<CENSUS_BIT>* pRCode = rCode;
	for (int y = 0; y < height; y++) {
		uchar* pLData = (uchar*)(lSobelX.ptr<uchar>(y));//第y行
		uchar* pRData = (uchar*)(rSobelX.ptr<uchar>(y));

		uchar* gpLData = (uchar*)(lSobelY.ptr<uchar>(y));//第y行
		uchar* gpRData = (uchar*)(rSobelY.ptr<uchar>(y));

		for (int x = 0; x < width; x++) { //第x列
			int bitCnt = 0;
			for (int wy = -H_WD; wy <= H_WD; wy++) {
				//int qy = (y + wy + height) % height;//这里+hei是为了实现循环
				int qy = y + wy;
				if (qy < 0) qy = 0;
				if (qy >= height) qy = height - 1;

				uchar* qLData = (uchar*)(lSobelX.ptr<uchar>(qy));
				uchar* qRData = (uchar*)(rSobelX.ptr<uchar>(qy));

				uchar* gqLData = (uchar*)(lSobelY.ptr<uchar>(qy));
				uchar* gqRData = (uchar*)(rSobelY.ptr<uchar>(qy));

				for (int wx = -W_WD; wx <= W_WD; wx++) {
					if (wy != 0 || wx != 0) { //中心点
											  //int qx = (x + wx + width) % width;
						int qx = x + wx;
						if (qx < 0) qx = 0;
						if (qx >= width) qx = width - 1;

						(*pLCode)[bitCnt] = (pLData[x] > qLData[qx]);// pLData[ x ]是中心点
						(*pRCode)[bitCnt] = (pRData[x] > qRData[qx]);// pRData[ x ]是中心点
						bitCnt++;

						(*pLCode)[bitCnt] = (gpLData[x] > gqLData[qx]);// pLData[ x ]是中心点
						(*pRCode)[bitCnt] = (gpRData[x] > gqRData[qx]);// pRData[ x ]是中心点
						bitCnt++; //
					}

				}
			}

			pLCode++;
			pRCode++;
		}
	}

	// build cost volume
	bitset<CENSUS_BIT> lB;
	bitset<CENSUS_BIT> rB;
	pRCode = rCode;
	for (int y = 0; y < height; y++) {
		int index = y * width;
		for (int x = 0; x < width; x++) {
			rB = *pRCode;
			for (int d = 0; d < maxLevel; d++) {
				rcostVolPtr(y, x, d) = CENSUS_BIT + 1;
				if (x + d < width) {
					lB = lCode[index + x + d];
					rcostVolPtr(y, x, d) = (rB ^ lB).count();
				}

			}
			pRCode++;
		}
	}

	delete[] lCode;
	delete[] rCode;


	return RightcostVol;
}

//---------------------------end

//-----------------------------------Gray+GradX+GradY
cv::Mat CDisparityHelper::Get_GrayGradXGradY_Census_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel)
{
#define CENSUS_H 7
#define CENSUS_W 7
#define CENSUS_BIT (3*CENSUS_H*CENSUS_W-3)  //覆盖前面的CENSUS_BIT定义

	Mat limg = imL.clone();
	Mat rimg = imR.clone();

	int height = limg.rows;
	int width = limg.cols;

	Mat lGray, rGray;
	if (imL.channels() == 3 && imR.channels() == 3) {
		Mat limg = imL.clone();
		Mat rimg = imR.clone();
		cv::cvtColor(limg, lGray, CV_BGR2GRAY);
		cv::cvtColor(rimg, rGray, CV_BGR2GRAY);
	}
	else if (imL.channels() == 1 && imR.channels() == 1)
	{
		lGray = imL.clone();
		rGray = imR.clone();
	}

	cv::Mat costVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, height, width, maxLevel);


	Mat lSobelX, rSobelX; //左右视图的Sobel_X梯度图像
	lSobelX = sobel_gradX(limg);
	rSobelX = sobel_gradX(rimg);

	Mat lSobelY, rSobelY; //左右视图的Sobel_Y梯度图像
	lSobelY = sobel_gradY(limg);
	rSobelY = sobel_gradY(rimg);

	// prepare binary code 
	//int H_WD = CENSUS_WIND / 2; // 4
	int H_WD = CENSUS_H / 2;
	int W_WD = CENSUS_W / 2;

	int mm = CENSUS_W;
	int nn = CENSUS_BIT;

	bitset<CENSUS_BIT>* lCode = new bitset<CENSUS_BIT>[height * width];
	bitset<CENSUS_BIT>* rCode = new bitset<CENSUS_BIT>[height * width];//每个像素有80bit位序列
	bitset<CENSUS_BIT>* pLCode = lCode;//lCode存着数组首地址
	bitset<CENSUS_BIT>* pRCode = rCode;
	for (int y = 0; y < height; y++) {
		uchar* pLData = (uchar*)(lGray.ptr<uchar>(y));//第y行
		uchar* pRData = (uchar*)(rGray.ptr<uchar>(y));

		uchar* gpXLData = (uchar*)(lSobelX.ptr<uchar>(y));//第y行
		uchar* gpXRData = (uchar*)(rSobelX.ptr<uchar>(y));

		uchar* gpYLData = (uchar*)(lSobelY.ptr<uchar>(y));//第y行
		uchar* gpYRData = (uchar*)(rSobelY.ptr<uchar>(y));
		for (int x = 0; x < width; x++) { //第x列
			int bitCnt = 0;
			for (int wy = -H_WD; wy <= H_WD; wy++) {
				//int qy = (y + wy + height) % height;//这里+hei是为了实现循环
				int qy = y + wy;
				if (qy < 0) qy = 0;
				if (qy >= height) qy = height - 1;
				uchar* qLData = (uchar*)(lGray.ptr<uchar>(qy));
				uchar* qRData = (uchar*)(rGray.ptr<uchar>(qy));

				uchar* gqXLData = (uchar*)(lSobelX.ptr<uchar>(qy));
				uchar* gqXRData = (uchar*)(rSobelX.ptr<uchar>(qy));

				uchar* gqYLData = (uchar*)(lSobelY.ptr<uchar>(qy));
				uchar* gqYRData = (uchar*)(rSobelY.ptr<uchar>(qy));
				for (int wx = -W_WD; wx <= W_WD; wx++) {
					if (wy != 0 || wx != 0) { //中心点
											  //int qx = (x + wx + width) % width;
						int qx = x + wx;
						if (qx < 0) qx = 0;
						if (qx >= width) qx = width - 1;

						//比较灰度
						(*pLCode)[bitCnt] = (pLData[x] > qLData[qx]);// pLData[ x ]是中心点
						(*pRCode)[bitCnt] = (pRData[x] > qRData[qx]);// pRData[ x ]是中心点	
						bitCnt++;

						//比较X方向梯度
						(*pLCode)[bitCnt] = (gpXLData[x] > gqXLData[qx]);// pLData[ x ]是中心点
						(*pRCode)[bitCnt] = (gpXRData[x] > gqXRData[qx]);// pRData[ x ]是中心点	
						bitCnt++;

						//比较Y方向梯度，在灰度后面进行拼接
						(*pLCode)[bitCnt] = (gpYLData[x] > gqYLData[qx]);// gpLData[ x ]是中心点
						(*pRCode)[bitCnt] = (gpYRData[x] > gqYRData[qx]);// gpRData[ x ]是中心点	
						bitCnt++;
						//
					}

				}
			}
			pLCode++;
			pRCode++;
		}
	}
	// build cost volume
	bitset<CENSUS_BIT> lB;
	bitset<CENSUS_BIT> rB;
	pLCode = lCode;
	for (int y = 0; y < height; y++) {
		int index = y * width;
		for (int x = 0; x < width; x++) {
			lB = *pLCode;
			for (int d = 0; d < maxLevel; d++) {
				costVolPtr(y, x, d) = CENSUS_BIT + 1;// INT_MAX; // 
				if (x - d >= 0) {
					rB = rCode[index + x - d];//x-d是右视图中index行的第x-d个像素
					costVolPtr(y, x, d) = (lB ^ rB).count();
				}

			}
			pLCode++;
		}
	}
	delete[] lCode;
	delete[] rCode;

	return costVol;
}

cv::Mat CDisparityHelper::GetRight_GrayGradXGradY_Census_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel)
{
#define CENSUS_H 7
#define CENSUS_W 7
#define CENSUS_BIT (3*CENSUS_H*CENSUS_W-3)  //覆盖前面的CENSUS_BIT定义

	Mat limg = imL.clone();
	Mat rimg = imR.clone();

	int height = limg.rows;
	int width = limg.cols;

	Mat lGray, rGray;
	if (imL.channels() == 3 && imR.channels() == 3) {
		Mat limg = imL.clone();
		Mat rimg = imR.clone();
		cv::cvtColor(limg, lGray, CV_BGR2GRAY);
		cv::cvtColor(rimg, rGray, CV_BGR2GRAY);
	}
	else if (imL.channels() == 1 && imR.channels() == 1)
	{
		lGray = imL.clone();
		rGray = imR.clone();
	}

	cv::Mat RightcostVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> rcostVolPtr((double *)RightcostVol.data, height, width, maxLevel);


	Mat lSobelX, rSobelX; //左右视图的Sobel_X梯度图像
	lSobelX = sobel_gradX(limg);
	rSobelX = sobel_gradX(rimg);

	Mat lSobelY, rSobelY; //左右视图的Sobel_Y梯度图像
	lSobelY = sobel_gradY(limg);
	rSobelY = sobel_gradY(rimg);

	// prepare binary code 
	//int H_WD = CENSUS_WIND / 2; // 4
	int H_WD = CENSUS_H / 2;
	int W_WD = CENSUS_W / 2;

	int mm = CENSUS_W;
	int nn = CENSUS_BIT;

	bitset<CENSUS_BIT>* lCode = new bitset<CENSUS_BIT>[height * width];
	bitset<CENSUS_BIT>* rCode = new bitset<CENSUS_BIT>[height * width];//每个像素有80bit位序列
	bitset<CENSUS_BIT>* pLCode = lCode;//lCode存着数组首地址
	bitset<CENSUS_BIT>* pRCode = rCode;
	for (int y = 0; y < height; y++) {
		uchar* pLData = (uchar*)(lGray.ptr<uchar>(y));//第y行
		uchar* pRData = (uchar*)(rGray.ptr<uchar>(y));

		uchar* gpXLData = (uchar*)(lSobelX.ptr<uchar>(y));//第y行
		uchar* gpXRData = (uchar*)(rSobelX.ptr<uchar>(y));

		uchar* gpYLData = (uchar*)(lSobelY.ptr<uchar>(y));//第y行
		uchar* gpYRData = (uchar*)(rSobelY.ptr<uchar>(y));
		for (int x = 0; x < width; x++) { //第x列
			int bitCnt = 0;
			for (int wy = -H_WD; wy <= H_WD; wy++) {
				//int qy = (y + wy + height) % height;//这里+hei是为了实现循环
				int qy = y + wy;
				if (qy < 0) qy = 0;
				if (qy >= height) qy = height - 1;
				uchar* qLData = (uchar*)(lGray.ptr<uchar>(qy));
				uchar* qRData = (uchar*)(rGray.ptr<uchar>(qy));

				uchar* gqXLData = (uchar*)(lSobelX.ptr<uchar>(qy));
				uchar* gqXRData = (uchar*)(rSobelX.ptr<uchar>(qy));

				uchar* gqYLData = (uchar*)(lSobelY.ptr<uchar>(qy));
				uchar* gqYRData = (uchar*)(rSobelY.ptr<uchar>(qy));
				for (int wx = -W_WD; wx <= W_WD; wx++) {
					if (wy != 0 || wx != 0) { //中心点
											  //int qx = (x + wx + width) % width;
						int qx = x + wx;
						if (qx < 0) qx = 0;
						if (qx >= width) qx = width - 1;

						//比较灰度
						(*pLCode)[bitCnt] = (pLData[x] > qLData[qx]);// pLData[ x ]是中心点
						(*pRCode)[bitCnt] = (pRData[x] > qRData[qx]);// pRData[ x ]是中心点	
						bitCnt++;

						//比较X方向梯度
						(*pLCode)[bitCnt] = (gpXLData[x] > gqXLData[qx]);// pLData[ x ]是中心点
						(*pRCode)[bitCnt] = (gpXRData[x] > gqXRData[qx]);// pRData[ x ]是中心点	
						bitCnt++;

						//比较Y方向梯度，在灰度后面进行拼接
						(*pLCode)[bitCnt] = (gpYLData[x] > gqYLData[qx]);// gpLData[ x ]是中心点
						(*pRCode)[bitCnt] = (gpYRData[x] > gqYRData[qx]);// gpRData[ x ]是中心点	
						bitCnt++;
						//
					}

				}
			}
			pLCode++;
			pRCode++;
		}
	}

	// build cost volume
	bitset<CENSUS_BIT> lB;
	bitset<CENSUS_BIT> rB;
	pRCode = rCode;
	for (int y = 0; y < height; y++) {
		int index = y * width;
		for (int x = 0; x < width; x++) {
			rB = *pRCode;
			for (int d = 0; d < maxLevel; d++) {
				rcostVolPtr(y, x, d) = CENSUS_BIT + 1;
				if (x + d < width) {
					lB = lCode[index + x + d];
					rcostVolPtr(y, x, d) = (rB ^ lB).count();
				}

			}
			pRCode++;
		}
	}

	delete[] lCode;
	delete[] rCode;


	return RightcostVol;
}
//------------------------------Gray+GradX+GradY


cv::Mat CDisparityHelper::Get_CensusPlusGradXGradYCensus_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel, float th)
{
	Mat limg = imL.clone();
	Mat rimg = imR.clone();

	int height = limg.rows;
	int width = limg.cols;

	Mat costVolCensus = GetCensusMatchingCost(limg, rimg, maxLevel);  //Get_ColorCensus_MatchingCost(limg, rimg, maxLevel); //GetCensusMatchingCost(limg, rimg, maxLevel);// // //  //// //计算census代价	
	KIdx_<double, 3> costVolCensusPtr((double *)costVolCensus.data, height, width, maxLevel);

	cv::Mat costVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, height, width, maxLevel);

	//判断是否存在光照变化
	/*
	cv::Mat lgray,rgray;
	cvtColor(limg, lgray, CV_BGR2GRAY); cvtColor(rimg, rgray, CV_BGR2GRAY);
	cv::Scalar     lmean,rmean;
	cv::Scalar     dev;
	cv::meanStdDev(lgray, lmean, dev); cv::meanStdDev(rgray, rmean, dev);
	float       leftmean, rightmean;
	leftmean = lmean.val[0]; rightmean = rmean.val[0];
	*/

	Mat gcostVolCensus;
	bool isGradCensus = 0; // (1.0*abs(leftmean - rightmean)) / min(leftmean, rightmean) > 0.4; //就可以认为亮度变化比较大，适合使用梯度进行求解？	
	if (isGradCensus) // using sobel Grad, or using GradX, or using GradY
	{
		Mat lsobel, rsobel;
		lsobel = sobel_grad(limg);
		rsobel = sobel_grad(rimg);
		gcostVolCensus = GetCensusMatchingCost(lsobel, rsobel, maxLevel); //计算census代价			
	}
	else // using GradXGradY
	{
		gcostVolCensus = Get_GradXGradY_Census_MatchingCost(limg, rimg, maxLevel);
	}

	KIdx_<double, 3> gcostVolCensusPtr((double *)gcostVolCensus.data, height, width, maxLevel);

	for (int d = 0; d < maxLevel; d++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int rx = x - d; // left image中的x对应着right image 中的rx
				if (rx < 0) rx = 0; //修正为0
				{
					//float th = 0.3333;
					costVolPtr(y, x, d) = th * costVolCensusPtr(y, x, d) + (1 - th)*gcostVolCensusPtr(y, x, d);

				}
			}
		}
	}

	return costVol;
}

cv::Mat CDisparityHelper::GetRight_CensusPlusGradXGradYCensus_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel, float th)
{
	Mat limg = imL.clone();
	Mat rimg = imR.clone();

	int height = limg.rows;
	int width = limg.cols;

	Mat costVolCensus = GetRightCensusMatchingCost(limg, rimg, maxLevel); //GetRight_ColorCensus_MatchingCost(limg, rimg, maxLevel); // //GetRight_ColorCensus_MatchingCost(limg, rimg, maxLevel); // // ////  //计算census代价	
	KIdx_<double, 3> costVolCensusPtr((double *)costVolCensus.data, height, width, maxLevel);

	cv::Mat costVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, height, width, maxLevel);

	//判断是否存在光照变化
	/*
	cv::Mat lgray,rgray;
	cvtColor(limg, lgray, CV_BGR2GRAY); cvtColor(rimg, rgray, CV_BGR2GRAY);
	cv::Scalar     lmean,rmean;
	cv::Scalar     dev;
	cv::meanStdDev(lgray, lmean, dev); cv::meanStdDev(rgray, rmean, dev);
	float       leftmean, rightmean;
	leftmean = lmean.val[0]; rightmean = rmean.val[0];
	*/

	Mat gcostVolCensus;
	bool isGradCensus = 0; // (1.0*abs(leftmean - rightmean)) / min(leftmean, rightmean) > 0.4; //就可以认为亮度变化比较大，适合使用梯度进行求解？	
	if (isGradCensus) // using sobel Grad, or using GradX, or using GradY
	{
		Mat lsobel, rsobel;
		lsobel = sobel_grad(limg);
		rsobel = sobel_grad(rimg);
		gcostVolCensus = GetRightCensusMatchingCost(lsobel, rsobel, maxLevel); //计算census代价			
	}
	else // using GradXGradY
	{
		gcostVolCensus = GetRight_GradXGradY_Census_MatchingCost(limg, rimg, maxLevel);
	}

	KIdx_<double, 3> gcostVolCensusPtr((double *)gcostVolCensus.data, height, width, maxLevel);

	for (int d = 0; d < maxLevel; d++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int rx = x - d; // left image中的x对应着right image 中的rx
				if (rx < 0) rx = 0; //修正为0
				{   //float th = 0.3333;
					costVolPtr(y, x, d) = th * costVolCensusPtr(y, x, d) + (1 - th)*gcostVolCensusPtr(y, x, d);

				}
			}
		}
	}

	return costVol;
}
//----------------------------


cv::Mat CDisparityHelper::GetMatchingCost_RWR(cv::Mat imL, cv::Mat imR, int maxLevel)
{
	Mat limg = imL.clone();
	Mat rimg = imR.clone();

	int height = limg.rows;
	int width = limg.cols;

	Mat costVolCensus = GetCensusMatchingCost(limg, rimg, maxLevel); //Get_Census_ColorGrad_MatchingCost(limg, rimg, maxLevel); // GetCensusMatchingCost(limg, rimg, maxLevel); //计算census代价	
	KIdx_<double, 3> costVolCensusPtr((double *)costVolCensus.data, height, width, maxLevel);

	cv::Mat costVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, height, width, maxLevel);

	Mat lSobelX, rSobelX; //左右视图的Sobel_X梯度图像
	lSobelX = sobel_gradX(limg);
	rSobelX = sobel_gradX(rimg);

	Mat lSobelY, rSobelY; //左右视图的Sobel_Y梯度图像
	lSobelY = sobel_gradY(limg);
	rSobelY = sobel_gradY(rimg);

	for (int d = 0; d < maxLevel; d++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int rx = x - d; // left image中的x对应着right image 中的rx
				if (rx < 0) rx = 0; //修正为0
				{
					double deta1 = 0.2, deta2 = 1.0;
					double T1 = 5, T2 = 1.7;

					double costCensus = std::min(T1, costVolCensusPtr(y, x, d)); //gcostVolCensus(y,x,d);

					double costGradient = std::fabs(lSobelX.at<uchar>(y, x) - rSobelX.at<uchar>(y, rx)) + std::fabs(lSobelY.at<uchar>(y, x) - rSobelY.at<uchar>(y, rx));
					costGradient = std::min(T2, costGradient);

					costVolPtr(y, x, d) = deta1 * costCensus + deta2 * costGradient;
				}
			}
		}
	}
	return costVol;
}

cv::Mat CDisparityHelper::GetRightMatchingCost_RWR(cv::Mat imL, cv::Mat imR, int maxLevel)
{
	Mat limg = imL.clone();
	Mat rimg = imR.clone();

	int height = limg.rows;
	int width = limg.cols;

	Mat costVolCensus = GetRightCensusMatchingCost(limg, rimg, maxLevel); //Get_Census_ColorGrad_MatchingCost(limg, rimg, maxLevel); // GetCensusMatchingCost(limg, rimg, maxLevel); //计算census代价	
	KIdx_<double, 3> costVolCensusPtr((double *)costVolCensus.data, height, width, maxLevel);

	cv::Mat costVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, height, width, maxLevel);

	Mat lSobelX, rSobelX; //左右视图的Sobel_X梯度图像
	lSobelX = sobel_gradX(limg);
	rSobelX = sobel_gradX(rimg);

	Mat lSobelY, rSobelY; //左右视图的Sobel_Y梯度图像
	lSobelY = sobel_gradY(limg);
	rSobelY = sobel_gradY(rimg);

	for (int d = 0; d < maxLevel; d++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int lx = x + d; // left image中的x对应着right image 中的rx
				if (lx >= width) lx = width-1; //修正为0
				{
					double deta1 = 0.2, deta2 = 1.0;
					double T1 = 5, T2 = 1.7;

					double costCensus = std::min(T1, costVolCensusPtr(y, x, d)); //gcostVolCensus(y,x,d);

					double costGradient = std::fabs(lSobelX.at<uchar>(y, lx) - rSobelX.at<uchar>(y, x)) + std::fabs(lSobelY.at<uchar>(y, lx) - rSobelY.at<uchar>(y, x));
					costGradient = std::min(T2, costGradient);

					costVolPtr(y, x, d) = deta1 * costCensus + deta2 * costGradient;
				}
			}
		}
	}

	return costVol;
}

cv::Mat CDisparityHelper::GetCensusMatchingCost_kongdong(cv::Mat imL, cv::Mat imR, int maxLevel)
{
#define CENSUS_H 11
#define CENSUS_W 11
#define CENSUS_BIT (CENSUS_H*CENSUS_W-1)

	int height = imL.rows;//高度
	int width = imL.cols;//宽度


	Mat lGray, rGray;
	if (imL.channels() == 3 && imR.channels() == 3) {
		Mat limg = imL.clone();
		Mat rimg = imR.clone();
		cv::cvtColor(limg, lGray, CV_BGR2GRAY);
		cv::cvtColor(rimg, rGray, CV_BGR2GRAY);
	}
	else if (imL.channels() == 1 && imR.channels() == 1)
	{
		lGray = imL.clone();
		rGray = imR.clone();
	}

	cv::Mat costVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, height, width, maxLevel);

	// prepare binary code 
	//int H_WD = CENSUS_WIND / 2; // 4
	int H_WD = CENSUS_H / 2;
	int W_WD = CENSUS_W / 2;
	bitset<CENSUS_BIT>* lCode = new bitset<CENSUS_BIT>[height * width];
	bitset<CENSUS_BIT>* rCode = new bitset<CENSUS_BIT>[height * width];//每个像素有80bit位序列
	bitset<CENSUS_BIT>* pLCode = lCode;//lCode存着数组首地址
	bitset<CENSUS_BIT>* pRCode = rCode;
	for (int y = 0; y < height; y++) {
		uchar* pLData = (uchar*)(lGray.ptr<uchar>(y));//第y行
		uchar* pRData = (uchar*)(rGray.ptr<uchar>(y));
		for (int x = 0; x < width; x++) { //第x列
			int bitCnt = 0;
			for (int wy = -H_WD; wy <= H_WD; wy = wy + 2) {
				//int qy = (y + wy + height) % height;//这里+hei是为了实现循环
				int qy = y + wy;
				if (qy < 0) qy = 0;
				if (qy >= height) qy = height - 1;
				uchar* qLData = (uchar*)(lGray.ptr<uchar>(qy));
				uchar* qRData = (uchar*)(rGray.ptr<uchar>(qy));
				for (int wx = -W_WD; wx <= W_WD; wx = wx + 2) {
					if (wy != 0 || wx != 0) { //中心点
						//int qx = (x + wx + width) % width;
						int qx = x + wx;
						if (qx < 0) qx = 0;
						if (qx >= width) qx = width - 1;
						(*pLCode)[bitCnt] = (pLData[x] > qLData[qx]); // pLData[ x ]是中心点
						(*pRCode)[bitCnt] = (pRData[x] > qRData[qx]); // pRData[ x ]是中心点												
						bitCnt++;
						//0到79
					}
				}
			}

			/*
			//增加上下对角的比较
			int left = max(0, x - 1);
			int right = min(x + 1, width - 1);
			int up = max(0, y - 1);
			int down = min(y + 1, height - 1);

			(*pLCode)[bitCnt] = lGray.at<uchar>(up, left) > lGray.at<uchar>(down, right);
			bitCnt++;
			(*pLCode)[bitCnt] = lGray.at<uchar>(up, x) > lGray.at<uchar>(down, x);
			bitCnt++;
			(*pLCode)[bitCnt] = lGray.at<uchar>(up, right) > lGray.at<uchar>(down, left);
			bitCnt++;
			(*pLCode)[bitCnt] = lGray.at<uchar>(y, left) > lGray.at<uchar>(down, right);
			bitCnt++;
			*/

			pLCode++;
			pRCode++;
		}
	}
	// build cost volume
	bitset<CENSUS_BIT> lB;
	bitset<CENSUS_BIT> rB;
	pLCode = lCode;
	for (int y = 0; y < height; y++) {
		int index = y * width;
		for (int x = 0; x < width; x++) {
			lB = *pLCode;
			for (int d = 0; d < maxLevel; d++) {
				costVolPtr(y, x, d) = CENSUS_BIT + 1; // 
				if (x - d >= 0) {
					rB = rCode[index + x - d];//x-d是右视图中index行的第x-d个像素
					costVolPtr(y, x, d) = (lB ^ rB).count();
				}

			}
			pLCode++;
		}
	}
	delete[] lCode;
	delete[] rCode;

	return costVol;
}

//--------------------left image为参考图像，AD+Census代价函数，Census 可以为ColorCensus,GrayCensus和GradCensus等
cv::Mat CDisparityHelper::Get_AD_Census_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel)
{
	Mat limg = imL.clone();
	Mat rimg = imR.clone();

	int height = limg.rows;
	int width = limg.cols;

	Mat costVolCensus = GetCensusMatchingCost(limg, rimg, maxLevel); //Get_Census_ColorGrad_MatchingCost(limg, rimg, maxLevel); //  //// GetCensusMatchingCost(limg, rimg, maxLevel); //计算census代价	
	KIdx_<double, 3> costVolCensusPtr((double *)costVolCensus.data, height, width, maxLevel);

	cv::Mat costVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, height, width, maxLevel);

	//判断是否存在光照变化
	/*
	cv::Mat lgray,rgray;
	cvtColor(limg, lgray, CV_BGR2GRAY); cvtColor(rimg, rgray, CV_BGR2GRAY);
	cv::Scalar     lmean,rmean;
	cv::Scalar     dev;
	cv::meanStdDev(lgray, lmean, dev); cv::meanStdDev(rgray, rmean, dev);
	float       leftmean, rightmean;
	leftmean = lmean.val[0]; rightmean = rmean.val[0];
	*/

	bool isGradCensus = 0; // (1.0*abs(leftmean - rightmean)) / min(leftmean, rightmean) > 0.4; //就可以认为亮度变化比较大，适合使用梯度进行求解？
	Mat gcostVolCensus;
	if (isGradCensus)
	{
		Mat lsobel, rsobel;
		lsobel = sobel_grad(limg);
		rsobel = sobel_grad(rimg);

		//gcostVolCensus = GetCensusMatchingCost(lsobel, rsobel, maxLevel); //计算census代价	
		gcostVolCensus = Get_GradXGradY_Census_MatchingCost(limg, rimg, maxLevel);
	}

	KIdx_<double, 3> gcostVolCensusPtr((double *)gcostVolCensus.data, height, width, maxLevel);

	for (int d = 0; d < maxLevel; d++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int rx = x - d; // left image中的x对应着right image 中的rx
				if (rx < 0) rx = 0; //修正为0
				{
					cv::Vec3b &colorL = limg.at<cv::Vec3b>(y, x);
					cv::Vec3b &colorR = rimg.at<cv::Vec3b>(y, rx);

					float lamda1 = 10, lamda2 = 30;
					float tempcostColor = (std::fabs(colorL(0) - colorR(0)) + std::fabs(colorL(1) - colorR(1)) + std::fabs(colorL(2) - colorR(2))) / 3.0;
					float tempcostCensus = costVolCensusPtr(y, x, d); //gcostVolCensus(y,x,d);
					costVolPtr(y, x, d) = 2 - exp(-tempcostColor / lamda1) - exp(-tempcostCensus / (lamda2));

				}
			}
		}
	}

	return costVol;
}

//--------right image为参考图像，AD+Census代价函数，或将GrayCensus和GradCensus进行加权求和
cv::Mat CDisparityHelper::GetRight_AD_Census_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel)
{
	Mat limg = imL.clone(); //BRG char
	Mat rimg = imR.clone();

	int height = limg.rows;
	int width = limg.cols;

	Mat RightcostVolCensus = GetRightCensusMatchingCost(limg, rimg, maxLevel); //计算census代价	
	KIdx_<double, 3> rcostVolCensusPtr((double *)RightcostVolCensus.data, height, width, maxLevel);

	cv::Mat RightcostVol(1, height* width* maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)RightcostVol.data, height, width, maxLevel);

	//判断是否存在光照变化
	/*
	cv::Mat lgray, rgray;
	cvtColor(limg, lgray, CV_BGR2GRAY); cvtColor(rimg, rgray, CV_BGR2GRAY);
	cv::Scalar     lmean, rmean;
	cv::Scalar     dev;
	cv::meanStdDev(lgray, lmean, dev); cv::meanStdDev(rgray, rmean, dev);
	float       leftmean, rightmean;
	leftmean = lmean.val[0]; rightmean = rmean.val[0];
	*/
	bool isGradCensus = 0;// (1.0*abs(leftmean - rightmean)) / min(leftmean, rightmean) > 0.4; //就可以认为亮度变化比较大，适合使用梯度进行求解？
	Mat gRightcostVolCensus;
	if (isGradCensus)
	{

		Mat lsobel, rsobel;
		lsobel = sobel_grad(limg);
		rsobel = sobel_grad(rimg);
		//gRightcostVolCensus = GetRightCensusMatchingCost(lsobel, rsobel, maxLevel); //计算census代价
		gRightcostVolCensus = Get_GradXGradY_Census_MatchingCost(limg, rimg, maxLevel);
	}
	KIdx_<double, 3> grcostVolCensusPtr((double *)gRightcostVolCensus.data, height, width, maxLevel);

	for (int d = 0; d < maxLevel; d++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int lx = x + d; // right image中的x对应着left image 中的lx
				if (lx > width - 1) lx = width - 1; //修正为0

				cv::Vec3b &colorL = limg.at<cv::Vec3b>(y, lx);
				cv::Vec3b &colorR = rimg.at<cv::Vec3b>(y, x);
				float lamda1 = 10, lamda2 = 30;
				float tempcostColor = (std::fabs(colorL(0) - colorR(0)) + std::fabs(colorL(1) - colorR(1)) + std::fabs(colorL(2) - colorR(2))) / 3.0;
				float tempcostCensus = rcostVolCensusPtr(y, x, d);
				costVolPtr(y, x, d) = 2 - exp(-tempcostColor / lamda1) - exp(-tempcostCensus / (lamda2));

			}
		}
	}
	return RightcostVol;
}
//----------------------------




//----------------------------
cv::Mat CDisparityHelper::GetGradient(cv::InputArray img_) { //计算梯度
	cv::Mat img = img_.getMat();
	cv::Size imageSize = img.size();
	
	CV_Assert(img.type() == CV_8UC3);
	
	cv::Mat gradient(imageSize, CV_32F, cv::Scalar(0.0f));
	
	cv::Mat imgGray(imageSize, CV_8U);
	for(int i = 0;i < imageSize.area();i++) {
		imgGray.data[i] = rgb_2_gray(&img.data[i * 3]);
	}

	cv::Mat1b imgGrayPtr = imgGray;
	cv::Mat1f gradientPtr = gradient;
	float grayMinus, grayPlus;
	
	for(int y = 0;y < imageSize.height;y++) {
		grayPlus  = imgGrayPtr(y, 1);
		grayMinus = imgGrayPtr(y, 0);
		gradientPtr(y, 0) = grayPlus - grayMinus + 127.5f;

		for(int x = 1;x < imageSize.width - 1;x++) {
			grayPlus  = imgGrayPtr(y, x + 1);
			grayMinus = imgGrayPtr(y, x - 1);
			gradientPtr(y, x) = 0.5f * (grayPlus - grayMinus) + 127.5f;
		}

		grayPlus  = imgGrayPtr(y, imageSize.width - 1);
		grayMinus = imgGrayPtr(y, imageSize.width - 2);
		gradientPtr(y, imageSize.width - 1) = grayPlus - grayMinus + 127.5f;
	}

	return gradient;
}

cv::Mat CDisparityHelper::GetMatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel) { //改写了yqx的代码，实现原理一模一样
	cv::Mat gradL = GetGradient(imL); //左视图的梯度
	cv::Mat gradR = GetGradient(imR); // 右视图的梯度

//default: set the same as the non-local cost aggregation from QingXiong Yang in his CVPR 2012 paper
	double max_color_difference = 7;
	double max_gradient_difference = 2;
	double weight_on_color = 0.11;
	double weight_on_gradient = 1.0 - weight_on_color;
//end of default

	cv::Size imageSize = imL.size();
	
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	cv::Mat imageShifted(imageSize, CV_8UC3);
	cv::Mat gradientShifted(imageSize, CV_32F);

	KIdx_<uchar, 3> imLPtr(imL.data, imageSize.height, imageSize.width, 3);
	KIdx_<uchar, 3> imRPtr(imR.data, imageSize.height, imageSize.width, 3);
	KIdx_<uchar, 3> imShiftPtr(imageShifted.data, imageSize.height, imageSize.width, 3);

	KIdx_<float, 2> gradLPtr((float *)gradL.data, imageSize.height, imageSize.width);
	KIdx_<float, 2> gradRPtr((float *)gradR.data, imageSize.height, imageSize.width);
	KIdx_<float, 2> gShiftPtr((float *)gradientShifted.data, imageSize.height, imageSize.width);

	for(int i = 0;i < maxLevel;i++) {
		//shift the right image by i pixels
		for(int y = 0;y < imageSize.height;y++) {//xL-i=xR，故右视图的每一行向右平移i个像素，这样才能和左视图对齐。
			memcpy(&imShiftPtr(y, i, 0), &imRPtr(y, 0, 0), sizeof(uchar) * (imageSize.width - i) * 3);
			memcpy(&gShiftPtr(y, i), &gradRPtr(y, 0), sizeof(float) * (imageSize.width - i));
			for(int x = 0;x < i;x++) { //当视差为i时，补所遮挡的部分
				memcpy(&imShiftPtr(y, x, 0), &imRPtr(y, 0, 0), sizeof(uchar) * 3);
				gShiftPtr(y, x) = gradRPtr(y, 0);
			}
		}

		for(int y = 0;y < imageSize.height;y++) {
			for(int x = 0;x < imageSize.width;x++) {
				double costColor = 0, costGradient;
				
				for(int c = 0; c < 3;c++) costColor += abs(imLPtr(y, x, c) - imShiftPtr(y, x, c));
				costColor = std::min(costColor / 3, max_color_difference);//Trunck
				
				costGradient = fabs(gradLPtr(y, x) - gShiftPtr(y, x));
				costGradient = std::min(costGradient, max_gradient_difference);//Trunck
				
				costVolPtr(y, x, i) = double(weight_on_color * costColor + weight_on_gradient * costGradient);
			}
		}
	}
	
	return costVol;
}

/*
cv::Mat CDisparityHelper::GetMatchingCost_CrossScale_AdGradient(cv::Mat imL, cv::Mat imR, int maxLevel) {
	// set image format
	cv::Mat lImg = imL.clone();
	cv::Mat rImg = imR.clone();
	cvtColor(lImg, lImg, CV_BGR2RGB);
	cvtColor(rImg, rImg, CV_BGR2RGB);
	lImg.convertTo(lImg, CV_64F, 1 / 255.0f);
	rImg.convertTo(rImg, CV_64F, 1 / 255.0f);

	cv::Mat * costVol_cs = new Mat[maxLevel];
	for (int mIdx = 0; mIdx < maxLevel; mIdx++) {
		costVol_cs[mIdx] = Mat::zeros(imL.rows, imL.cols, CV_64FC1);
	}

	GrdCC* Grdptr = new GrdCC();
	Grdptr->buildCV(lImg, rImg, maxLevel, costVol_cs);

	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = costVol_cs[i].at<double>(y,x);
			}
		}
	}

	return costVol;
}
*/

/*
cv::Mat CDisparityHelper::GetRightMatchingCost_CrossScale_AdGradient(const cv::Mat imL, const cv::Mat imR, int maxLevel) { //RGB格式
																													  // set image format
	cv::Mat lImg = imL.clone();
	cv::Mat rImg = imR.clone();
	cvtColor(lImg, lImg, CV_BGR2RGB);
	cvtColor(rImg, rImg, CV_BGR2RGB);
	lImg.convertTo(lImg, CV_64F, 1 / 255.0f);
	rImg.convertTo(rImg, CV_64F, 1 / 255.0f);

	cv::Mat * costVol_cs = new Mat[maxLevel];
	for (int mIdx = 0; mIdx < maxLevel; mIdx++) {
		costVol_cs[mIdx] = Mat::zeros(imL.rows, imL.cols, CV_64FC1);
	}

	GrdCC* Grdptr = new GrdCC();
	Grdptr->buildRightCV(lImg, rImg, maxLevel, costVol_cs);

	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = costVol_cs[i].at<double>(y, x);
			}
		}
	}
	delete[]costVol_cs;
	return costVol;
}
*/

/*
cv::Mat CDisparityHelper::GetMatchingCost_CrossScale_Census(cv::Mat imL, cv::Mat imR, int maxLevel) {
	// set image format
	cv::Mat lImg = imL.clone();
	cv::Mat rImg = imR.clone();
	cvtColor(lImg, lImg, CV_BGR2RGB);
	cvtColor(rImg, rImg, CV_BGR2RGB);
	lImg.convertTo(lImg, CV_64F, 1 / 255.0f);
	rImg.convertTo(rImg, CV_64F, 1 / 255.0f);

	cv::Mat * costVol_cs = new Mat[maxLevel];
	for (int mIdx = 0; mIdx < maxLevel; mIdx++) {
		costVol_cs[mIdx] = Mat::zeros(imL.rows, imL.cols, CV_64FC1);
	}

	CenCC* Cenptr = new CenCC();
	Cenptr->buildCV(lImg, rImg, maxLevel, costVol_cs);

	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = costVol_cs[i].at<double>(y, x);
			}
		}
	}

	return costVol;
}
*/

/*
cv::Mat CDisparityHelper::GetMatchingCost_CrossScale_CensusGradient(cv::Mat imL, cv::Mat imR, int maxLevel) {
	// set image format
	cv::Mat lImg = imL.clone();
	cv::Mat rImg = imR.clone();
	cvtColor(lImg, lImg, CV_BGR2RGB);
	cvtColor(rImg, rImg, CV_BGR2RGB);
	lImg.convertTo(lImg, CV_64F, 1 / 255.0f);
	rImg.convertTo(rImg, CV_64F, 1 / 255.0f);

	cv::Mat * costVol_cs = new Mat[maxLevel];
	for (int mIdx = 0; mIdx < maxLevel; mIdx++) {
		costVol_cs[mIdx] = Mat::zeros(imL.rows, imL.cols, CV_64FC1);
	}

	CGCC* CGptr = new CGCC();
	CGptr->buildCV(lImg, rImg, maxLevel, costVol_cs);


	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = costVol_cs[i].at<double>(y, x);
			}
		}
	}

	return costVol;
}
*/

/*
//下面封装MeshStereo中的代价体的计算
cv::Mat CDisparityHelper::GetMatchingCost_MeshStereo_AdGradient(cv::Mat imL, cv::Mat imR, int maxLevel) //使用MeshStero的ADGradiend计算代价体
{
	float GRANULARITY = 1.0;
	MCImg<float>	gDsiL;
	MCImg<float>	gDsiR;
	gDsiL = ComputeAdGradientCostVolume(imL, imR, maxLevel, -1, GRANULARITY); //左视图为参考视图，计算出的代价体
	//gDsiR = ComputeAdGradientCostVolume(imR, imL, maxLevel, +1, GRANULARITY); //右视图为参考视图，计算出的代价体

	//下面将MCImg<float>	gDsiL转换成 cv::Mat costVol(1, imageSize.area() * maxLevel, CV_32F);
	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = gDsiL.get(y, x)[i];
			}
		}
	}

	return costVol;
}
*/

/*
cv::Mat CDisparityHelper::GetMatchingCost_MeshStereo_Census(cv::Mat imL, cv::Mat imR, int maxLevel)
{
	float GRANULARITY = 1.0;
	MCImg<float>	gDsiL;
	MCImg<float>	gDsiR;
	gDsiL = Compute9x7CensusCostVolume(imL, imR, maxLevel, -1, GRANULARITY); //左视图为参考视图，计算出的代价体
																			 //gDsiR = ComputeAdCensusCostVolume(imR, imL, maxLevel, +1, GRANULARITY); //右视图为参考视图，计算出的代价体

																			 //下面将MCImg<float>	gDsiL转换成 cv::Mat costVol(1, imageSize.area() * maxLevel, CV_32F);
	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = gDsiL.get(y, x)[i];
			}
		}
	}
	return costVol;
}
*/

/*
cv::Mat CDisparityHelper::GetMatchingCost_MeshStereo_AdCensus(cv::Mat imL, cv::Mat imR, int maxLevel) //使用MeshStero的AdCensus计算代价体
{
	float GRANULARITY = 1.0;
	MCImg<float>	gDsiL;
	MCImg<float>	gDsiR;
	gDsiL = ComputeAdCensusCostVolume(imL, imR, maxLevel, -1, GRANULARITY); //左视图为参考视图，计算出的代价体
																			//gDsiR = ComputeAdCensusCostVolume(imR, imL, maxLevel, +1, GRANULARITY); //右视图为参考视图，计算出的代价体

																			//下面将MCImg<float>	gDsiL转换成 cv::Mat costVol(1, imageSize.area() * maxLevel, CV_32F);
	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = gDsiL.get(y, x)[i];
			}
		}
	}
	return costVol;
}
*/

/*
cv::Mat CDisparityHelper::GetRightMatchingCost_MeshStereo_AdCensus(cv::Mat imL, cv::Mat imR, int maxLevel) //使用MeshStero的AdCensus计算代价体
{
	float GRANULARITY = 1.0;
	MCImg<float>	gDsiL;
	MCImg<float>	gDsiR;
	//gDsiL = ComputeAdCensusCostVolume(imL, imR, maxLevel, -1, GRANULARITY); //左视图为参考视图，计算出的代价体
	gDsiR = ComputeAdCensusCostVolume(imR, imL, maxLevel, +1, GRANULARITY); //右视图为参考视图，计算出的代价体

																			//下面将MCImg<float>	gDsiL转换成 cv::Mat costVol(1, imageSize.area() * maxLevel, CV_32F);
	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = gDsiR.get(y, x)[i];
			}
		}
	}
	return costVol;
}
*/

/*
cv::Mat CDisparityHelper::GetMatchingCost_PatchMatchStereo_AdGradient(cv::Mat imL, cv::Mat imR, int maxLevel)
{
	float granularity = 1.0;// 1.0 / 4;
	VECBITMAP<float> dsiL = ComputeAdGradientCostVolume_Patch(imL, imR, maxLevel, -1, granularity);
	//VECBITMAP<float> dsiR = ComputeAdGradientCostVolume(imR, imL, maxLevel, +1, granularity);//将视差范围放大了4倍

	//下面将VECBITMAP<float> dsiL	转换成 cv::Mat costVol(1, imageSize.area() * maxLevel, CV_32F);
	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = dsiL.get(y, x)[i];
			}
		}
	}
	return costVol;
}
*/

/*
cv::Mat CDisparityHelper::GetMatchingCost_PatchMatchStereo_Census(cv::Mat imL, cv::Mat imR, int maxLevel)
{
	float granularity = 1.0;// 1.0 / 4;

	VECBITMAP<float> dsiL = ComputeCensusTensor(imL, imR, -1, maxLevel);
	//VECBITMAP<float> dsiR = ComputeAdGradientCostVolume(imR, imL, maxLevel, +1, granularity);//将视差范围放大了4倍

	//下面将VECBITMAP<float> dsiL	转换成 cv::Mat costVol(1, imageSize.area() * maxLevel, CV_32F);
	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = dsiL.get(y, x)[i];
			}
		}
	}
	return costVol;
}
*/

/*
cv::Mat CDisparityHelper::GetMatchingCost_PatchMatchStereo_AdCensus(cv::Mat imL, cv::Mat imR, int maxLevel)
{
	
	VECBITMAP<float> dsiL = ComputeAdCensusCostVolume_Patch(imL, imR, maxLevel, -1);
	//VECBITMAP<float> dsiR =  ComputeAdCensusCostVolume_Patch(imL, imR, maxLevel, -1);//将视差范围放大了4倍

	//下面将VECBITMAP<float> dsiL	转换成 cv::Mat costVol(1, imageSize.area() * maxLevel, CV_32F);
	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = dsiL.get(y, x)[i];
			}
		}
	}
	return costVol;
}
*/

/*
cv::Mat CDisparityHelper::GetMatchingCost_ADCensusStereo_AdCensus(cv::Mat imL, cv::Mat imR, int maxLevel)
{
	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	Size censusWin;
	censusWin.height = 9;
	censusWin.width = 7;
	float lambdaAD = 10.0;
	float lambdaCensus = 30.0;
	ADCensusCV *AdCensus = new ADCensusCV(imL,imR,censusWin, lambdaAD, lambdaCensus);

	cv::Mat AdCensuscostVol=AdCensus->ComputeAdCensusCostVolume(imL,  imR,maxLevel);
	KIdx_<float, 3> AdCensuscostVolPtr((float *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = AdCensuscostVolPtr(y, x, i);
			}
		}
	}

	return costVol;
}
*/

/*
//-------------------------
cv::Mat CDisparityHelper::GetMatchingCost_SGMStereo_CensusGradient(std::string &filePathImageL, std::string &filePathImageR, int maxLevel) //使用SGMStereo的CensusGradient计算代价体
{
	cv::Mat imL = cv::imread(filePathImageL);
	MCImg<unsigned short> gDsiL, gDsiR;
	int numRows = imL.rows, numCols = imL.cols;
	gDsiL = MCImg<unsigned short>(numRows, numCols, maxLevel);
	gDsiR = MCImg<unsigned short>(numRows, numCols, maxLevel);
	std::cout << filePathImageL << "\n";
	std::cout << filePathImageR << "\n";
	BuildCensusGradientCostVolume<unsigned short>(filePathImageL, filePathImageR, gDsiL, gDsiR, maxLevel);

	//现在将gDsiL装备到cv::Mat costVol(1, imageSize.area() * maxLevel, CV_32F);中
	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = gDsiL.get(y, x)[i];
			}
		}
	}
	return costVol;

}
*/

/*
//----------------------------，这里实际上可以同时求得 左右两个代价体 gDsiL, gDsiR
cv::Mat CDisparityHelper::GetMatchingCost_SGMStereo_CensusGradient(const cv::Mat imL, const cv::Mat imR, int maxLevel) //使用SGMStereo的CensusGradient计算代价体, RGB格式
{
	cv::Mat leftImg = imL.clone();
	cv::Mat rightImg = imR.clone();
	cvtColor(leftImg, leftImg, CV_BGR2RGB);
	cvtColor(rightImg, rightImg, CV_BGR2RGB);

	MCImg<unsigned short> gDsiL, gDsiR;
	int numRows = leftImg.rows, numCols = leftImg.cols;
	gDsiL = MCImg<unsigned short>(numRows, numCols, maxLevel);
	gDsiR = MCImg<unsigned short>(numRows, numCols, maxLevel);

	BuildCensusGradientCostVolume<unsigned short>(leftImg, rightImg, gDsiL, gDsiR, maxLevel);

	//现在将gDsiL装备到cv::Mat costVol(1, imageSize.area() * maxLevel, CV_32F);中
	cv::Size imageSize = imL.size();
	cv::Mat costVol(1, imageSize.area() * maxLevel, CV_64F);
	KIdx_<double, 3> costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, maxLevel);

	for (int i = 0; i < maxLevel; i++) {
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				costVolPtr(y, x, i) = gDsiL.get(y, x)[i];//(double)
			}
		}
	}

	return costVol;

}
//------------------------------------------------
*/

//-----------------------------------------------
cv::Mat CDisparityHelper::GetRightMatchingCostFromLeft(cv::Mat leftVol, int w, int h, int maxLevel) { //改写yqx的代码
	cv::Mat rightVol = leftVol.clone();
	KIdx_<double, 3>  leftPtr((double*)leftVol.data, h, w, maxLevel);
	KIdx_<double, 3>  rightPtr((double*)rightVol.data, h, w, maxLevel);

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w - maxLevel; x++) {
			for (int d = 0; d < maxLevel; d++) {
				rightPtr(y, x, d) = leftPtr(y, x + d, d);
			}
		}

		for (int x = w - maxLevel; x < w; x++) {
			for (int d = 0; d < maxLevel; d++) {
				if ((x + d) < w) {
					rightPtr(y, x, d) = leftPtr(y, x + d, d);
				}
				else {
					rightPtr(y, x, d) = rightPtr(y, x, d - 1);
				}
			}
		}
	}

	return rightVol;
}

//左右一致性检查, 左右一致性检查，检测遮挡
void CDisparityHelper::Detect_occlusion_cross_check(cv::Mat &DispL, cv::Mat &DispR, cv::Mat &Mask, int maxLevel) 
{
	CV_Assert(DispL.type() == CV_8U&&DispR.type() == CV_8U);
	cv::Size imageSize = DispL.size();
	//left-right check with right view disparity
	
	cv::Mat occtable(imageSize, CV_8U, cv::Scalar(0));

	cv::Mat1b leftPtr = DispL;
	cv::Mat1b rightPtr = DispR;
	cv::Mat1b occPtr = occtable;
	cv::Mat1b maskPtr = Mask;
	for (int y = 0; y < imageSize.height; y++) {
		for (int x = 0; x < imageSize.width; x++) {
			int d = leftPtr(y, x);
			if (x - d >= 0) {
				int d_cor = rightPtr(y, x - d);
				occPtr(y, x) = (d == 0 || abs(d - d_cor) > 1);
			}
			else {
				occPtr(y, x) = 1;
			}
			maskPtr(y, x) = !occPtr(y, x);
		}
	}
}



cv::Mat CDisparityHelper::GetDisparity_WTA(double *costVol, int w, int h, int maxLevel) {
	cv::Mat disparity(h, w, CV_8U);
	cv::Mat1b disparityPtr = disparity;

	KIdx_<double, 3>  costPtr(costVol, h, w, maxLevel);

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int minPos = 0;
			double minValue = costPtr(y, x, minPos);

			for (int i = 1; i < maxLevel; i++) {
				if (costPtr(y, x, i) < minValue) {
					minValue = costPtr(y, x, i);
					minPos = i;
				}
			}

			disparityPtr(y, x) = (uchar)minPos;
		}
	}

	return disparity;
}




// convert gray disparity image into a float image
// scale values by dispfact and convert 0's (unk values) into inf's 
void CDisparityHelper::disp2floatImg(cv::Mat &img, CFloatImage &fimg, int dispfact, int mapzero)
{
	//CShape sh = img.Shape();
	CShape sh;
	cv::Size sz = img.size();
	int width = sz.width, height = sz.height;
	sh.width = width;
	sh.height = height;
	sh.nBands = 1;
	fimg.ReAllocate(sh);

	float s = 1.0 / dispfact;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int v = img.at<uchar>(y, x);//img.Pixel(x, y, 0);
			float f = s * v;
			if (v == 0 && mapzero)
				f = INFINITY;
			fimg.Pixel(x, y, 0) = f;
		}
	}
}


