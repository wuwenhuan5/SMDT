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
#include <ctime>
#include "StereoDisparity.h"
#include "StereoHelper.h"
#include "Toolkit.h"
#include "SegmentTree.h"
#include "../ADCensusStereo/scanlineoptimization.h"
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>
#include "../CrossScaleStereo/TreeWMPP.h"
using namespace std;


void stereo_routine(const char *left_input, const char *right_input, const char *dispFile,
					int max_dis_level, int scale, float sigma, METHOD method) {
	cv::Mat leftImg = cv::imread(left_input);
	cv::Mat rightImg = cv::imread(right_input);
	CV_Assert(leftImg.type() == CV_8UC3 && rightImg.type() == CV_8UC3);

	cv::Mat dispImg;

	if (method == DT_RAW) {
		stereo_DT_disparity(leftImg, rightImg, dispImg, max_dis_level, scale, sigma, 0);
	}
	else if (method == DT_REFINED) {
		stereo_DT_disparity(leftImg, rightImg, dispImg, max_dis_level, scale, sigma, 1);
	}


	CDisparityHelper dispHelper;
	bool ispfm = 1;  // 对于Data3, 需要保存成pfm格式图像
	if (ispfm)
	{
		CFloatImage fdisp;
		dispHelper.disp2floatImg(dispImg, fdisp, 1.0, 0);
		WriteImageVerb(fdisp, dispFile, 0);
	}
	else
	{
		cv::imwrite(dispFile, dispImg);
	}
}

void stereo_routine_kitti(const string left_input, const string right_input, const string output,int max_dis_level, int scale, float sigma, METHOD method)
{
	if (method==DT_RAW) {
		stereo_DT_disparity_kitti(left_input, right_input, output,max_dis_level, scale, sigma, 0);
	}
	else if (method == DT_REFINED) {
		stereo_DT_disparity_kitti(left_input, right_input, output, max_dis_level, scale, sigma, 1);
	}
}

void stereo_buildTree_filter1(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel) //创建树，然后根据原始代价体，进行滤波，生成新的代价体
{
	int max_dis_level = maxLevel;
	CSegmentTree stree; //分割树
	CColorWeight cWeight(imL);//ColorWeight

	cv::Mat tempcostVol1 = costVol.clone();
	
	stree.UpdateTable(sigma);//更新查找表
	stree.BuildTree_H(imL.size(), sigma, cWeight);  // 先用横向树来滤波
	stree.Filter(tempcostVol1, max_dis_level);

	costVol =1.0* tempcostVol1 ;
}

void stereo_buildTree_filter2(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel) //创建树，然后根据原始代价体，进行滤波，生成新的代价体
{
	int max_dis_level = maxLevel;
	CSegmentTree stree; //分割树
	CColorWeight cWeight(imL);//ColorWeight

	cv::Mat tempcostVol;// = costVol.clone();

	cv::Mat tempcostVol1 = costVol.clone();
	cv::Mat tempcostVol2 = costVol.clone();
	
	stree.UpdateTable(sigma);//更新查找表
	stree.BuildTree_H(imL.size(), sigma, cWeight);  // 先用横向树来滤波
	stree.Filter(tempcostVol1, max_dis_level);

	stree.BuildTree_V(imL.size(), sigma, cWeight); // 再用纵向树来滤波
	stree.Filter(tempcostVol2, max_dis_level);

    costVol = tempcostVol1 + tempcostVol2 -costVol;

	//costVol = tempcostVol1 + tempcostVol2;
}

void stereo_buildTree_filter4(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel) //创建树，然后根据原始代价体，进行滤波，生成新的代价体
{
	int max_dis_level = maxLevel;
	CSegmentTree stree; //分割树
	CColorWeight cWeight(imL);//ColorWeight

	cv::Mat tempcostVol;// = costVol.clone();

	cv::Mat tempcostVol1 = costVol.clone();
	cv::Mat tempcostVol2 = costVol.clone();
	cv::Mat tempcostVol3 = costVol.clone();
	cv::Mat tempcostVol4 = costVol.clone();

	stree.UpdateTable(sigma);//更新查找表
	stree.BuildTree_H(imL.size(), sigma, cWeight);  // 先用横向树来滤波
	stree.Filter(tempcostVol1, max_dis_level);

	stree.BuildTree_V(imL.size(), sigma, cWeight); // 再用纵向树来滤波
	stree.Filter(tempcostVol2, max_dis_level);

	stree.BuildTree_DU(imL.size(), sigma, cWeight); // 再用对角向树来滤波
	stree.Filter(tempcostVol3, max_dis_level);

	stree.BuildTree_DD(imL.size(), sigma, cWeight); // 再用对角向树来滤波
	stree.Filter(tempcostVol4, max_dis_level);

	costVol = tempcostVol1 + tempcostVol2 + 0.5*tempcostVol3 + 0.5*tempcostVol4 - 2 * costVol;
	//costVol = tempcostVol1 + tempcostVol2 + tempcostVol3 + tempcostVol4-3 * costVol;
}

void stereo_buildTree_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel) //创建树，然后根据原始代价体，进行滤波，生成新的代价体
{
	int max_dis_level=maxLevel;
	CSegmentTree stree; //分割树
	CColorWeight cWeight(imL);//ColorWeight

	cv::Mat tempcostVol;// = costVol.clone();

	cv::Mat tempcostVol1 = costVol.clone();
	cv::Mat tempcostVol2 = costVol.clone();
	cv::Mat tempcostVol3 = costVol.clone();
	cv::Mat tempcostVol4 = costVol.clone();
	cv::Mat tempcostVol5 = costVol.clone();
	cv::Mat tempcostVol6 = costVol.clone();
	cv::Mat tempcostVol7 = costVol.clone();
	cv::Mat tempcostVol8 = costVol.clone();
	

	stree.UpdateTable(sigma);//更新查找表
	stree.BuildTree_H(imL.size(), sigma, cWeight);  // 先用横向树来滤波
	stree.Filter(tempcostVol1, max_dis_level);

	
	stree.BuildTree_V(imL.size(), sigma, cWeight); // 再用纵向树来滤波
	stree.Filter(tempcostVol2, max_dis_level);

	stree.BuildTree_DU(imL.size(), sigma, cWeight); // 再用对角向树来滤波
	stree.Filter(tempcostVol3, max_dis_level);

	stree.BuildTree_DD(imL.size(), sigma, cWeight); // 再用对角向树来滤波
	stree.Filter(tempcostVol4, max_dis_level);
	//costVol = tempcostVol1 + tempcostVol2 + tempcostVol3 + tempcostVol4; //为了四个树滤波后，再进行最小生成树滤波
	
	float sigma2 = sigma;//2.236*sigma;//// 
	stree.BuildTree_xyx2y1(imL.size(), sigma2, cWeight);
	stree.Filter(tempcostVol5, max_dis_level);
	stree.BuildTree_xyx1y2(imL.size(), sigma2, cWeight);
	stree.Filter(tempcostVol6, max_dis_level);
	stree.BuildTree_xyx_1y2(imL.size(), sigma2, cWeight);
	stree.Filter(tempcostVol7, max_dis_level);
	stree.BuildTree_xyx_2y1(imL.size(), sigma2, cWeight);
	stree.Filter(tempcostVol8, max_dis_level);
	
	
	costVol = tempcostVol1+tempcostVol2 + tempcostVol3 + tempcostVol4 + tempcostVol5 + tempcostVol6 + tempcostVol7 + tempcostVol8 - 7 * costVol ;

}

void stereo_LinearTree_filter(cv::Mat &imL, cv::Mat &costVol, float sigma_range, int channel,bool eightTree) //注意当进行视差代价聚合时，这里channel实际上就是max_dis_level，即最大视差个数
{ //结论：与MST的树，得到的结果完全相同，算法没有任何问题
	
	float weight_table[256];
	sigma_range = std::max(0.01f, sigma_range);
	for (int i = 0; i <= 255; i++) {
		weight_table[i] = exp(-float(i) / (255 * sigma_range));
	}

	Size imageSize = imL.size();	
	int height = imageSize.height;
	int width = imageSize.width;

	CColorWeight distProvider(imL);//ColorWeight，主要用于求解线性树中点与点之间的权重，即w(p,q)=w(q,p)=|I_p-I_q|
	
	KIdx_<double, 3>  costVolPtr((double *)costVol.data, imageSize.height, imageSize.width, channel);
	
	//前向聚合
	cv::Mat For_Buffer1, For_Buffer2, For_Buffer3, For_Buffer4; //主要的四个方向
	For_Buffer1 = costVol.clone();
	KIdx_<double, 3>  For_bufferPtr1((double *)For_Buffer1.data, imageSize.height, imageSize.width, channel); //水平方向1，前向从[x,y]->[x+1,y]，或者后向[x-1,y]<-[x,y]

	 For_Buffer2 = costVol.clone();
	KIdx_<double, 3>  For_bufferPtr2((double *)For_Buffer2.data, imageSize.height, imageSize.width, channel); //左上右下正对角方向2，前向从[x,y]->[x+1,y+1]，或者后向[x-1,y-1]<-[x,y]

	 For_Buffer3 = costVol.clone();
	KIdx_<double, 3>  For_bufferPtr3((double *)For_Buffer3.data, imageSize.height, imageSize.width, channel); //垂直方向3，前向从[x,y]->[x,y+1]，或者后向[x,y-1]<-[x,y]

	For_Buffer4 = costVol.clone();
	KIdx_<double, 3>  For_bufferPtr4((double *)For_Buffer4.data, imageSize.height, imageSize.width, channel); //左下右上正对角方向4，前向从[x,y]->[x+1,y-1]，或者后向[x-1,y+1]<-[x,y]

	
	cv::Mat For_Buffer5, For_Buffer6, For_Buffer7, For_Buffer8;
	
    For_Buffer5 = costVol.clone();
	KIdx_<double, 3>  For_bufferPtr5((double *)For_Buffer5.data, imageSize.height, imageSize.width, channel);  //左上右下斜对角方向5，前向从[x,y]->[x+2,y+1]，或者后向[x-2,y-1]<-[x,y]

	 For_Buffer6 = costVol.clone();
	KIdx_<double, 3>  For_bufferPtr6((double *)For_Buffer6.data, imageSize.height, imageSize.width, channel);  //左上右下斜对角方向，前向从[x,y]->[x+1,y+2]，或者后向[x-1,y-2]<-[x,y]

	 For_Buffer7 = costVol.clone();
	KIdx_<double, 3>  For_bufferPtr7((double *)For_Buffer7.data, imageSize.height, imageSize.width, channel); //左下右上斜对角方向，前向从[x,y]->[x+1,y-2]，或者后向[x-1,y+2]<-[x,y]

	 For_Buffer8 = costVol.clone();
	KIdx_<double, 3>  For_bufferPtr8((double *)For_Buffer8.data, imageSize.height, imageSize.width, channel); //左下右上斜对角方向，前向从[x,y]->[x+2,y-1]，或者后向[x-2,y+1]<-[x,y]
	
	
	//后向聚合
	cv::Mat Back_Buffer1, Back_Buffer2, Back_Buffer3, Back_Buffer4;
	Back_Buffer1 = costVol.clone();
	KIdx_<double, 3>  Back_bufferPtr1((double *)Back_Buffer1.data, imageSize.height, imageSize.width, channel); //水平方向1，前向从[x,y]->[x+1,y]，或者后向[x-1,y]<-[x,y]

	Back_Buffer2 = costVol.clone();
	KIdx_<double, 3>  Back_bufferPtr2((double *)Back_Buffer2.data, imageSize.height, imageSize.width, channel); //左上右下正对角方向2，前向从[x,y]->[x+1,y+1]，或者后向[x-1,y-1]<-[x,y]

    Back_Buffer3 = costVol.clone();
	KIdx_<double, 3>  Back_bufferPtr3((double *)Back_Buffer3.data, imageSize.height, imageSize.width, channel); //垂直方向3，前向从[x,y]->[x,y+1]，或者后向[x,y-1]<-[x,y]

	Back_Buffer4 = costVol.clone();
	KIdx_<double, 3>  Back_bufferPtr4((double *)Back_Buffer4.data, imageSize.height, imageSize.width, channel); //左下右上正对角方向4，前向从[x,y]->[x+1,y-1]，或者后向[x-1,y+1]<-[x,y]

	
	cv::Mat Back_Buffer5, Back_Buffer6, Back_Buffer7, Back_Buffer8;
	
	Back_Buffer5 = costVol.clone();
	KIdx_<double, 3>  Back_bufferPtr5((double *)Back_Buffer5.data, imageSize.height, imageSize.width, channel);  //左上右下斜对角方向5，前向从[x,y]->[x+2,y+1]，或者后向[x-2,y-1]<-[x,y]

	Back_Buffer6 = costVol.clone();
	KIdx_<double, 3>  Back_bufferPtr6((double *)Back_Buffer6.data, imageSize.height, imageSize.width, channel);  //左上右下斜对角方向，前向从[x,y]->[x+1,y+2]，或者后向[x-1,y-2]<-[x,y]

	Back_Buffer7 = costVol.clone();
	KIdx_<double, 3>  Back_bufferPtr7((double *)Back_Buffer7.data, imageSize.height, imageSize.width, channel); //左下右上斜对角方向，前向从[x,y]->[x+1,y-2]，或者后向[x-1,y+2]<-[x,y]

	Back_Buffer8 = costVol.clone();
	KIdx_<double, 3>  Back_bufferPtr8((double *)Back_Buffer8.data, imageSize.height, imageSize.width, channel); //左下右上斜对角方向，前向从[x,y]->[x+2,y-1]，或者后向[x-2,y+1]<-[x,y]
	

	cout << "开始线性树滤波..." << endl;
	//向前聚合
   // #pragma omp parallel for,慎用，用与不用结果有时存在差别
	for (int y = 0; y < height; y++) { //方向是从上至下
		for (int x = 0; x < width; x++) {//从左至右，要求与方向树的前向或后向聚合方向一致
			double *tempcost = NULL;
			
			int id = y*width + x;
					
			//树一：水平方向1，从左[x-1,y]至右[x,y]
			tempcost= &For_bufferPtr1(id * channel);
			if ((x-1) >= 0) {
				int pre_id = y*width + x - 1;
				double *pre_cost = &For_bufferPtr1(pre_id* channel); //
				double w = distProvider.GetWeight(x, y, x - 1, y);
				int dist = std::min(int( w+ 0.5f), 255);
				double weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}


			//树二：左上右下正对角方向2，从左上[x-1,y-1]至右下[x,y]
			tempcost = &For_bufferPtr2(id * channel);
			if ((x-1) >= 0&& (y-1) >= 0) {
				int pre_id = (y-1)*width + x - 1;
				double *pre_cost = &For_bufferPtr2(pre_id* channel);
                double w=distProvider.GetWeight(x, y, x - 1, y-1);
				int dist = std::min(int(w + 0.5f), 255);
				double weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}

			//树三：垂直方向3，从上[x,y-1]至下[x,y]
			tempcost = &For_bufferPtr3(id * channel);
			if ( (y-1) >= 0) {
				int pre_id = (y - 1)*width + x;
				double *pre_cost = &For_bufferPtr3(pre_id* channel);
				double w= distProvider.GetWeight(x, y, x, y - 1);
				int dist = std::min(int(w + 0.5f), 255);
				double weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}

			//树四：右上至左下正对角方向4，从右上[x+1,y-1]至左下[x,y]
			tempcost = &For_bufferPtr4(id * channel);
			if ( (x+1)<=(width-1) && (y-1) >=0 ) {
				int pre_id = (y - 1)*width + x + 1;
				double *pre_cost = &For_bufferPtr4(pre_id* channel);
				double w = distProvider.GetWeight(x, y, x + 1, y - 1);
				int dist = std::min(int(w + 0.5f), 255);
				double weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}

			if (eightTree)
			{
				//树五：左上至右下对角方向5，左上[x-2,y-1]至右下[x,y]
				tempcost = &For_bufferPtr5(id * channel);
				if ((x-2)>=0&&(y-1)>=0) {
					int pre_id = (y - 1)*width + x -2;
					double *pre_cost = &For_bufferPtr5(pre_id* channel);
					double w= distProvider.GetWeight(x, y, x - 2, y - 1);
					int dist = std::min(int(w + 0.5f), 255);
					double weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}

				//树六：左上至右下对角方向6，左上[x-1,y-2]至右下[x,y]
				tempcost = &For_bufferPtr6(id * channel);
				if ((x - 1) >= 0 && (y - 2) >= 0) {
					int pre_id = (y - 2)*width + x - 1;
					double *pre_cost = &For_bufferPtr6(pre_id* channel);
					double w= distProvider.GetWeight(x, y, x - 1, y - 2);
					int dist = std::min(int(w + 0.5f), 255);
					double weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}

				//树七：右上至左下对角方向7，右上[x+1,y-2]至左下[x,y]
				tempcost = &For_bufferPtr7(id * channel);
				if ((x + 1) <= (width - 1) && (y - 2) >= 0) {
					int pre_id = (y - 2)*width + x + 1;
					double *pre_cost = &For_bufferPtr7(pre_id* channel);
					double w= distProvider.GetWeight(x, y, x + 1, y - 2);
					int dist = std::min(int(w + 0.5f), 255);
					double weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}

					//树八：右上至左下对角方向8，右上[x+2,y-1]至右下[x,y]
					tempcost = &For_bufferPtr8(id * channel);
					if ((x + 2) <= (width - 1) && (y - 1) >= 0) {
						int pre_id = (y - 1)*width + x + 2;
						double *pre_cost = &For_bufferPtr8(pre_id* channel);
						double w= distProvider.GetWeight(x, y, x + 2, y - 1);
						int dist = std::min(int(w + 0.5f), 255);
						double weight = weight_table[dist];
						for (int k = 0; k < channel; k++) {
							tempcost[k] += pre_cost[k] * weight;
						}
				}
			}
		}
	}

   //#pragma omp parallel for,慎用，用与不用结果有时存在差别
	//向后聚合
	for (int y = height-1; y >= 0; y--) { //方向是从下至上
		for (int x = width-1; x >= 0; x--) {//从右至左，要求与方向树的前向或后向聚合方向一致
			double *tempcost = NULL;

			int id = y*width + x;

			//树一：水平方向1，从右[x+1,y]至左[x,y]
			tempcost = &Back_bufferPtr1(id * channel);
			if ((x +1)<=(width-1)) {
				int pre_id = y*width + x + 1;
				double *pre_cost = &Back_bufferPtr1(pre_id* channel);
				double w= distProvider.GetWeight(x, y, x + 1, y);
				int dist =  std::min(int(w + 0.5f), 255);
				double weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}


			//树二：右下至左上正对角方向2，从右下[x+1,y+1]至左上[x,y]
			tempcost = &Back_bufferPtr2(id * channel);
			if ((x + 1) <=(width-1)  && (y + 1) <=(height-1)) {
				int pre_id = (y + 1)*width + x + 1;
				double *pre_cost = &Back_bufferPtr2(pre_id* channel);
				double w= distProvider.GetWeight(x, y, x + 1, y + 1);
				int dist = std::min(int(w + 0.5f), 255);
				double weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}

			//树三：垂直方向3，从下[x,y+1]至上[x,y]
			tempcost = &Back_bufferPtr3(id * channel);
			if ((y + 1) <= (height-1)) {
				int pre_id = (y + 1)*width + x;
				double *pre_cost = &Back_bufferPtr3(pre_id* channel);
				double w= distProvider.GetWeight(x, y, x, y + 1);
				int dist = std::min(int(w + 0.5f), 255);
				double weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}

			//树四：左下至右上正对角方向4，从左下[x-1,y+1]至右上[x,y]
			tempcost = &Back_bufferPtr4(id * channel);
			if ((x - 1) >= 0 && (y + 1) <=(height-1)) {
				int pre_id = (y + 1)*width + x - 1;
				double *pre_cost = &Back_bufferPtr4(pre_id* channel);
				double w= distProvider.GetWeight(x, y, x - 1, y + 1);
				int dist = std::min(int(w + 0.5f), 255);
				double weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}

			if (eightTree)
			{
				//树五：右下至左上对角方向5，右下[x+2,y+1]至左上[x,y]
				tempcost = &Back_bufferPtr5(id * channel);
				if ((x + 2) <=(width-1) && (y + 1) <=(height-1)) {
					int pre_id = (y + 1)*width + x + 2;
					double *pre_cost = &Back_bufferPtr5(pre_id* channel);
					double w= distProvider.GetWeight(x, y, x + 2, y + 1);
					int dist = std::min(int(w + 0.5f), 255);
					double weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}

				//树六：右下至左上对角方向6，右下[x+1,y+2]至左上[x,y]
				tempcost = &Back_bufferPtr6(id * channel);
				if ((x + 1) <=(width-1) && (y + 2) <=(height-1)) {
					int pre_id = (y + 2)*width + x + 1;
					double *pre_cost = &Back_bufferPtr6(pre_id* channel);
					double w= distProvider.GetWeight(x, y, x + 1, y + 2);
					int dist = std::min(int(w + 0.5f), 255);
					double weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}

				//树七：左下至右上对角方向7，左下[x-1,y+2]至右上[x,y]
				tempcost = &Back_bufferPtr7(id * channel);
				if ((x - 1)>=0 && (y + 2) <=(height-1)) {
					int pre_id = (y + 2)*width + x - 1;
					double *pre_cost = &Back_bufferPtr7(pre_id* channel);
					double w= distProvider.GetWeight(x, y, x - 1, y + 2);
					int dist = std::min(int(w + 0.5f), 255);
					double weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}

				//树八：左下至右上对角方向8，左下[x-2,y+1]至右上[x,y]
				tempcost = &Back_bufferPtr8(id * channel);
				if ((x - 2) >=0 && (y + 1) <=(height-1)) {
					int pre_id = (y + 1)*width + x - 2;
					double *pre_cost = &Back_bufferPtr8(pre_id* channel);
					double w= distProvider.GetWeight(x, y, x - 2, y + 1);
					int dist = std::min(int(w + 0.5f), 255);
					double weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}
			}
		}
	}

	costVol =For_Buffer1+Back_Buffer1 + For_Buffer2 + Back_Buffer2 + For_Buffer3 + Back_Buffer3 + For_Buffer4 + Back_Buffer4 +
		For_Buffer5 + Back_Buffer5 + For_Buffer6 + Back_Buffer6 + For_Buffer7 + Back_Buffer7 + For_Buffer8 + Back_Buffer8-8*costVol-7*costVol;

	/*
    #pragma omp parallel for
	for (int y = 0; y < height; y++) { 
		for (int x = 0; x < width; x++) {
			for (int k = 0; k < channel; k++) {
				if (eightTree) {
					costVolPtr(y, x, k) = (For_bufferPtr1(y, x, k) + Back_bufferPtr1(y, x, k) - costVolPtr(y, x, k)) +
						(For_bufferPtr2(y, x, k) + Back_bufferPtr2(y, x, k) - costVolPtr(y, x, k)) +
						(For_bufferPtr3(y, x, k) + Back_bufferPtr3(y, x, k) - costVolPtr(y, x, k)) +
						(For_bufferPtr4(y, x, k) + Back_bufferPtr4(y, x, k) - costVolPtr(y, x, k)) +
						(For_bufferPtr5(y, x, k) + Back_bufferPtr5(y, x, k) - costVolPtr(y, x, k)) +
						(For_bufferPtr6(y, x, k) + Back_bufferPtr6(y, x, k) - costVolPtr(y, x, k)) +
						(For_bufferPtr7(y, x, k) + Back_bufferPtr7(y, x, k) - costVolPtr(y, x, k)) +
						(For_bufferPtr8(y, x, k) + Back_bufferPtr8(y, x, k) - costVolPtr(y, x, k))-7* costVolPtr(y, x, k);
				}
				else {
					costVolPtr(y, x, k) = (For_bufferPtr1(y, x, k) + Back_bufferPtr1(y, x, k) - costVolPtr(y, x, k)) +
						(For_bufferPtr2(y, x, k) + Back_bufferPtr2(y, x, k) - costVolPtr(y, x, k)) +
						(For_bufferPtr3(y, x, k) + Back_bufferPtr3(y, x, k) - costVolPtr(y, x, k)) +
						(For_bufferPtr4(y, x, k) + Back_bufferPtr4(y, x, k) - costVolPtr(y, x, k));
				}

			}
		}
	}
	*/
	cout << "线性树滤波结束!" << endl;
}


//以下的树是对float代价体进行滤波，并且使用了更少的1D路径
void stereo_LinearTree_filter_horizontal_vertical(cv::Mat &imL, cv::Mat &costVol, float sigma_range, int channel, bool eightTree) //注意当进行视差代价聚合时，这里channel实际上就是max_dis_level，即最大视差个数
{ //结论：与MST的树，得到的结果完全相同，算法没有任何问题

	float weight_table[256];
	sigma_range = std::max(0.01f, sigma_range);
	for (int i = 0; i <= 255; i++) {
		weight_table[i] = exp(-float(i) / (255 * sigma_range));
	}

	Size imageSize = imL.size();
	int height = imageSize.height;
	int width = imageSize.width;

	CColorWeight distProvider(imL);//ColorWeight，主要用于求解线性树中点与点之间的权重，即w(p,q)=w(q,p)=|I_p-I_q|

	KIdx_<float, 3>  costVolPtr((float *)costVol.data, imageSize.height, imageSize.width, channel);

	//前向聚合
	cv::Mat For_Buffer1, For_Buffer2, For_Buffer3, For_Buffer4; //主要的四个方向
	For_Buffer1 = costVol.clone();
	KIdx_<float, 3>  For_bufferPtr1((float *)For_Buffer1.data, imageSize.height, imageSize.width, channel); //水平方向1，前向从[x,y]->[x+1,y]，或者后向[x-1,y]<-[x,y]

	For_Buffer2 = costVol.clone();
	KIdx_<float, 3>  For_bufferPtr2((float *)For_Buffer2.data, imageSize.height, imageSize.width, channel); //左上右下正对角方向2，前向从[x,y]->[x+1,y+1]，或者后向[x-1,y-1]<-[x,y]

	For_Buffer3 = costVol.clone();
	KIdx_<float, 3>  For_bufferPtr3((float *)For_Buffer3.data, imageSize.height, imageSize.width, channel); //垂直方向3，前向从[x,y]->[x,y+1]，或者后向[x,y-1]<-[x,y]

	For_Buffer4 = costVol.clone();
	KIdx_<float, 3>  For_bufferPtr4((float *)For_Buffer4.data, imageSize.height, imageSize.width, channel); //左下右上正对角方向4，前向从[x,y]->[x+1,y-1]，或者后向[x-1,y+1]<-[x,y]


	cv::Mat For_Buffer5, For_Buffer6, For_Buffer7, For_Buffer8;

	For_Buffer5 = costVol.clone();
	KIdx_<float, 3>  For_bufferPtr5((float *)For_Buffer5.data, imageSize.height, imageSize.width, channel);  //左上右下斜对角方向5，前向从[x,y]->[x+2,y+1]，或者后向[x-2,y-1]<-[x,y]

	For_Buffer6 = costVol.clone();
	KIdx_<float, 3>  For_bufferPtr6((float *)For_Buffer6.data, imageSize.height, imageSize.width, channel);  //左上右下斜对角方向，前向从[x,y]->[x+1,y+2]，或者后向[x-1,y-2]<-[x,y]

	For_Buffer7 = costVol.clone();
	KIdx_<float, 3>  For_bufferPtr7((float *)For_Buffer7.data, imageSize.height, imageSize.width, channel); //左下右上斜对角方向，前向从[x,y]->[x+1,y-2]，或者后向[x-1,y+2]<-[x,y]

	For_Buffer8 = costVol.clone();
	KIdx_<float, 3>  For_bufferPtr8((float *)For_Buffer8.data, imageSize.height, imageSize.width, channel); //左下右上斜对角方向，前向从[x,y]->[x+2,y-1]，或者后向[x-2,y+1]<-[x,y]


																											//后向聚合
	cv::Mat Back_Buffer1, Back_Buffer2, Back_Buffer3, Back_Buffer4;
	Back_Buffer1 = costVol.clone();
	KIdx_<float, 3>  Back_bufferPtr1((float *)Back_Buffer1.data, imageSize.height, imageSize.width, channel); //水平方向1，前向从[x,y]->[x+1,y]，或者后向[x-1,y]<-[x,y]

	Back_Buffer2 = costVol.clone();
	KIdx_<float, 3>  Back_bufferPtr2((float *)Back_Buffer2.data, imageSize.height, imageSize.width, channel); //左上右下正对角方向2，前向从[x,y]->[x+1,y+1]，或者后向[x-1,y-1]<-[x,y]

	Back_Buffer3 = costVol.clone();
	KIdx_<float, 3>  Back_bufferPtr3((float *)Back_Buffer3.data, imageSize.height, imageSize.width, channel); //垂直方向3，前向从[x,y]->[x,y+1]，或者后向[x,y-1]<-[x,y]

	Back_Buffer4 = costVol.clone();
	KIdx_<float, 3>  Back_bufferPtr4((float *)Back_Buffer4.data, imageSize.height, imageSize.width, channel); //左下右上正对角方向4，前向从[x,y]->[x+1,y-1]，或者后向[x-1,y+1]<-[x,y]


	cv::Mat Back_Buffer5, Back_Buffer6, Back_Buffer7, Back_Buffer8;

	Back_Buffer5 = costVol.clone();
	KIdx_<float, 3>  Back_bufferPtr5((float *)Back_Buffer5.data, imageSize.height, imageSize.width, channel);  //左上右下斜对角方向5，前向从[x,y]->[x+2,y+1]，或者后向[x-2,y-1]<-[x,y]

	Back_Buffer6 = costVol.clone();
	KIdx_<float, 3>  Back_bufferPtr6((float *)Back_Buffer6.data, imageSize.height, imageSize.width, channel);  //左上右下斜对角方向，前向从[x,y]->[x+1,y+2]，或者后向[x-1,y-2]<-[x,y]

	Back_Buffer7 = costVol.clone();
	KIdx_<float, 3>  Back_bufferPtr7((float *)Back_Buffer7.data, imageSize.height, imageSize.width, channel); //左下右上斜对角方向，前向从[x,y]->[x+1,y-2]，或者后向[x-1,y+2]<-[x,y]

	Back_Buffer8 = costVol.clone();
	KIdx_<float, 3>  Back_bufferPtr8((float *)Back_Buffer8.data, imageSize.height, imageSize.width, channel); //左下右上斜对角方向，前向从[x,y]->[x+2,y-1]，或者后向[x-2,y+1]<-[x,y]


	cout << "开始线性树滤波..." << endl;
	//向前聚合
	// #pragma omp parallel for,慎用，用与不用结果有时存在差别
	for (int y = 0; y < height; y++) { //方向是从上至下
		for (int x = 0; x < width; x++) {//从左至右，要求与方向树的前向或后向聚合方向一致
			float *tempcost = NULL;

			int id = y*width + x;

			//树一：水平方向1，从左[x-1,y]至右[x,y]
			tempcost = &For_bufferPtr1(id * channel);
			if ((x - 1) >= 0) {
				int pre_id = y*width + x - 1;
				float *pre_cost = &For_bufferPtr1(pre_id* channel); //
				float w = distProvider.GetWeight(x, y, x - 1, y);
				int dist = std::min(int(w + 0.5f), 255);
				float weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}


			//树二：左上右下正对角方向2，从左上[x-1,y-1]至右下[x,y]
			tempcost = &For_bufferPtr2(id * channel);
			if ((x - 1) >= 0 && (y - 1) >= 0) {
				int pre_id = (y - 1)*width + x - 1;
				float *pre_cost = &For_bufferPtr2(pre_id* channel);
				float w = distProvider.GetWeight(x, y, x - 1, y - 1);
				int dist = std::min(int(w + 0.5f), 255);
				float weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}

			//树三：垂直方向3，从上[x,y-1]至下[x,y]
			tempcost = &For_bufferPtr3(id * channel);
			if ((y - 1) >= 0) {
				int pre_id = (y - 1)*width + x;
				float *pre_cost = &For_bufferPtr3(pre_id* channel);
				float w = distProvider.GetWeight(x, y, x, y - 1);
				int dist = std::min(int(w + 0.5f), 255);
				float weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}

			//树四：右上至左下正对角方向4，从右上[x+1,y-1]至左下[x,y]
			tempcost = &For_bufferPtr4(id * channel);
			if ((x + 1) <= (width - 1) && (y - 1) >= 0) {
				int pre_id = (y - 1)*width + x + 1;
				float *pre_cost = &For_bufferPtr4(pre_id* channel);
				float w = distProvider.GetWeight(x, y, x + 1, y - 1);
				int dist = std::min(int(w + 0.5f), 255);
				float weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}

			if (eightTree)
			{
				//树五：左上至右下对角方向5，左上[x-2,y-1]至右下[x,y]
				tempcost = &For_bufferPtr5(id * channel);
				if ((x - 2) >= 0 && (y - 1) >= 0) {
					int pre_id = (y - 1)*width + x - 2;
					float *pre_cost = &For_bufferPtr5(pre_id* channel);
					float w = distProvider.GetWeight(x, y, x - 2, y - 1);
					int dist = std::min(int(w + 0.5f), 255);
					float weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}

				//树六：左上至右下对角方向6，左上[x-1,y-2]至右下[x,y]
				tempcost = &For_bufferPtr6(id * channel);
				if ((x - 1) >= 0 && (y - 2) >= 0) {
					int pre_id = (y - 2)*width + x - 1;
					float *pre_cost = &For_bufferPtr6(pre_id* channel);
					float w = distProvider.GetWeight(x, y, x - 1, y - 2);
					int dist = std::min(int(w + 0.5f), 255);
					float weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}

				//树七：右上至左下对角方向7，右上[x+1,y-2]至左下[x,y]
				tempcost = &For_bufferPtr7(id * channel);
				if ((x + 1) <= (width - 1) && (y - 2) >= 0) {
					int pre_id = (y - 2)*width + x + 1;
					float *pre_cost = &For_bufferPtr7(pre_id* channel);
					float w = distProvider.GetWeight(x, y, x + 1, y - 2);
					int dist = std::min(int(w + 0.5f), 255);
					float weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}

				//树八：右上至左下对角方向8，右上[x+2,y-1]至右下[x,y]
				tempcost = &For_bufferPtr8(id * channel);
				if ((x + 2) <= (width - 1) && (y - 1) >= 0) {
					int pre_id = (y - 1)*width + x + 2;
					float *pre_cost = &For_bufferPtr8(pre_id* channel);
					float w = distProvider.GetWeight(x, y, x + 2, y - 1);
					int dist = std::min(int(w + 0.5f), 255);
					float weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}
			}
		}
	}

	//#pragma omp parallel for,,慎用，用与不用结果有时存在差别
	//向后聚合
	for (int y = height - 1; y >= 0; y--) { //方向是从下至上
		for (int x = width - 1; x >= 0; x--) {//从右至左，要求与方向树的前向或后向聚合方向一致
			float *tempcost = NULL;

			int id = y*width + x;

			//树一：水平方向1，从右[x+1,y]至左[x,y]
			tempcost = &Back_bufferPtr1(id * channel);
			if ((x + 1) <= (width - 1)) {
				int pre_id = y*width + x + 1;
				float *pre_cost = &Back_bufferPtr1(pre_id* channel);
				float w = distProvider.GetWeight(x, y, x + 1, y);
				int dist = std::min(int(w + 0.5f), 255);
				float weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}


			//树二：右下至左上正对角方向2，从右下[x+1,y+1]至左上[x,y]
			tempcost = &Back_bufferPtr2(id * channel);
			if ((x + 1) <= (width - 1) && (y + 1) <= (height - 1)) {
				int pre_id = (y + 1)*width + x + 1;
				float *pre_cost = &Back_bufferPtr2(pre_id* channel);
				float w = distProvider.GetWeight(x, y, x + 1, y + 1);
				int dist = std::min(int(w + 0.5f), 255);
				float weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}

			//树三：垂直方向3，从下[x,y+1]至上[x,y]
			tempcost = &Back_bufferPtr3(id * channel);
			if ((y + 1) <= (height - 1)) {
				int pre_id = (y + 1)*width + x;
				float *pre_cost = &Back_bufferPtr3(pre_id* channel);
				float w = distProvider.GetWeight(x, y, x, y + 1);
				int dist = std::min(int(w + 0.5f), 255);
				float weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}

			//树四：左下至右上正对角方向4，从左下[x-1,y+1]至右上[x,y]
			tempcost = &Back_bufferPtr4(id * channel);
			if ((x - 1) >= 0 && (y + 1) <= (height - 1)) {
				int pre_id = (y + 1)*width + x - 1;
				float *pre_cost = &Back_bufferPtr4(pre_id* channel);
				float w = distProvider.GetWeight(x, y, x - 1, y + 1);
				int dist = std::min(int(w + 0.5f), 255);
				float weight = weight_table[dist];
				for (int k = 0; k < channel; k++) {
					tempcost[k] += pre_cost[k] * weight;
				}
			}

			if (eightTree)
			{
				//树五：右下至左上对角方向5，右下[x+2,y+1]至左上[x,y]
				tempcost = &Back_bufferPtr5(id * channel);
				if ((x + 2) <= (width - 1) && (y + 1) <= (height - 1)) {
					int pre_id = (y + 1)*width + x + 2;
					float *pre_cost = &Back_bufferPtr5(pre_id* channel);
					float w = distProvider.GetWeight(x, y, x + 2, y + 1);
					int dist = std::min(int(w + 0.5f), 255);
					float weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}

				//树六：右下至左上对角方向6，右下[x+1,y+2]至左上[x,y]
				tempcost = &Back_bufferPtr6(id * channel);
				if ((x + 1) <= (width - 1) && (y + 2) <= (height - 1)) {
					int pre_id = (y + 2)*width + x + 1;
					float *pre_cost = &Back_bufferPtr6(pre_id* channel);
					float w = distProvider.GetWeight(x, y, x + 1, y + 2);
					int dist = std::min(int(w + 0.5f), 255);
					float weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}

				//树七：左下至右上对角方向7，左下[x-1,y+2]至右上[x,y]
				tempcost = &Back_bufferPtr7(id * channel);
				if ((x - 1) >= 0 && (y + 2) <= (height - 1)) {
					int pre_id = (y + 2)*width + x - 1;
					float *pre_cost = &Back_bufferPtr7(pre_id* channel);
					float w = distProvider.GetWeight(x, y, x - 1, y + 2);
					int dist = std::min(int(w + 0.5f), 255);
					float weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}

				//树八：左下至右上对角方向8，左下[x-2,y+1]至右上[x,y]
				tempcost = &Back_bufferPtr8(id * channel);
				if ((x - 2) >= 0 && (y + 1) <= (height - 1)) {
					int pre_id = (y + 1)*width + x - 2;
					float *pre_cost = &Back_bufferPtr8(pre_id* channel);
					float w = distProvider.GetWeight(x, y, x - 2, y + 1);
					int dist = std::min(int(w + 0.5f), 255);
					float weight = weight_table[dist];
					for (int k = 0; k < channel; k++) {
						tempcost[k] += pre_cost[k] * weight;
					}
				}
			}
		}
	}

	costVol = For_Buffer1 + Back_Buffer1 + For_Buffer3 + Back_Buffer3 - 3 * costVol;
		/*+ For_Buffer2 + Back_Buffer2   + For_Buffer4 + Back_Buffer4 +
		For_Buffer5 + Back_Buffer5 + For_Buffer6 + Back_Buffer6 + For_Buffer7 + Back_Buffer7 + For_Buffer8 + Back_Buffer8 - 8 * costVol - 7 * costVol;*/

	/*
	#pragma omp parallel for
	for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	for (int k = 0; k < channel; k++) {
	if (eightTree) {
	costVolPtr(y, x, k) = (For_bufferPtr1(y, x, k) + Back_bufferPtr1(y, x, k) - costVolPtr(y, x, k)) +
	(For_bufferPtr2(y, x, k) + Back_bufferPtr2(y, x, k) - costVolPtr(y, x, k)) +
	(For_bufferPtr3(y, x, k) + Back_bufferPtr3(y, x, k) - costVolPtr(y, x, k)) +
	(For_bufferPtr4(y, x, k) + Back_bufferPtr4(y, x, k) - costVolPtr(y, x, k)) +
	(For_bufferPtr5(y, x, k) + Back_bufferPtr5(y, x, k) - costVolPtr(y, x, k)) +
	(For_bufferPtr6(y, x, k) + Back_bufferPtr6(y, x, k) - costVolPtr(y, x, k)) +
	(For_bufferPtr7(y, x, k) + Back_bufferPtr7(y, x, k) - costVolPtr(y, x, k)) +
	(For_bufferPtr8(y, x, k) + Back_bufferPtr8(y, x, k) - costVolPtr(y, x, k))-7* costVolPtr(y, x, k);
	}
	else {
	costVolPtr(y, x, k) = (For_bufferPtr1(y, x, k) + Back_bufferPtr1(y, x, k) - costVolPtr(y, x, k)) +
	(For_bufferPtr2(y, x, k) + Back_bufferPtr2(y, x, k) - costVolPtr(y, x, k)) +
	(For_bufferPtr3(y, x, k) + Back_bufferPtr3(y, x, k) - costVolPtr(y, x, k)) +
	(For_bufferPtr4(y, x, k) + Back_bufferPtr4(y, x, k) - costVolPtr(y, x, k));
	}

	}
	}
	}
	*/
	cout << "线性树滤波结束!" << endl;
}

void stereo_DT_disparity(cv::InputArray left_image, cv::InputArray right_image, cv::OutputArray disp_,
	int max_dis_level, int scale, float sigma, bool use_nonlocal_post_processing)
{
	cv::Mat imL = left_image.getMat();
	cv::Mat imR = right_image.getMat();

	CV_Assert(imL.size() == imR.size());
	CV_Assert(imL.type() == CV_8UC3 && imR.type() == CV_8UC3);

	cv::Size imageSize = imL.size();

	disp_.create(imageSize, CV_8U);
	cv::Mat disp = disp_.getMat();

	CDisparityHelper dispHelper;

	//step 1: cost initialization
	// 调用GetMatchingCost_SGMStereo_CensusGradient需要传入路径
	//std::string leftimg = "Data/cones/J032_left.jpg";
	//std::string rightimg = "Data/cones/J032_right.jpg";
	//cv::Mat costVolLeft =dispHelper.GetMatchingCost_SGMStereo_CensusGradient(leftimg, rightimg, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_CrossScale_CensusGradient(imL, imR, max_dis_level);

	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_PatchMatchStereo_AdCensus(imL, imR, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_MeshStereo_AdCensus(imL, imR, max_dis_level); //Data3比较好
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_ADCensusStereo_AdCensus(imL, imR, max_dis_level);

	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_PatchMatchStereo_Census(imL, imR, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_MeshStereo_Census(imL, imR, max_dis_level);
    //cv::Mat costVolLeft = dispHelper.GetMatchingCost_CrossScale_Census(imL, imR, max_dis_level);

	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_MeshStereo_AdGradient(imL, imR, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_CrossScale_AdGradient(imL, imR, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_PatchMatchStereo_AdGradient(imL, imR, max_dis_level);  
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost(imL, imR, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.Get_GrayGradXGradY_Census_MatchingCost(imL, imR, max_dis_level);
	cv::Mat costVolLeft = dispHelper.Get_CensusPlusGradXGradYCensus_MatchingCost(imL, imR, max_dis_level, 0.7);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_RWR(imL, imR, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_SGMStereo_CensusGradient(imL, imR, max_dis_level);

	cout << "Cost Volume is over!" << endl;
	KIdx_<double, 3> costVolPtr((double*)costVolLeft.data, imageSize.height, imageSize.width, max_dis_level); //重新封装下costVol,便于访问
	

	//使用树对左视图的代价体进行滤波	
	
	//stereo_buildTreeH_filter(imL, costVolLeft, sigma, max_dis_level);
	//stereo_buildTreeV_filter(imL, costVolLeft, sigma, max_dis_level);

	stereo_buildTree_filter4(imL, costVolLeft, sigma, max_dis_level); //8个方向
	//stereo_buildMSTree_filter(imL, costVolLeft, sigma, max_dis_level);
	//stereo_LinearTree_filter(imL, costVolLeft, sigma, max_dis_level, true);//false
	
	cv::Mat leftdisparity = dispHelper.GetDisparity_WTA((double*)costVolLeft.data,
		imageSize.width, imageSize.height, max_dis_level);
	MeanFilter(leftdisparity, leftdisparity, 3); //得到左视图视差图

	if (use_nonlocal_post_processing)
	{
		//cv::Mat costVolRight = dispHelper.GetRightMatchingCostFromLeft(costVolLeft,imageSize.width, imageSize.height, max_dis_level);	
		//cv::Mat costVolRight = dispHelper.GetRightMatchingCost_CrossScale_AdGradient(imL, imR, max_dis_level);
		//cv::Mat costVolRight = dispHelper.GetRight_GrayGradXGradY_Census_MatchingCost(imL, imR, max_dis_level);
		//cv::Mat costVolRight = dispHelper.GetRightCensusMatchingCost(imL, imR, max_dis_level);
		cv::Mat costVolRight = dispHelper.GetRight_CensusPlusGradXGradYCensus_MatchingCost(imL, imR, max_dis_level, 0.8);
		//cv::Mat costVolRight = dispHelper.GetRightMatchingCost_RWR(imL, imR, max_dis_level);
						
		//stereo_LinearTree_filter(imR, costVolRight, sigma, max_dis_level,true);
		//stereo_buildTreeH_filter(imL, costVolLeft, sigma, max_dis_level);
		//stereo_buildTreeV_filter(imL, costVolLeft, sigma, max_dis_level);

		stereo_buildTree_filter4(imR, costVolRight, sigma, max_dis_level);

		cv::Mat rightdisparity = dispHelper.GetDisparity_WTA((double*)costVolRight.data, imageSize.width, imageSize.height, max_dis_level);
		MeanFilter(rightdisparity, rightdisparity, 3); //得到右视图视差图

		if (0) {
		//下面进行左右交叉检验
		cv::Mat mask(imageSize, CV_8U, cv::Scalar(0));
		dispHelper.Detect_occlusion_cross_check(leftdisparity, rightdisparity, mask, max_dis_level);
		cv::Mat1b maskPtr = mask;
		//重新设置代价体
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				if (maskPtr(y, x) == 1) {  // mask=1表示没遮挡				
					for (int d = 0; d < max_dis_level; d++)
					{

						costVolPtr(y, x, d) =std::min(5, std::abs(leftdisparity.at<uchar>(y, x) - d));
						//costVolPtr(y, x, d) = std::abs(leftdisparity.at<uchar>(y, x) - d);
					}
				}
				else
				{
					for (int d = 0; d < max_dis_level; d++)
					{
						costVolPtr(y, x, d) = 0; // 
					}
				}
			}
		}
		//stereo_buildTree_filter2(imL, costVolLeft, sigma, max_dis_level); // 0.5*sigma
		stereo_buildTree_filter4(imL, costVolLeft,   sigma, max_dis_level); //0.5*
		//stereo_LinearTree_filter(imL, costVolLeft, 0.5*sigma, max_dis_level, true);
		//stereo_LinearTree_filter_horizontal_vertical(imL, costVolLeft, 0.5*sigma, max_dis_level, false); //只用2个方向
		cv::Mat leftdisparity_refine(imageSize, CV_8U);
		leftdisparity_refine = dispHelper.GetDisparity_WTA((double*)costVolLeft.data, imageSize.width, imageSize.height, max_dis_level);
		MeanFilter(leftdisparity_refine, leftdisparity_refine, 3); //得到左视图视差图

		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				if (maskPtr(y, x) == 0) {  // mask=0表示遮挡				
					leftdisparity.at<uchar>(y, x) = leftdisparity_refine.at<uchar>(y, x);
				}
				else
				{
					//leftdisparity.at<uchar>(y, x) = leftdisparity_refine.at<uchar>(y, x);
				}
			}
		}
		MeanFilter(leftdisparity_refine, leftdisparity_refine, 3);
		//使用半全局优化方法
		/*for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				if (maskPtr(y, x) == 1) {  // mask=1表示没遮挡
				   for (int d = 0; d < max_dis_level; d++)
				   {
					 costVolPtr(y, x, d) =  (std::abs(leftdisparity.at<uchar>(y, x) - d));
				   }
				}
			   else
			   {
				 for (int d = 0; d < max_dis_level; d++)
				 {
					 costVolPtr(y, x, d) = 0; // += (float)(std::abs(leftdisparity.at<uchar>(y, x) - d));
				 }
			  }
		   }
		 }
		scanlineOptimization(imL, imR, (float*)costVol_backup.data, max_dis_level);
		*/
	    }
		else if(1)
		{
			TreeWMpostProcess(imL, imR, max_dis_level, 1, leftdisparity, rightdisparity, 0.06); //para  sigma/2
		}
	}

	disp = leftdisparity * scale; // scale;	
}

void stereo_DT_disparity_kitti(string left_file, string right_file, string disp_file,
	int max_dis_level, int scale, float sigma, bool use_nonlocal_post_processing)
{
	cv::Mat imL = cv::imread(left_file);
	cv::Mat imR = cv::imread(right_file);

	CV_Assert(imL.type() == CV_8UC3 && imR.type() == CV_8UC3);


	CV_Assert(imL.size() == imR.size());
	CV_Assert(imL.type() == CV_8UC3 && imR.type() == CV_8UC3);

	cv::Size imageSize = imL.size();

	cv::Mat disp_;
	disp_.create(imageSize, CV_16U);//注意这里是short类型

	CDisparityHelper dispHelper;

	//step 1: cost initialization
	// 调用GetMatchingCost_SGMStereo_CensusGradient需要传入路径
	//std::string leftimg = "Data/cones/J032_left.jpg";
	//std::string rightimg = "Data/cones/J032_right.jpg";
	//cv::Mat costVolLeft =dispHelper.GetMatchingCost_SGMStereo_CensusGradient(left_file, right_file, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_CrossScale_CensusGradient(imL, imR, max_dis_level);

	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_PatchMatchStereo_AdCensus(imL, imR, max_dis_level);

	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_ADCensusStereo_AdCensus(imL, imR, max_dis_level);

	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_PatchMatchStereo_Census(imL, imR, max_dis_level);
   // cv::Mat costVolLeft = dispHelper.GetMatchingCost_MeshStereo_Census(imL, imR, max_dis_level);
	// cv::Mat costVolLeft = dispHelper.GetMatchingCost_CrossScale_Census(imL, imR, max_dis_level);

	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_MeshStereo_AdGradient(imL, imR, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_CrossScale_AdGradient(imL, imR, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_PatchMatchStereo_AdGradient(imL, imR, max_dis_level);  
	//cv::Mat costVolLeft = dispHelper.GetCensusMatchingCost(imL, imR, max_dis_level);
	cv::Mat costVolLeft = dispHelper.Get_CensusPlusGradXGradYCensus_MatchingCost(imL, imR, max_dis_level, 0.8);

	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_MeshStereo_AdCensus(imL, imR, max_dis_level); //Data3比较好
	cout << "Cost Volume is over!" << endl;

	KIdx_<double, 3> costVolPtr((double*)costVolLeft.data, imageSize.height, imageSize.width, max_dis_level); //重新封装下costVol,便于访问



	//使用树对左视图的代价体进行滤波
	stereo_buildTree_filter4(imL, costVolLeft, sigma, max_dis_level);
	//stereo_LinearTree_filter(imL, costVol_backup, sigma, max_dis_level,1);


	cv::Mat leftdisparity = dispHelper.GetDisparity_WTA((double*)costVolLeft.data,
		imageSize.width, imageSize.height, max_dis_level);
	MeanFilter(leftdisparity, leftdisparity, 3);  //得到左视图视差图

	if (use_nonlocal_post_processing)
	{
		cv::Mat costVolRight = dispHelper.GetRightMatchingCostFromLeft(costVolLeft,imageSize.width, imageSize.height, max_dis_level);
		//cv::Mat costVolRight = dispHelper.GetRight_CensusPlusGradXGradYCensus_MatchingCost(imL, imR, max_dis_level, 0.8);
		//cv::Mat costVolRight = dispHelper.GetRightMatchingCost_MeshStereo_AdCensus(imL, imR, max_dis_level);

		stereo_buildTree_filter4(imR, costVolRight, sigma, max_dis_level); 
		//stereo_buildMSTree_filter(imR, costVolRight, sigma, max_dis_level);
		cv::Mat rightdisparity = dispHelper.GetDisparity_WTA((double*)costVolRight.data,
			imageSize.width, imageSize.height, max_dis_level);
		MeanFilter(rightdisparity, rightdisparity, 3);  //得到右视图视差图


		TreeWMpostProcess(imL, imR, max_dis_level, 1, leftdisparity, rightdisparity, 0.06);		
	}

	leftdisparity.convertTo(leftdisparity, CV_16U); //需要转换
	disp_ = leftdisparity * scale;  // scale;
	cv::imwrite(disp_file, disp_);   //写入
}

void stereo_DT_disparity_kitti2(string left_file,  string right_file,string disp_file,
	int max_dis_level, int scale, float sigma, bool use_nonlocal_post_processing)
{
	cv::Mat imL = cv::imread(left_file);
	cv::Mat imR = cv::imread(right_file);

	CV_Assert(imL.type() == CV_8UC3 && imR.type() == CV_8UC3);
	

	CV_Assert(imL.size() == imR.size());
	CV_Assert(imL.type() == CV_8UC3 && imR.type() == CV_8UC3);

	cv::Size imageSize = imL.size();

	cv::Mat disp_;
	disp_.create(imageSize, CV_16U);//注意这里是short类型
	
	CDisparityHelper dispHelper;

	//step 1: cost initialization
	// 调用GetMatchingCost_SGMStereo_CensusGradient需要传入路径
	//std::string leftimg = "Data/cones/J032_left.jpg";
	//std::string rightimg = "Data/cones/J032_right.jpg";
	//cv::Mat costVolLeft =dispHelper.GetMatchingCost_SGMStereo_CensusGradient(left_file, right_file, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_CrossScale_CensusGradient(imL, imR, max_dis_level);

	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_PatchMatchStereo_AdCensus(imL, imR, max_dis_level);
    
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_ADCensusStereo_AdCensus(imL, imR, max_dis_level);

	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_PatchMatchStereo_Census(imL, imR, max_dis_level);
   // cv::Mat costVolLeft = dispHelper.GetMatchingCost_MeshStereo_Census(imL, imR, max_dis_level);
	// cv::Mat costVolLeft = dispHelper.GetMatchingCost_CrossScale_Census(imL, imR, max_dis_level);

	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_MeshStereo_AdGradient(imL, imR, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_CrossScale_AdGradient(imL, imR, max_dis_level);
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_PatchMatchStereo_AdGradient(imL, imR, max_dis_level);  
	//cv::Mat costVolLeft = dispHelper.GetMatchingCost(imL, imR, max_dis_level);
	cv::Mat costVolLeft = dispHelper.Get_CensusPlusGradXGradYCensus_MatchingCost(imL, imR, max_dis_level, 0.8);

	//cv::Mat costVolLeft = dispHelper.GetMatchingCost_MeshStereo_AdCensus(imL, imR, max_dis_level); //Data3比较好
	cout << "Cost Volume is over!" << endl;
	
	KIdx_<double, 3> costVolPtr((double*)costVolLeft.data, imageSize.height, imageSize.width, max_dis_level); //重新封装下costVol,便于访问
	


	//使用树对左视图的代价体进行滤波
	stereo_buildTree_filter4(imL, costVolLeft, sigma, max_dis_level);
	//stereo_LinearTree_filter(imL, costVol_backup, sigma, max_dis_level,1);
	

	cv::Mat leftdisparity = dispHelper.GetDisparity_WTA((double*)costVolLeft.data,
		imageSize.width, imageSize.height, max_dis_level);
	MeanFilter(leftdisparity, leftdisparity, 3);  //得到左视图视差图

	if (use_nonlocal_post_processing)
	{
		//cv::Mat costVolRight = dispHelper.GetRightMatchingCostFromLeft(costVolLeft,imageSize.width, imageSize.height, max_dis_level);
		cv::Mat costVolRight = dispHelper.GetRight_CensusPlusGradXGradYCensus_MatchingCost(imL, imR, max_dis_level, 0.8);
		//cv::Mat costVolRight = dispHelper.GetRightMatchingCost_MeshStereo_AdCensus(imL, imR, max_dis_level);
		
		stereo_buildTree_filter4(imR, costVolRight, sigma, max_dis_level); //
		//stereo_buildMSTree_filter(imR, costVolRight, sigma, max_dis_level);
		cv::Mat rightdisparity = dispHelper.GetDisparity_WTA((double*)costVolRight.data,
			imageSize.width, imageSize.height, max_dis_level);
		MeanFilter(rightdisparity, rightdisparity, 3);  //得到右视图视差图
				


		//下面进行左右交叉检验
		cv::Mat mask(imageSize, CV_8U, cv::Scalar(0));
		dispHelper.Detect_occlusion_cross_check(leftdisparity, rightdisparity, mask, max_dis_level);		
		cv::Mat1b maskPtr = mask;
		
		//重新设置代价体
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				if (maskPtr(y, x) == 1) {  // mask=1表示没遮挡				
					for (int d = 0; d < max_dis_level; d++)
					{
						//costVolPtr(y, x, d) +=10*( std::abs(leftdisparity.at<uchar>(y, x) - d));
						costVolPtr(y, x, d) = (std::abs(leftdisparity.at<uchar>(y, x) - d));
					}
				}
				else
				{
					for (int d = 0; d < max_dis_level; d++)
					{
						costVolPtr(y, x, d) = 0; // += (float)(std::abs(leftdisparity.at<uchar>(y, x) - d));
					}
				}
			}
		}

		stereo_buildTree_filter2(imL, costVolLeft, 0.06, max_dis_level);  // 0.5*sigma
		cv::Mat leftdisparity(imageSize, CV_8U);
		leftdisparity = dispHelper.GetDisparity_WTA((double*)costVolLeft.data, imageSize.width, imageSize.height, max_dis_level);
		MeanFilter(leftdisparity, leftdisparity, 3);  //得到左视图视差图

		/*
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				if (maskPtr(y, x) == 0) {  // mask=0表示遮挡				
					leftdisparity.at<uchar>(y, x) = leftdisparity_refine.at<uchar>(y, x);
				}
				else
				{
					//leftdisparity.at<uchar>(y, x) = leftdisparity_refine.at<uchar>(y, x);
				}
			}
		}
		*/


	  /*//使用半全局优化方法
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				if (maskPtr(y, x) == 1) {  // mask=1表示没遮挡
					for (int d = 0; d < max_dis_level; d++)
					{
						costVolPtr(y, x, d) = (std::abs(leftdisparity.at<uchar>(y, x) - d));
					}
				}
				else
				{
					for (int d = 0; d < max_dis_level; d++)
					{
						costVolPtr(y, x, d) = 0; // += (float)(std::abs(leftdisparity.at<uchar>(y, x) - d));
					}
				}
			}
		}
		scanlineOptimization(imL, imR, (float*)costVol_backup.data, max_dis_level);*/
	}

	leftdisparity.convertTo(leftdisparity, CV_16U); //需要转换
	disp_ = leftdisparity * scale;  // scale;
	cv::imwrite(disp_file,disp_);   //写入
}

void scanlineOptimization(cv::Mat imL, cv::Mat imR, double *costVol,int maxLevel,int scale) //与dispHelper.GetDisparity_WTA的调用形式一样
{
	CDisparityHelper dispHelper;
	int max_dis_level = maxLevel;
	cv::Size imageSize = imL.size();
	//SGM扫描线优化，直接调用AD_Census的扫描优化算法
   uint colorDifference = 15;
   float pi1 = 0.1; float pi2 = 0.3;
   ScanlineOptimization sO(imL, imR, 0, max_dis_level - 1, colorDifference, pi1, pi2); //dmin=0, dmax=max_dis_level-1
   vector<vector<Mat>> costMaps0; costMaps0.resize(1); //设置2个代价体
   vector<Mat> *costMaps = &costMaps0[0];
  
	   (*costMaps).resize(abs(max_dis_level - 1 - 0) + 1); //第i个代价体有abs(dMax - dMin) + 1个面
	   for (size_t j = 0; j < (*costMaps).size(); j++)
	   {
		   (*costMaps)[j].create(imL.size(), CV_64F); //costMaps[i][j]的面大小是imgSize
	   }
     

   //将costVol封装成vector<Mat> *costMaps或  vector<vector<Mat> > costMaps; 双层vector，外层vector表示2个代价体（左右，右左），内层vector表示Mat矩阵，每个视差对应一个Mat
   //cv::Mat costVol(1, imageSize.area() * maxLevel, CV_32F);  //costVol的结构
   KIdx_<double, 3> costVolPtr(costVol, imageSize.height, imageSize.width, max_dis_level); //重新封装下costVol,便于访问
															
   for (int i = 0; i < max_dis_level; i++) {
	  for (int y = 0; y < imageSize.height; y++) {
		 for (int x = 0; x < imageSize.width; x++) {
			(*costMaps)[i].at<double>(y, x) = (double)(costVolPtr(y, x, i));
		 }
	 }
   }

  sO.optimization(costMaps, false);

  for (int i = 0; i < max_dis_level; i++) {
	  for (int y = 0; y < imageSize.height; y++) {
		for (int x = 0; x < imageSize.width; x++) {
			costVolPtr(y, x, i) = (*costMaps)[i].at<double>(y, x);
		}
	  }
   }

  cv::Mat disparity = dispHelper.GetDisparity_WTA(costVol,imageSize.width, imageSize.height, max_dis_level);

  MeanFilter(disparity, disparity, 3);
  disparity *= scale;   // ;
  cv::namedWindow("scan");
  cv::imshow("scan", disparity);
  cv::waitKey();
}

void stereo_buildMSTree_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel) //创建树，然后根据原始代价体，进行滤波，生成新的代价体
{
	int max_dis_level = maxLevel;
	CSegmentTree stree; //分割树
	CColorWeight cWeight(imL);//ColorWeight

	cv::Mat tempcostVol1 = costVol.clone();

	stree.UpdateTable(sigma); //更新表格

	stree.BuildMSTree(imL.size(), sigma, cWeight); // 再用最小生成树来滤波
	stree.Filter(tempcostVol1, max_dis_level);  //
	costVol = 1.0*tempcostVol1;  // costVol = tempcostVol1; //可能在后面void TreePropagate()访问costVolPtr(y, x, d)会出错
}

void stereo_buildSegmentTree_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel) //创建树，然后根据原始代价体，进行滤波，生成新的代价体
{
	int max_dis_level = maxLevel;
	CSegmentTree stree; //分割树
	CColorWeight cWeight(imL);//ColorWeight

	cv::Mat tempcostVol1 = costVol.clone();

	stree.UpdateTable(sigma); //更新表格

	stree.BuildSegmentTree(imL.size(), sigma, TAU, cWeight); // 再用最小生成树来滤波
	stree.Filter(tempcostVol1, max_dis_level);  //
	costVol = 1.0*tempcostVol1;  // costVol = tempcostVol1; //可能在后面void TreePropagate()访问costVolPtr(y, x, d)会出错
}

void stereo_buildTreeHV_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel) //创建树，然后根据原始代价体，进行滤波，生成新的代价体
{
	int max_dis_level = maxLevel;
	CSegmentTree stree; //分割树
	CColorWeight cWeight(imL);//ColorWeight

	cv::Mat tempcostVol1 = costVol.clone();
	cv::Mat tempcostVol2 = costVol.clone();



	stree.UpdateTable(sigma); //使用新的加权函数
	stree.BuildTree_H(imL.size(), sigma, cWeight);  // 先用横向树来滤波
	stree.Filter(tempcostVol1, max_dis_level);

	stree.BuildTree_V(imL.size(), sigma, cWeight); // 再用纵向树来滤波
	stree.Filter(tempcostVol2, max_dis_level);



	costVol = tempcostVol1 + tempcostVol2 - costVol;

	//costVol = 1.0*tempcostVol1;

	//为了减去原始代价的3倍
	/*cv::Size imageSize =imL.size();
	KIdx_<float, 3> costVolPtr1((float *)costVol.data, imageSize.height, imageSize.width, maxLevel);
	tempcostVol1 = lemada*(tempcostVol1+ tempcostVol2 +tempcostVol3+ tempcostVol4 )/1+ (1 - lemada)* tempcostVol5;//
	KIdx_<float, 3> costVolPtr2((float *)tempcostVol1.data, imageSize.height, imageSize.width, maxLevel);
	for (int i = 0; i < maxLevel; i++) {
	for (int y = 0; y < imageSize.height; y++) {
	for (int x = 0; x < imageSize.width; x++) {
	costVolPtr1(y, x, i) = costVolPtr2(y,x,i)-3* costVolPtr1(y, x, i);
	}
	}
	}*/

}

void stereo_buildTreeH_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel) //创建树，然后根据原始代价体，进行滤波，生成新的代价体
{
	int max_dis_level = maxLevel;
	CSegmentTree stree; //分割树
	CColorWeight cWeight(imL);//ColorWeight

	cv::Mat tempcostVol1 = costVol.clone();

	//stree.GT_UpdateTable(sigma); //使用新的加权函数
	stree.UpdateTable(sigma);
	stree.BuildTree_H(imL.size(), sigma, cWeight);  // 先用横向树来滤波
	stree.Filter(tempcostVol1, max_dis_level);


	costVol = 1.0*tempcostVol1;


}


void stereo_buildTreeV_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel) //创建树，然后根据原始代价体，进行滤波，生成新的代价体
{
	int max_dis_level = maxLevel;
	CSegmentTree stree; //分割树
	CColorWeight cWeight(imL);//ColorWeight

	cv::Mat tempcostVol1 = costVol.clone();

	//stree.GT_UpdateTable(sigma); //使用新的加权函数
	stree.UpdateTable(sigma);
	stree.BuildTree_V(imL.size(), sigma, cWeight);  // 先用横向树来滤波
	stree.Filter(tempcostVol1, max_dis_level);


	costVol = 1.0*tempcostVol1;


}

void stereo_disparity_normal(cv::InputArray left_image, cv::InputArray right_image, cv::OutputArray disp_,
	int max_dis_level, int scale, float sigma) {
	cv::Mat imL = left_image.getMat();
	cv::Mat imR = right_image.getMat();

	CV_Assert(imL.size() == imR.size());
	CV_Assert(imL.type() == CV_8UC3 && imR.type() == CV_8UC3);

	cv::Size imageSize = imL.size();

	disp_.create(imageSize, CV_8U);
	cv::Mat disp = disp_.getMat();

	CDisparityHelper dispHelper;

	//step 1: cost initialization
	// GetMatchingCost_SGMStereo_CensusGradient需要传入路径
	std::string leftimg = "Data/00/imL.png";
	std::string rightimg = "Data/00/imR.png";
	//cv::Mat costVol=dispHelper.GetMatchingCost_SGMStereo_CensusGradient(leftimg, rightimg, max_dis_level);
	cv::Mat costVol = dispHelper.GetMatchingCost_PatchMatchStereo_AdCensus(imL, imR, max_dis_level);
	//cv::Mat costVol = dispHelper.GetMatchingCost_MeshStereo_AdCensus(imL, imR, max_dis_level);
	//cv::Mat costVol = dispHelper.GetMatchingCost_PatchMatchStereo_AdGradient(imL, imR, max_dis_level); // 
	//dispHelper.GetMatchingCost_MeshStereo_AdGradient
	//cv::Mat costVol = dispHelper.GetMatchingCost(imL, imR, max_dis_level);

	//step 2: cost aggregation，使用组合树进行聚合

	CSegmentTree stree; //分割树
	CColorWeight cWeight(imL);//ColorWeight
	//char *leftimg = "Data/teddy/imL.png";
	//stree.img = cv::imread(leftimg);

	stree.img = imL.clone();// 可以改成 stree.img=imL; 只是为了显示分割图像

	cv::Mat tempcostVol1 = costVol.clone();
	cv::Mat tempcostVol2 = costVol.clone();
	cv::Mat tempcostVol3 = costVol.clone();
	cv::Mat tempcostVol4 = costVol.clone();
	cv::Mat tempcostVol5 = costVol.clone();

	stree.BuildTree_H(imL.size(), sigma, cWeight);   // 先用横向树来滤波
	stree.Filter(tempcostVol1, max_dis_level);

	stree.BuildTree_V(imL.size(), sigma, cWeight);  // 再用纵向树来滤波
	stree.Filter(tempcostVol2, max_dis_level);

	stree.BuildTree_DU(imL.size(), sigma, cWeight); // 再用对角向树来滤波
	stree.Filter(tempcostVol3, max_dis_level);

	stree.BuildTree_DD(imL.size(), sigma, cWeight); // 再用对角向树来滤波
	stree.Filter(tempcostVol4, max_dis_level);

	//costVol = tempcostVol1 + tempcostVol2 + tempcostVol3 + tempcostVol4;
	float sigma2 = 0.1;
	if (1)
	{
		stree.BuildSegmentTree(imL.size(), sigma2, TAU, cWeight);
	}
	else
	{
		stree.BuildMSTree(imL.size(), sigma2, cWeight);// 再用最小生成树来滤波
	}
	stree.Filter(tempcostVol5, max_dis_level);

	float lemada = 1;
	costVol = lemada * (tempcostVol1 + tempcostVol2) / 1 + (1 - lemada)* tempcostVol5; //;//(tempcostVol1 + tempcostVol2 + tempcostVol3 + tempcostVol4+) / 5;// tempcostVol5; //;  //;

	//这里可以SGM优化一下

   //step 3: disparity computation
	cv::Mat disparity = dispHelper.GetDisparity_WTA((double*)costVol.data,
		imageSize.width, imageSize.height, max_dis_level);

	MeanFilter(disparity, disparity, 3);

	disparity *= scale; // 
	disparity.copyTo(disp);
}


void stereo_disparity_iteration(cv::InputArray left_image, cv::InputArray right_image, cv::OutputArray disp_, 
					  int max_dis_level, int scale, float sigma) {
	cv::Mat imL = left_image.getMat();
	cv::Mat imR = right_image.getMat();
	
	CV_Assert(imL.size() == imR.size());
	CV_Assert(imL.type() == CV_8UC3 && imR.type() == CV_8UC3);

	cv::Size imageSize = imL.size();

	disp_.create(imageSize, CV_8U);
	cv::Mat disp = disp_.getMat();
	
	CDisparityHelper dispHelper;

   //start of first run
	cv::Mat costVolLeft = dispHelper.GetMatchingCost(imL, imR, max_dis_level); //dispHelper.GetMatchingCost_PatchMatchStereo_AdCensus(imL, imR, max_dis_level);
	cv::Mat costVolRight = dispHelper.GetRightMatchingCostFromLeft(costVolLeft, 
		imageSize.width, imageSize.height, max_dis_level);

	CSegmentTree stree;
	CColorWeight colorLWeight(imL), colorRWeight(imR);

	//left disparity
	stree.img = imL.clone();
	stree.BuildSegmentTree(imL.size(), SIGMA_ONE, TAU, colorLWeight);
	stree.Filter(costVolLeft, max_dis_level);
	cv::Mat disparityLeft = dispHelper.GetDisparity_WTA((double *)costVolLeft.data,
		imageSize.width, imageSize.height, max_dis_level);
	MeanFilter(disparityLeft, disparityLeft, 3);

	//right disparity
	stree.BuildSegmentTree(imR.size(), SIGMA_ONE, TAU, colorRWeight);
	stree.Filter(costVolRight, max_dis_level);
	cv::Mat disparityRight =  dispHelper.GetDisparity_WTA((double *)costVolRight.data,
		imageSize.width, imageSize.height, max_dis_level);
	MeanFilter(disparityRight, disparityRight, 3);
	
	//left-right check with right view disparity
	cv::Mat mask(imageSize, CV_8U, cv::Scalar(0));
	dispHelper.Detect_occlusion_cross_check(disparityLeft, disparityRight, mask, max_dis_level); //左右一致性检查，检测遮挡像素

    //re-segmentation and second run	
	cv::Mat costVol = dispHelper.GetMatchingCost(imL, imR, max_dis_level);
	CColorDepthWeight colorDepthWeight(imL, disparityLeft, mask, max_dis_level);
	stree.BuildSegmentTree(imL.size(), sigma, TAU, colorDepthWeight);
	stree.Filter(costVol, max_dis_level);
	cv::Mat disparity = dispHelper.GetDisparity_WTA((double *)costVol.data, 
		imageSize.width, imageSize.height, max_dis_level);
	MeanFilter(disparity, disparity, 3);

	disparity *= scale;
	disparity.copyTo(disp);
}

// image:待滤波的图像
// max_dis_level=1
// sigma:exp(-[|Ip-Iq|/sigma_)
void tree_filter_gray(cv::InputArray coclorimgArray, cv::InputArray grayimgArray,  float sigma)
{
	cv::Mat srcimg = coclorimgArray.getMat();
	cv::Size imageSize = srcimg.size();
	//CV_Assert(srcimg.type() == CV_8UC3);

	CSegmentTree stree; //分割树
	CColorWeight cWeight(srcimg); //ColorWeight

	cv::Mat gray= grayimgArray.getMat();

	cv::Mat dstimgArray(1, imageSize.area(), CV_64F);
	KIdx_<double, 2> dstimgArrayPtr((double *)dstimgArray.data, imageSize.height, imageSize.width);
	 
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				//dstimgArrayPtr(y, x, i) =(double) srcimg.at<cv::Vec3b>(y, x)[i];
				dstimgArrayPtr(y, x) = (double)(gray.at<uchar>(y, x));
			}
		}
	
	cv::Mat tempdstimg1A = dstimgArray.clone();
	cv::Mat tempdstimg2A = dstimgArray.clone();
	cv::Mat tempdstimg3A = dstimgArray.clone();
	cv::Mat tempdstimg4A = dstimgArray.clone();
	cv::Mat tempdstimg5A = dstimgArray.clone();

	//下面是创建单位图像，主要用于求解满窗双边滤波中归一化的分母
	cv::Mat UimgArray(1, imageSize.area() * 1, CV_32F);
	UimgArray.setTo(1.0f);
	//这里的KIdx_<float, 2>的float必须与cv::Mat UimgArray(1, imageSize.area() * 1,)的 CV_32F保持一致，否则用double会将2个float转成1个double;
	KIdx_<double, 2> UimgArrayPtr((double *)UimgArray.data, imageSize.height, imageSize.width); 
	
	cv::Mat tempUimg1A = UimgArray.clone();
	cv::Mat tempUimg2A = UimgArray.clone();
	cv::Mat tempUimg3A = UimgArray.clone();
	cv::Mat tempUimg4A = UimgArray.clone();
	cv::Mat tempUimg5A = UimgArray.clone();

	stree.BuildTree_H(srcimg.size(), sigma,  cWeight); // 先用横向树来滤波
	stree.Filter_gray(tempdstimg1A);
	stree.Filter_gray(tempUimg1A); 

	stree.BuildTree_V(srcimg.size(), sigma,  cWeight); // 再用纵向树来滤波
	stree.Filter_gray(tempdstimg2A);
	stree.Filter_gray(tempUimg2A);

	stree.BuildTree_DU(srcimg.size(), sigma,  cWeight); // 再用对角向树来滤波
	stree.Filter_gray(tempdstimg3A);
	stree.Filter_gray(tempUimg3A);

	stree.BuildTree_DD(srcimg.size(), sigma,  cWeight); // 再用对角向树来滤波
	stree.Filter_gray(tempdstimg4A);
	stree.Filter_gray(tempUimg4A);

	if (1)
	{
		stree.BuildSegmentTree(srcimg.size(), sigma, TAU, cWeight); // 再用最小生成树来滤波
	}
	else
	{
		stree.BuildMSTree(srcimg.size(), sigma, cWeight);
	}
	stree.Filter_gray(tempdstimg5A);
	stree.Filter_gray(tempUimg5A);

	
	float lemada = 1;
	cv::Mat dstimgArray2 = lemada* (tempdstimg1A) +tempdstimg2A + tempdstimg3A + tempdstimg4A + (1.0 - lemada)* tempdstimg5A;
	cv::Mat UimgArray2 = lemada* (tempUimg1A) +tempUimg2A + tempUimg3A + tempUimg4A + (1.0 - lemada)* tempUimg5A;

	dstimgArray2.copyTo(dstimgArray);
	UimgArray2.copyTo(UimgArray);

	//cout <<"改变后的像素"<< UimgArrayPtr(50, 50) << endl;
	
	for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {				
				dstimgArrayPtr(y, x)  = dstimgArrayPtr(y, x)/ UimgArrayPtr(y, x);
			}
		}

	//显示图像

	cv::Mat dstimg(imageSize.height, imageSize.width, CV_8UC1);
	
	for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				dstimg.at<uchar>(y, x) = (uchar)dstimgArrayPtr(y, x);
			}
	}
	
	cv::namedWindow("result");
	cv::imshow("result", dstimg);
	cv::waitKey();

}

// image:待滤波的图像
// max_dis_level=1
// sigma:exp(-[|Ip-Iq|/sigma_)
void tree_filter_color(cv::InputArray srcimgArray, int channel, float sigma) //将彩色图像img进行滤波
{
	cv::Mat srcimg = srcimgArray.getMat();
	//CV_Assert(srcimg.type() == CV_8UC3);

	cv::Size imageSize = srcimg.size();

	CSegmentTree stree; //分割树
	CColorWeight cWeight(srcimg); //ColorWeight


	cv::Mat dstimgArray(1, imageSize.area()*channel, CV_64F);
	KIdx_<double, 3> dstimgArrayPtr((double *)dstimgArray.data, imageSize.height, imageSize.width,channel);

	for(int i=0;i<channel;i++)
	{
		for (int y = 0; y < imageSize.height; y++) {
			for (int x = 0; x < imageSize.width; x++) {
				dstimgArrayPtr(y, x, i) =(double) (srcimg.at<cv::Vec3b>(y, x)[i]);				
			}
		}
	}

	cv::Mat tempdstimg1A = dstimgArray.clone();
	cv::Mat tempdstimg2A = dstimgArray.clone();
	cv::Mat tempdstimg3A = dstimgArray.clone();
	cv::Mat tempdstimg4A = dstimgArray.clone();
	cv::Mat tempdstimg5A = dstimgArray.clone();

	//下面是创建单位图像，主要用于求解满窗双边滤波中归一化的分母
	cv::Mat UimgArray(1, imageSize.area() * 1, CV_64F);
	UimgArray.setTo(1.0f);
	//这里的KIdx_<float, 2>的float必须与cv::Mat UimgArray(1, imageSize.area() * 1,)的 CV_32F保持一致，否则用double会将2个float转成1个double;
	KIdx_<double, 2> UimgArrayPtr((double *)UimgArray.data, imageSize.height, imageSize.width);

	cv::Mat tempUimg1A = UimgArray.clone();
	cv::Mat tempUimg2A = UimgArray.clone();
	cv::Mat tempUimg3A = UimgArray.clone();
	cv::Mat tempUimg4A = UimgArray.clone();
	cv::Mat tempUimg5A = UimgArray.clone();

	stree.BuildTree_H(srcimg.size(), sigma, cWeight); // 先用横向树来滤波
	stree.Filter(tempdstimg1A, channel);
	stree.Filter_gray(tempUimg1A);

	stree.BuildTree_V(srcimg.size(), sigma, cWeight); // 再用纵向树来滤波
	stree.Filter(tempdstimg2A, channel);
	stree.Filter_gray(tempUimg2A);

	stree.BuildTree_DU(srcimg.size(), sigma, cWeight); // 再用对角向树来滤波
	stree.Filter(tempdstimg3A, channel);
	stree.Filter_gray(tempUimg3A);

	stree.BuildTree_DD(srcimg.size(), sigma, cWeight); // 再用对角向树来滤波
	stree.Filter(tempdstimg4A, channel);
	stree.Filter_gray(tempUimg4A);

	if (0)
	{ 
	    stree.BuildSegmentTree(srcimg.size(), sigma, TAU, cWeight); // 再用最小生成树来滤波
	}
	else
	{
		stree.BuildMSTree(srcimg.size(), sigma,  cWeight);
	}
	stree.Filter(tempdstimg5A, channel);
	stree.Filter_gray(tempUimg5A);

	float lemada = 1;
	cv::Mat dstimgArray2 = lemada* (tempdstimg1A+tempdstimg2A + tempdstimg3A + tempdstimg4A) + (1.0 - lemada)* tempdstimg5A; //
	cv::Mat UimgArray2 = lemada* (tempUimg1A+tempUimg2A + tempUimg3A + tempUimg4A) + (1.0 - lemada)* tempUimg5A;//

	dstimgArray2.copyTo(dstimgArray);
	UimgArray2.copyTo(UimgArray);

	for (int i = 0; i<channel; i++){
	    for (int y = 0; y < imageSize.height; y++) {
		   for (int x = 0; x < imageSize.width; x++) {
			dstimgArrayPtr(y, x, i) = dstimgArrayPtr(y, x, i) / UimgArrayPtr(y, x);
		 }
	  }
	}

	//显示图像
	cv::Mat dstimg(imageSize.height, imageSize.width, CV_8UC3);

	for (int i = 0; i<channel; i++){
	  for (int y = 0; y < imageSize.height; y++) {
		 for (int x = 0; x < imageSize.width; x++) {
			dstimg.at<cv::Vec3b>(y, x)[i] = (uchar)(dstimgArrayPtr(y, x,i)+0.5);
		}
	  }
	}

	cv::namedWindow("result");
	cv::imshow("result", dstimg);
	cv::waitKey();
}


