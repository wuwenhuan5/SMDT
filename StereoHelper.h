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

#ifndef __STEREO_HELPER__
#define __STEREO_HELPER__

#include <opencv2/core/core.hpp>
#include "../MeshStereo/MatchingCost.h"
#include "../imageLib/imageLib.h"

class CDisparityHelper {
public:
	cv::Mat CDisparityHelper::GetCensusMatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel); //灰度Census变换
	cv::Mat CDisparityHelper::GetRightCensusMatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel); //计算右代价体

	cv::Mat CDisparityHelper::GetMatchingCost_CrossScale_AdGradient(cv::Mat imL, cv::Mat imR, int maxLevel);
	cv::Mat CDisparityHelper::GetRightMatchingCost_CrossScale_AdGradient(const cv::Mat imL, const cv::Mat imR, int maxLevel);

	cv::Mat CDisparityHelper::Get_GradXGradY_Census_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel); //GradXGradY的Census变换级联
	cv::Mat CDisparityHelper::GetRight_GradXGradY_Census_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel); //计算右代价体

	cv::Mat CDisparityHelper::Get_GrayGradXGradY_Census_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel); //GrayGradXGradY的Census变换级联
	cv::Mat CDisparityHelper::GetRight_GrayGradXGradY_Census_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel);//计算右代价体

	cv::Mat CDisparityHelper::Get_CensusPlusGradXGradYCensus_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel, float th); //Gray+GradXGradY的Census变换加权和
	cv::Mat CDisparityHelper::GetRight_CensusPlusGradXGradYCensus_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel, float th);//计算右代价体

	cv::Mat CDisparityHelper::Get_AD_Census_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel); //AD+Census
	cv::Mat CDisparityHelper::GetRight_AD_Census_MatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel);
	//--------------------------------

	cv::Mat GetGradient(cv::InputArray image); //使用ST的Gradiend计算代价体,和MST的代码类似
	cv::Mat GetMatchingCost(cv::Mat imL, cv::Mat imR, int maxLevel); //使用ST的ADGradiend计算代价体
	//-------------------------------

	cv::Mat CDisparityHelper::GetMatchingCost_MeshStereo_AdGradient(cv::Mat imL, cv::Mat imR, int maxLevel); //使用MeshStero的ADGradiend计算代价体
	cv::Mat CDisparityHelper::GetMatchingCost_PatchMatchStereo_AdGradient(cv::Mat imL, cv::Mat imR, int maxLevel);

	cv::Mat CDisparityHelper::GetMatchingCost_MeshStereo_AdCensus(cv::Mat imL, cv::Mat imR, int maxLevel); //使用MeshStero的AdCensus计算代价体
	cv::Mat CDisparityHelper::GetRightMatchingCost_MeshStereo_AdCensus(cv::Mat imL, cv::Mat imR, int maxLevel);

	cv::Mat CDisparityHelper::GetMatchingCost_PatchMatchStereo_AdCensus(cv::Mat imL, cv::Mat imR, int maxLevel);
	cv::Mat CDisparityHelper::GetMatchingCost_ADCensusStereo_AdCensus(cv::Mat imL, cv::Mat imR, int maxLevel);
	
	cv::Mat CDisparityHelper::GetMatchingCost_SGMStereo_CensusGradient(std::string &filePathImageL, std::string &filePathImageR, int maxLevel); //使用SGMStereo的CensusGradient计算代价体
	cv::Mat CDisparityHelper::GetMatchingCost_SGMStereo_CensusGradient(const cv::Mat imL, const cv::Mat imR, int maxLevel);
	cv::Mat CDisparityHelper::GetMatchingCost_CrossScale_CensusGradient(cv::Mat imL, cv::Mat imR, int maxLevel);
	
	cv::Mat CDisparityHelper::GetMatchingCost_PatchMatchStereo_Census(cv::Mat imL, cv::Mat imR, int maxLevel);
	cv::Mat CDisparityHelper::GetMatchingCost_MeshStereo_Census(cv::Mat imL, cv::Mat imR, int maxLevel);
	cv::Mat CDisparityHelper::GetMatchingCost_CrossScale_Census(cv::Mat imL, cv::Mat imR, int maxLevel);
	cv::Mat CDisparityHelper::GetMatchingCost_RWR(cv::Mat imL, cv::Mat imR, int maxLevel);
	cv::Mat CDisparityHelper::GetRightMatchingCost_RWR(cv::Mat imL, cv::Mat imR, int maxLevel);


	cv::Mat CDisparityHelper::GetCensusMatchingCost_kongdong(cv::Mat imL, cv::Mat imR, int maxLevel);
    //--------------------------------

	cv::Mat GetDisparity_WTA(double *costVol, int w, int h, int maxLevel);
	cv::Mat GetRightMatchingCostFromLeft(cv::Mat leftVol, int w, int h, int maxLevel);
	void CDisparityHelper::Detect_occlusion_cross_check(cv::Mat &DispL, cv::Mat &DispR, cv::Mat &Mask, int maxLevel);
	void CDisparityHelper::disp2floatImg(cv::Mat &img, CFloatImage &fimg, int dispfact, int mapzero);
};

#endif