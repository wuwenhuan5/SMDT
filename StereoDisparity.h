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

#ifndef __STEREO_DISPARITY_H__
#define __STEREO_DISPARITY_H__

#include <opencv2/core/core.hpp>
#include <string>
using namespace std;

enum METHOD {ST_RAW, ST_REFINED, OLT_RAW, OLT_REFINED};

void stereo_routine(const char *left_input, const char *right_input, const char *output, int max_dis_level, int scale, float sigma, METHOD method);
void stereo_routine_kitti(const string left_input, const string right_input,const string output,int max_dis_level, int scale, float sigma, METHOD method);

void stereo_LinearTree_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int channel, bool eightTree);
void stereo_LinearTree_filter_horizontal_vertical(cv::Mat &imL, cv::Mat &costVol, float sigma_range, int channel, bool eightTree);
void stereo_buildTree_filter1(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel);
void stereo_buildTree_filter2(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel);
void stereo_buildTree_filter4(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel);
void stereo_buildTree_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel);
void stereo_OLT_disparity(cv::InputArray left_image, cv::InputArray right_image, cv::OutputArray disp_,int max_dis_level, int scale, float sigma, bool use_nonlocal_post_processing);
void stereo_OLT_disparity_kitti(string left_file, string right_file, string disp_file,int max_dis_level, int scale, float sigma, bool use_nonlocal_post_processing);

void scanlineOptimization(cv::Mat imL, cv::Mat imR, float *costVol, int maxLevel,int scale);

void stereo_buildMSTree_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel);
void stereo_buildSegmentTree_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel);
void stereo_buildTreeHV_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel);
void stereo_buildTreeH_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel);
void stereo_buildTreeV_filter(cv::Mat &imL, cv::Mat &costVol, float sigma, int maxLevel);

void stereo_disparity_normal(cv::InputArray left_image, cv::InputArray right_image, cv::OutputArray disp,int max_dis_level, int scale, float sigma);
void stereo_disparity_iteration(cv::InputArray left_image, cv::InputArray right_image, cv::OutputArray disp,int max_dis_level, int scale, float sigma);

void tree_filter_gray(cv::InputArray coclorimgArray, cv::InputArray grayimgArray, float sigma); //将彩色图像img，经过灰度化后，然后进行滤波
void tree_filter_color(cv::InputArray srcimgArray, int channel,  float sigma); //将彩色图像img进行滤波

#endif