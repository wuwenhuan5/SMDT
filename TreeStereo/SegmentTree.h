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
#pragma once
#ifndef __SEGMENT_TREE_H__
#define __SEGMENT_TREE_H__
#include <opencv2/core/core.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#define DEF_CHAR_MAX 255
#define NUM_NEIGHBOR 4
#define MAXNUM_CHILD (NUM_NEIGHBOR - 1)
class CWeightProvider {
public:
	virtual float GetWeight(int x0, int y0, int x1, int y1) const = 0;
	virtual float GetScale() const = 0;
};

struct NodePointer {
	int id;
	uchar dist;

	NodePointer():id(0), dist(0) {}
};

struct TreeNode {
	NodePointer father;
	NodePointer children[NUM_NEIGHBOR];
	int id;
	int childrenNum;

	TreeNode():id(0), childrenNum(0) {}
};

class CSegmentTree {
private:
	float m_table[DEF_CHAR_MAX+1];
	cv::Size m_imgSize;
	std::vector<TreeNode> m_tree;
	
public:
	void BuildTree_H(cv::Size size, float sigma,  CWeightProvider &weightProvider); //横向树
	void BuildTree_V(cv::Size size, float sigma,  CWeightProvider &weightProvider); //纵向树
	void BuildTree_DU(cv::Size size, float sigma,  CWeightProvider &weightProvider); //上对角线树
	void BuildTree_DD(cv::Size size, float sigma,  CWeightProvider &weightProvider); //下对角线树
	void BuildTree_xyx2y1(cv::Size size, float sigma,  CWeightProvider &weightProvider); // 左上-右下任意对角线树，即连接(x,y)到(x+2,y+1),根号5倍的sigma
	void BuildTree_xyx1y2(cv::Size size, float sigma,  CWeightProvider &weightProvider); // 左上-右下任意对角线树，即连接(x,y)到(x+1,y+2),根号5倍的sigma
	void BuildTree_xyx_1y2(cv::Size size, float sigma, CWeightProvider &weightProvider); // 左下-右上对角线树，即连接(x,y)到(x-1,y+2),根号5倍的sigma
	void BuildTree_xyx_2y1(cv::Size size, float sigma, CWeightProvider &weightProvider); // 左下-右上对角线树，即连接(x,y)到(x-2,y+1),根号5倍的sigma
	void BuildSegmentTree(cv::Size size, float sigma, float tau, CWeightProvider &weightProvider); //分割的最小生成树
	void BuildMSTree(cv::Size size, float sigma,  CWeightProvider &weightProvider); //最小生成树
	void UpdateTable(float sigma_range);
	void CSegmentTree::UpdateTable2(float sigma_range);
	void Filter(cv::Mat costVol, int channel); // void Filter(cv::Mat &costVol, int channel); 添加引用与不添加，都表示引用同一个图像矩阵
	void Filter_gray(cv::Mat costVol); //对灰度图像进行滤波
	cv::Mat img; //自己添加，为了显示分割图像
};

class CColorWeight: public CWeightProvider {
private:
	cv::Mat img;
	cv::Mat3b imgPtr;
	cv::Mat gray;
	cv::Mat1b grayPtr;
	cv::Mat cannymask;
	cv::Mat1b cannyPtr;
	
public:
	CColorWeight(cv::Mat &img_);

	virtual float GetWeight(int x0, int y0, int x1, int y1) const;
	virtual float GetScale() const {return 1.0f;}
};

class CColorDepthWeight: public CWeightProvider {
private:
	cv::Mat img;
	cv::Mat3b imgPtr;
	cv::Mat1b disp;
	cv::Mat1b mask;
	float level;

public:
	CColorDepthWeight(cv::Mat &img_, cv::Mat &disp_, cv::Mat& mask_, int maxLevel);

	virtual float GetWeight(int x0, int y0, int x1, int y1) const;
	virtual float GetScale() const {return 255.0f;}
};


struct edge
{
	float w;
	int a, b;
	bool operator < (const edge &b) const { //定义了小于号的运算，用于排序中
		if(this->w != b.w) {
			return this->w < b.w;
		} else if(this->b != b.b) {
			return this->b < b.b;
		} else {
			return this->a < b.a;
		}
	}
};//graph edge



#endif