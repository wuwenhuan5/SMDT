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

#include "SegmentTree.h"
#include "disjoint-set.h"
#include "segment-graph.h"
#include "Toolkit.h"
#include <algorithm>
//#include "image.h"
//#include "misc.h"
//#include "filter.h"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// random color
/*rgb random_rgb() {
	rgb c;
	double r;

	c.r = (uchar)rand() % 255;
	c.g = (uchar)rand() % 255;
	c.b = (uchar)rand() % 255;

	return c;
}*/

void CSegmentTree::BuildTree_V(cv::Size size, float sigma,  CWeightProvider &weightProvider) //纵向树
{
	//UpdateTable(sigma); //更新表格
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2];// 边的总数为 height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int x = 0; x < m_imgSize.width; x++) {
		for (int y = 0; y < m_imgSize.height; y++) {
			if (y >= 1) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y - 1) * m_imgSize.width + x;
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x, y - 1);// 100000;
				edgeNum++;
			}

			
			if (y == 0 && x < m_imgSize.width - 1) {  // 最上端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + (x + 1);
				edges[edgeNum].w = 100000; // weightProvider.GetWeight(x, y, x + 1, y);  // 100000;
				edgeNum++;
			}
			
		}
	}

	//printf("edgeNum=%d\n",edgeNum);
	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // 置0
	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // 这里不分割


	//采用不同颜色输出分割图像_______	
	/*int width = m_imgSize.width;
	int height = m_imgSize.height;
	cv::Mat out(m_imgSize, CV_8UC3);
	//image<rgb> *output = new image<rgb>(width, height);

	// pick random colors for each component
	rgb *colors = new rgb[width*height];
	for (int i = 0; i < width*height; i++)
	colors[i] = random_rgb();

	for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	int comp = u->find(y * width + x);
	out.at<Vec3b>(y, x) = Vec3b(colors[comp].r, colors[comp].g, colors[comp].b);
	}
	}

	cv::imshow("out",out);
	cv::waitKey(30);*/
	//___输出结束__________

	//采用边界线的方式输出分割图像
	/*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//边界颜色
	*/

	/* //将子图分割连成单连通图
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing，将多个子图连起来
	CV_Assert(1 == u->num_sets());// 单一树
	*/


	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //树中节点，第i个像素作为第i个节点

	for (int i = 0; i < edgeNum; i++) {  //给每个节点装备其子节点
										 // if(!edges_mask[i]) continue;
										 //对于所有连接边，即edges_mask[]=255时
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa的坐标
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb的坐标

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb是pa的子节点
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa是pb的子节点
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //队列

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	//m_tree[0] = AdjTable[0]; //第0个像素，即第0个节点
	int center = 0; // ((int)(m_imgSize.width / 2.0 + 0.5) - 1);
	m_tree[0] = AdjTable[center];//坐标点[height/2-1,0]
	isVisited[center] = true;//isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) 
	{ //队列
		TreeNode &p = m_tree[start++]; //出队
		for (int i = 0; i < p.childrenNum; i++) { //广度优先遍历
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;
			
			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //装备父节点，这里节点c需要单独拿出来，因为p已经作为其父节点，不能再让其作为c的子节点了

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {				
				 if (t.children[j].id != p.id) {  //避免出现环if(t.children[j].id != true){ 
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //入队
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);
	//printf("pixelsNum=%d\n", pixelsNum);

	// CV_Assert(start == pixelsNum && end == pixelsNum); //断言，队头与队尾都在最后

	cout << "BuildTree_V is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_H(cv::Size size, float sigma,  CWeightProvider &weightProvider) { // 横向树
	//UpdateTable(sigma); //更新表格
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2];// 边的总数为 height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = 0; x < m_imgSize.width; x++) {
			if (x < m_imgSize.width - 1) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + (x + 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x + 1, y); // 边的权重
				edgeNum++;
			}

			// 可以不是连通图
			if (y >= 1 && x == 0) {  // 最左端的线
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y - 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1); //100000;
				edgeNum++;
			}
		}
	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // 置0
	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // 这里不分割

	//采用不同颜色输出分割图像_______	
	/*int width = m_imgSize.width;
	int height = m_imgSize.height;
	cv::Mat out(m_imgSize, CV_8UC3);
	//image<rgb> *output = new image<rgb>(width, height);

	// pick random colors for each component
	rgb *colors = new rgb[width*height];
	for (int i = 0; i < width*height; i++)
		colors[i] = random_rgb();

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int comp = u->find(y * width + x);
			out.at<Vec3b>(y, x) = Vec3b(colors[comp].r, colors[comp].g, colors[comp].b);
		}
	}

	cv::imshow("out",out);
	cv::waitKey(30);*/
	//___输出结束__________

	//采用边界线的方式输出分割图像
	/*cv::Vec3b color = {255,255,255};
	 DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//边界颜色
	*/

	/* //将子图分割连成单连通图
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing，将多个子图连起来
	CV_Assert(1 == u->num_sets());// 单一树
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //树中节点，第i个像素作为第i个节点

	for (int i = 0; i < edgeNum; i++) {  //给每个节点装备其子节点
		// if(!edges_mask[i]) continue;

		//对于所有连接边，即edges_mask[]=255时
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa的坐标
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb的坐标

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb是pa的子节点
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa是pb的子节点
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //队列

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	//m_tree[0] = AdjTable[0]; //第0个像素，即第0个节点
	int center = 0; // ((int)(m_imgSize.height / 2.0 + 0.5) - 1)*m_imgSize.width + 0;
	m_tree[0] = AdjTable[center];//坐标点[height/2-1,0]
	isVisited[center] = true;;//isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //队列
		TreeNode &p = m_tree[start++]; //出队

		for (int i = 0; i < p.childrenNum; i++) { //广度优先遍历
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //装备父节点，这里节点c需要单独拿出来，因为p已经作为其父节点，不能再让其作为c的子节点了

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //避免出现环
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //入队
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);
	//printf("pixelsNum=%d\n", pixelsNum);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //断言，队头与队尾都在最后
	cout << "BuildTree_H is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_DU(cv::Size size, float sigma,  CWeightProvider &weightProvider) { // 左上-右下 对角线树
	//UpdateTable(sigma); //更新表格
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2]; // 边的总数为 height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = 0; x < m_imgSize.width; x++) {
			if ((x+1 <= m_imgSize.width - 1) && ( y+1<=m_imgSize.height-1) ) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y+1) * m_imgSize.width + (x + 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x + 1, y+1); // 边的权重
				edgeNum++;
			}

			// 可以不是连通图
			if ( y + 1 <= m_imgSize.height - 1 && x == 0) {   //连接最左端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y + 1);//100000;
				edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == 0) {   //连接最上端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x + 1, y); //100000;
				edgeNum++;
			}
		}
	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // 置0

	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // 这里不分割

    //采用不同颜色输出分割图像_______	
    /*int width = m_imgSize.width;
	int height = m_imgSize.height;
	cv::Mat out(m_imgSize, CV_8UC3);
    //image<rgb> *output = new image<rgb>(width, height);

	// pick random colors for each component
	rgb *colors = new rgb[width*height];
	for (int i = 0; i < width*height; i++)
	colors[i] = random_rgb();

	for (int y = 0; y < height; y++) {
	   for (int x = 0; x < width; x++) {
			int comp = u->find(y * width + x);
			out.at<Vec3b>(y, x) = Vec3b(colors[comp].r, colors[comp].g, colors[comp].b);
		}
	}

	cv::imshow("out",out);
	cv::waitKey(30);*/
	//___输出结束__________

   //采用边界线的方式输出分割图像
   /*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//边界颜色
   */

	/* //将子图分割连成单连通图
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing，将多个子图连起来
	CV_Assert(1 == u->num_sets());// 单一树
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //树中节点，第i个像素作为第i个节点

	for (int i = 0; i < edgeNum; i++) {  //给每个节点装备其子节点
										 // if(!edges_mask[i]) continue;

										 //对于所有连接边，即edges_mask[]=255时
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa的坐标
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb的坐标

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb是pa的子节点
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa是pb的子节点
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //队列

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	m_tree[0] = AdjTable[0]; //第0个像素，即第0个节点
	isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //队列
		TreeNode &p = m_tree[start++]; //出队

		for (int i = 0; i < p.childrenNum; i++) { //广度优先遍历
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //装备父节点，这里节点c需要单独拿出来，因为p已经作为其父节点，不能再让其作为c的子节点了

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //避免出现环
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //入队
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //断言，队头与队尾都在最后
	cout << "BuildTree_DU is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_DD(cv::Size size, float sigma,  CWeightProvider &weightProvider) { // 左下-右上 对角线树
	//UpdateTable(sigma); //更新表格
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2]; // 边的总数为 height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = m_imgSize.width-1; x >=0; x--) {
			if (( x-1 >= 0 ) && (y + 1 <= m_imgSize.height - 1)) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + (x - 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x - 1, y + 1);  // 边的权重
				edgeNum++;
			}

		/*	if (y + 1 <= m_imgSize.height - 1 && x == m_imgSize.width-1) {   //连接最右端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 1000; //weightProvider.GetWeight(x, y, x, y - 1);
				edgeNum++;
			}

	    if (x + 1 <= m_imgSize.width - 1 && y == 0) {   //连接最上端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 1000; //weightProvider.GetWeight(x, y, x, y - 1);
				edgeNum++;
			}
		}   */

			/* 可以不是连通图*/
			if (y + 1 <= m_imgSize.height - 1 && x == 0) {   //连接最左端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y + 1); //100000;
				edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == m_imgSize.height - 1) {   //连接最下端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x + 1, y); //100000;
				edgeNum++;
			}
			
		}

	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // 置0

	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // 这里不分割

	//采用不同颜色输出分割图像_______	
	/*int width = m_imgSize.width;
	int height = m_imgSize.height;
	cv::Mat out(m_imgSize, CV_8UC3);
	//image<rgb> *output = new image<rgb>(width, height);

	// pick random colors for each component
	rgb *colors = new rgb[width*height];
	for (int i = 0; i < width*height; i++)
	colors[i] = random_rgb();

	for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	int comp = u->find(y * width + x);
	out.at<Vec3b>(y, x) = Vec3b(colors[comp].r, colors[comp].g, colors[comp].b);
	}
	}

	cv::imshow("out",out);
	cv::waitKey(30);*/
	//___输出结束__________

	//采用边界线的方式输出分割图像
	/*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//边界颜色
	*/

	/* //将子图分割连成单连通图
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing，将多个子图连起来
	CV_Assert(1 == u->num_sets());// 单一树
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //树中节点，第i个像素作为第i个节点

	for (int i = 0; i < edgeNum; i++) {  //给每个节点装备其子节点
										 // if(!edges_mask[i]) continue;

										 //对于所有连接边，即edges_mask[]=255时
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa的坐标
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb的坐标

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb是pa的子节点
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa是pb的子节点
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //队列

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	//m_tree[0] = AdjTable[0]; //第0个像素，即第0个节点
	int center = 0;// (int)(m_imgSize.height - 1)*m_imgSize.width + 0;
	m_tree[0] = AdjTable[center];//坐标点[height/2-1,0]
	isVisited[center] = true;;//isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //队列
		TreeNode &p = m_tree[start++]; //出队

		for (int i = 0; i < p.childrenNum; i++) { //广度优先遍历
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //装备父节点，这里节点c需要单独拿出来，因为p已经作为其父节点，不能再让其作为c的子节点了

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //避免出现环
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //入队
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //断言，队头与队尾都在最后

	cout << "BuildTree_DD is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_xyx2y1(cv::Size size, float sigma,  CWeightProvider &weightProvider) { // 左上-右下任意对角线树，即连接(x,y)到(x+2,y+1),根号5倍的sigma
	//UpdateTable(sigma); //更新表格
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();
	
	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2]; // 边的总数为 height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = 0; x < m_imgSize.width; x++) {
			if ((x + 2 <= m_imgSize.width - 1) && (y+1<= m_imgSize.height - 1)) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y+1) * m_imgSize.width + (x + 2);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x + 2, y+1); // 边的权重
				edgeNum++;
			}

			/* 可以不是连通图?*/

			
			if (y + 1 <= m_imgSize.height - 1 && x == 0) {   //连接最左端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; // weightProvider.GetWeight(x, y, x, y + 1);//1000; 
				edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == 0) {   //连接最上端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000;// weightProvider.GetWeight(x, y, x + 1, y); //1000;
				edgeNum++;
			}

			//连接第2列与第1列之间的空隙，补连接(x,y)到(x+2,y+1)存在的空隙，为了避免[0,0]与[1,0]重复算2次
			if (x == 0 && y >= 1) {
				edges[edgeNum].a = y * m_imgSize.width + x; //第1列
				edges[edgeNum].b = y * m_imgSize.width + x + 1; //第2列
				edges[edgeNum].w = 100000; // weightProvider.GetWeight(x, y, x, y + 1);//1000; 
				edgeNum++;
			}
			
		}
	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // 置0

	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // 这里不分割

	//采用不同颜色输出分割图像_______	
	/*int width = m_imgSize.width;
	int height = m_imgSize.height;
	cv::Mat out(m_imgSize, CV_8UC3);
	//image<rgb> *output = new image<rgb>(width, height);

	// pick random colors for each component
	rgb *colors = new rgb[width*height];
	for (int i = 0; i < width*height; i++)
	colors[i] = random_rgb();

	for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	int comp = u->find(y * width + x);
	out.at<Vec3b>(y, x) = Vec3b(colors[comp].r, colors[comp].g, colors[comp].b);
	}
	}

	cv::imshow("out",out);
	cv::waitKey(30);*/
	//___输出结束__________

	//采用边界线的方式输出分割图像
	/*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//边界颜色
	*/

	/* //将子图分割连成单连通图
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing，将多个子图连起来
	CV_Assert(1 == u->num_sets());// 单一树
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //树中节点，第i个像素作为第i个节点

	for (int i = 0; i < edgeNum; i++) {  //给每个节点装备其子节点
										 // if(!edges_mask[i]) continue;

										 //对于所有连接边，即edges_mask[]=255时
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa的坐标
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb的坐标

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb是pa的子节点
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa是pb的子节点
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //队列

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	m_tree[0] = AdjTable[0]; //第0个像素，即第0个节点
	isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //队列
		TreeNode &p = m_tree[start++]; //出队

		for (int i = 0; i < p.childrenNum; i++) { //广度优先遍历
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //装备父节点，这里节点c需要单独拿出来，因为p已经作为其父节点，不能再让其作为c的子节点了

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //避免出现环
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //入队
		}
	}

	//printf("start=%d\n", pixelsNum);
	//printf("end=%d\n", edgeNum);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //断言，队头与队尾都在最后
	cout << "BuildTree_xyx2y1 is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_xyx1y2(cv::Size size, float sigma,  CWeightProvider &weightProvider) { // 左上-右下任意对角线树，即连接(x,y)到(x+1,y+2),根号5倍的sigma
	//UpdateTable(sigma); //更新表格
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2]; // 边的总数为 height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = 0; x < m_imgSize.width; x++) {
			if ((x + 1 <= m_imgSize.width - 1) && (y + 2 <= m_imgSize.height - 1)) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 2) * m_imgSize.width + (x + 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x + 1, y + 2); // 边的权重
				edgeNum++;
			}

			/* 可以不是连通图?*/
			if (y + 1 <= m_imgSize.height - 1 && x == 0) {   //连接最左端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; // weightProvider.GetWeight(x, y, x, y + 1);//1000; 
				edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == 0) {   //连接最上端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000;// weightProvider.GetWeight(x, y, x + 1, y); //1000;
				edgeNum++;
			}

			//连接第1行和第2行，补连接(x,y)到(x+1,y+2)出现的缝隙
			if (y == 0 && x >= 1)
			{
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; // weightProvider.GetWeight(x, y, x, y + 1);//1000; 
				edgeNum++;
			}
			
		}
	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // 置0

	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // 这里不分割

	//采用不同颜色输出分割图像_______	
	/*int width = m_imgSize.width;
	int height = m_imgSize.height;
	cv::Mat out(m_imgSize, CV_8UC3);
	//image<rgb> *output = new image<rgb>(width, height);

	// pick random colors for each component
	rgb *colors = new rgb[width*height];
	for (int i = 0; i < width*height; i++)
	colors[i] = random_rgb();

	for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	int comp = u->find(y * width + x);
	out.at<Vec3b>(y, x) = Vec3b(colors[comp].r, colors[comp].g, colors[comp].b);
	}
	}

	cv::imshow("out",out);
	cv::waitKey(30);*/
	//___输出结束__________

	//采用边界线的方式输出分割图像
	/*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//边界颜色
	*/

	/* //将子图分割连成单连通图
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing，将多个子图连起来
	CV_Assert(1 == u->num_sets());// 单一树
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //树中节点，第i个像素作为第i个节点

	for (int i = 0; i < edgeNum; i++) {  //给每个节点装备其子节点
										 // if(!edges_mask[i]) continue;

										 //对于所有连接边，即edges_mask[]=255时
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa的坐标
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb的坐标

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb是pa的子节点
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa是pb的子节点
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //队列

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	m_tree[0] = AdjTable[0]; //第0个像素，即第0个节点
	isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //队列
		TreeNode &p = m_tree[start++]; //出队

		for (int i = 0; i < p.childrenNum; i++) { //广度优先遍历
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //装备父节点，这里节点c需要单独拿出来，因为p已经作为其父节点，不能再让其作为c的子节点了

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //避免出现环
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //入队
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //断言，队头与队尾都在最后
	cout << "BuildTree_xyx1y2 is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_xyx_1y2(cv::Size size, float sigma, CWeightProvider &weightProvider) { // 左下-右上对角线树，即连接(x,y)到(x-1,y+2),根号5倍的sigma
	//UpdateTable(sigma); //更新表格
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2]; // 边的总数为 height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = m_imgSize.width - 1; x >= 0; x--) {
			if ((x - 1 >= 0) && (y + 2 <= m_imgSize.height - 1)) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 2) * m_imgSize.width + (x - 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x - 1, y + 2);  // 边的权重
				edgeNum++;
			}

			if (y + 1 <= m_imgSize.height - 1 && x == m_imgSize.width-1) {   //连接最右端的边
			edges[edgeNum].a = y * m_imgSize.width + x;
			edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
			edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1);
			edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == 0) {   //连接最上端的边
			edges[edgeNum].a = y * m_imgSize.width + x;
			edges[edgeNum].b = y * m_imgSize.width + x + 1;
			edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1);
			edgeNum++;
			}

			//连接第1行和第2行，补缝隙，x <m_imgSize.width - 1是为了避免[m_imgSize.width - 1,0]与[m_imgSize.width - 1,1]的重复
			if (y ==0 && x < m_imgSize.width - 1) {  
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1);
				edgeNum++;
			}
			
		  

			/*
			// 可以不是连通图
			if (y + 1 <= m_imgSize.height - 1 && x == 0) {   //连接最左端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000;// weightProvider.GetWeight(x, y, x, y + 1); //1000;
				edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == m_imgSize.height - 1) {   //连接最下端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000;// weightProvider.GetWeight(x, y, x + 1, y); //1000;
				edgeNum++;
			}
			*/
			
		} 

	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // 置0

	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // 这里不分割

	//采用不同颜色输出分割图像_______	
	/*int width = m_imgSize.width;
	int height = m_imgSize.height;
	cv::Mat out(m_imgSize, CV_8UC3);
	//image<rgb> *output = new image<rgb>(width, height);

	// pick random colors for each component
	rgb *colors = new rgb[width*height];
	for (int i = 0; i < width*height; i++)
	colors[i] = random_rgb();

	for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	int comp = u->find(y * width + x);
	out.at<Vec3b>(y, x) = Vec3b(colors[comp].r, colors[comp].g, colors[comp].b);
	}
	}

	cv::imshow("out",out);
	cv::waitKey(30);*/
	//___输出结束__________

	//采用边界线的方式输出分割图像
	/*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//边界颜色
	*/

	/* //将子图分割连成单连通图
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing，将多个子图连起来
	CV_Assert(1 == u->num_sets());// 单一树
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //树中节点，第i个像素作为第i个节点

	for (int i = 0; i < edgeNum; i++) {  //给每个节点装备其子节点
										 // if(!edges_mask[i]) continue;

										 //对于所有连接边，即edges_mask[]=255时
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa的坐标
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb的坐标

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb是pa的子节点
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa是pb的子节点
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //队列

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	//m_tree[0] = AdjTable[0]; //第0个像素，即第0个节点
	int center = 0;// (int)(m_imgSize.width - 1); //右上角
	m_tree[0] = AdjTable[center];//坐标点[height/2-1,0]
	isVisited[center] = true;;//isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //队列
		TreeNode &p = m_tree[start++]; //出队

		for (int i = 0; i < p.childrenNum; i++) { //广度优先遍历
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //装备父节点，这里节点c需要单独拿出来，因为p已经作为其父节点，不能再让其作为c的子节点了

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //避免出现环
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //入队
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //断言，队头与队尾都在最后

	cout<<"BuildTree_xyx_1y2 is over"<<endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_xyx_2y1(cv::Size size, float sigma, CWeightProvider &weightProvider) { // 左下-右上对角线树，即连接(x,y)到(x-2,y+1),根号5倍的sigma
	//UpdateTable(sigma); //更新表格
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2]; // 边的总数为 height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = m_imgSize.width - 1; x >= 0; x--) {
			if ((x - 2 >= 0) && (y + 1 <= m_imgSize.height - 1)) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + (x - 2);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x - 2, y + 1);  // 边的权重
				edgeNum++;
			}

			if (y + 1 <= m_imgSize.height - 1 && x == m_imgSize.width-1) {   //连接最右端的边
			edges[edgeNum].a = y * m_imgSize.width + x;
			edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
			edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1);
			edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == 0) {   //连接最上端的边
			edges[edgeNum].a = y * m_imgSize.width + x;
			edges[edgeNum].b = y * m_imgSize.width + x + 1;
			edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1);
			edgeNum++;
			}
			 
			//连接(x,y)到(x-2,y+1)会出现缝隙，连接倒数第2列和最后1列的缝隙，y != 0避免点[m_imgSize.width - 2,0]与[m_imgSize.width - 1,0]重复连接
			if (x + 1 == m_imgSize.width - 1 && y != 0) {   //连接最上端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1);
				edgeNum++;
			}

			/* 可以不是连通图
			if (y + 1 <= m_imgSize.height - 1 && x == 0) {   //连接最左端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000;// weightProvider.GetWeight(x, y, x, y + 1); //1000;
				edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == m_imgSize.height - 1) {   //连接最下端的边
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000;// weightProvider.GetWeight(x, y, x + 1, y); //1000;
				edgeNum++;
			}
			*/
		}

	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // 置0

	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // 这里不分割

	//采用不同颜色输出分割图像_______	基于图的分割算法中绘制的不同颜色块分割
	/*int width = m_imgSize.width;
	int height = m_imgSize.height;
	cv::Mat out(m_imgSize, CV_8UC3);
	//image<rgb> *output = new image<rgb>(width, height);

	// pick random colors for each component
	rgb *colors = new rgb[width*height];
	for (int i = 0; i < width*height; i++)
	colors[i] = random_rgb();

	for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	int comp = u->find(y * width + x);
	out.at<Vec3b>(y, x) = Vec3b(colors[comp].r, colors[comp].g, colors[comp].b);
	}
	}

	cv::imshow("out",out);
	cv::waitKey(30);*/
	//___输出结束__________

	//采用边界线的方式输出分割图像
	/*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//边界颜色
	*/

	/* //将子图分割连成单连通图
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing，将多个子图连起来
	CV_Assert(1 == u->num_sets());// 单一树
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //树中节点，第i个像素作为第i个节点

	for (int i = 0; i < edgeNum; i++) {  //给每个节点装备其子节点
										 // if(!edges_mask[i]) continue;

										 //对于所有连接边，即edges_mask[]=255时
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa的坐标
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb的坐标

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb是pa的子节点
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa是pb的子节点
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //队列

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	//m_tree[0] = AdjTable[0]; //第0个像素，即第0个节点
	int center = 0;// m_imgSize.width - 1; //(int)(m_imgSize.height - 1)*m_imgSize.width + 0;
	m_tree[0] = AdjTable[center];//坐标点[height/2-1,0]
	isVisited[center] = true;;//isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //队列
		TreeNode &p = m_tree[start++]; //出队

		for (int i = 0; i < p.childrenNum; i++) { //广度优先遍历
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //装备父节点，这里节点c需要单独拿出来，因为p已经作为其父节点，不能再让其作为c的子节点了

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //避免出现环
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //入队
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //断言，队头与队尾都在最后
	cout<<"BuildTree_xyx_2y1 is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildSegmentTree(cv::Size size, float sigma, float tau, CWeightProvider &weightProvider)  // 分割基础上的最小生成树
{
	UpdateTable(sigma); //更新表格
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2];// 边的总数为 height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = 0; x < m_imgSize.width; x++) {
			if (x < m_imgSize.width - 1) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + (x + 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x + 1, y); // 边的权重
				edgeNum++;
			}

			if (y >= 1) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y - 1) * m_imgSize.width + x;
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x, y - 1);
				edgeNum++;
			}
		}
	}

	uchar *edges_mask = new uchar[edgeNum];
	memset(edges_mask, 0, sizeof(uchar) * edgeNum); // 置0
	universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // 分割成多个子图

	//采用边界线的方式输出分割图像
	cv::Vec3b color = { 255,255,255 };
	DrawContoursAroundSegments(img, u, m_imgSize.width, m_imgSize.height, color /*边界颜色*/);


	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing，将多个子图连起来，这样将所有的边选出来了，并且个数为像素个数-1
	CV_Assert(1 == u->num_sets());// 单一树图

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //树中节点，第i个像素作为第i个节点

	for (int i = 0; i < edgeNum; i++) {  //给每个节点装备其子节点
		if (!edges_mask[i]) continue;

		//对于所有连接边，即edges_mask[]=255时
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa的坐标
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb的坐标

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb是pa的子节点
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa是pb的子节点
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //队列

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	m_tree[0] = AdjTable[0]; //第0个像素，即第0个节点
	isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //队列
		TreeNode &p = m_tree[start++]; //出队

		for (int i = 0; i < p.childrenNum; i++) { //广度优先遍历
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //装备父节点，这里节点c需要单独拿出来，因为p已经作为其父节点，不能再让其作为c的子节点了

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //避免出现环
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //入队
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	CV_Assert(start == pixelsNum && end == pixelsNum); //断言，队头与队尾都在最后
	cout << "BuildSegmentTree is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	delete u;
	delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildMSTree(cv::Size size, float sigma,  CWeightProvider &weightProvider)  //最小生成树
{
	UpdateTable(sigma); //更新表格
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2];// 边的总数为 height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = 0; x < m_imgSize.width; x++) {
			if (x < m_imgSize.width - 1) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + (x + 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x + 1, y); // 边的权重
				edgeNum++;
			}

			if (y >= 1) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y - 1) * m_imgSize.width + x;
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x, y - 1);
				edgeNum++;
			}
		}
	}

	uchar *edges_mask = new uchar[edgeNum];
	memset(edges_mask, 0, sizeof(uchar) * edgeNum);  // 置0
	universe *u = MST_graph(pixelsNum, edgeNum, edges, edges_mask); //已经选入了边

																	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //树中节点，第i个像素作为第i个节点

	for (int i = 0; i < edgeNum; i++) {  //给每个节点装备其子节点
		if (!edges_mask[i]) continue;

		//对于所有连接边，即edges_mask[]=255时
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa的坐标
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb的坐标

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb是pa的子节点
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa是pb的子节点
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //队列

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	m_tree[0] = AdjTable[0]; //第0个像素，即第0个节点
	isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //队列
		TreeNode &p = m_tree[start++]; //出队

		for (int i = 0; i < p.childrenNum; i++) { //广度优先遍历
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //装备父节点，这里节点c需要单独拿出来，因为p已经作为其父节点，不能再让其作为c的子节点了

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //避免出现环
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //入队
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	CV_Assert(start == pixelsNum && end == pixelsNum); //断言，队头与队尾都在最后
	cout<<"BuildMSTree is over!"<< endl;

	delete[]isVisited;
	delete[]AdjTable;
	delete u;
	delete[]edges_mask;
	delete[]edges;

	return;

}


void CSegmentTree::UpdateTable(float sigma_range) {
	sigma_range = std::max(0.01f, sigma_range);
	for (int i = 0; i <= DEF_CHAR_MAX; i++) {
		m_table[i] = exp(-float(i) / (DEF_CHAR_MAX * sigma_range));
	}
}

void CSegmentTree::UpdateTable2(float sigma_range) {
	sigma_range = std::max(0.01f, sigma_range);
	for (int i = 0; i <= DEF_CHAR_MAX; i++) {
		m_table[i] = 0.5*exp(-float(i) / (DEF_CHAR_MAX * sigma_range));
	}
}

void CSegmentTree::Filter(cv::Mat costVol, int channel) { // channel是视差数量
	cv::Mat costBuffer = costVol.clone();

	KIdx_<double, 3>  costPtr((double *)costVol.data, m_imgSize.height, m_imgSize.width, channel);
	KIdx_<double, 3>  bufferPtr((double *)costBuffer.data, m_imgSize.height, m_imgSize.width, channel);


	int pixelsNum = m_imgSize.area();
	//first pass: from leaf to root，从叶子节点到根节点,这里保证最后一个点是叶子节点，0为根节点
	for (int i = pixelsNum - 1; i >= 0; i--) {
		TreeNode &node = m_tree[i];
		double *cost = &bufferPtr(node.id * channel);
		for (int z = 0; z < node.childrenNum; z++) { //判断是否有子节点			
			double *child_cost = &bufferPtr(node.children[z].id * channel);
			double weight = m_table[node.children[z].dist];
			for (int k = 0; k < channel; k++) {
				cost[k] += child_cost[k] * weight;
			}
		}
	}

	
	//second pass: from root to leaf，从根节点到叶子节点
	memcpy(&costPtr(0), &bufferPtr(0), sizeof(double) * channel); //根节点
	for (int i = 1; i < pixelsNum; i++) { //从1开始，需要用到fater节点
		TreeNode &node = m_tree[i];
		double *final_cost = &costPtr(node.id * channel);
		double *cur_cost = &bufferPtr(node.id * channel);
		double *father_cost = &costPtr(node.father.id * channel);
		double weight = m_table[node.father.dist];
		for (int k = 0; k < channel; k++) {
			final_cost[k] = weight * (father_cost[k] - weight * cur_cost[k]) + cur_cost[k];
		}
	}
	
	cout << "Tree Filter is over!"<< endl;
}

void CSegmentTree::Filter_gray(cv::Mat costVol) {  //测试灰度图像， channel=1是通道数量
	cv::Mat costBuffer = costVol.clone();
	
	// KIdx_<float, 3>  costPtr((float *)costVol.data, m_imgSize.height, m_imgSize.width, channel);
	// KIdx_<float, 3>  bufferPtr((float *)costBuffer.data, m_imgSize.height, m_imgSize.width, channel);
     	
	KIdx_<double, 2>  costPtr((double *)costVol.data, m_imgSize.height, m_imgSize.width);
	KIdx_<double, 2>  bufferPtr((double *)costBuffer.data, m_imgSize.height, m_imgSize.width);
	

	int pixelsNum = m_imgSize.area();
	//first pass: from leaf to root，从叶子节点到根节点
	for (int i = pixelsNum - 1; i >= 0; i--) {
		TreeNode &node = m_tree[i];
		double *cost = &bufferPtr(node.id );
		for (int z = 0; z < node.childrenNum; z++) { //判断是否有子节点
			double *child_cost = &bufferPtr(node.children[z].id );
			double weight = m_table[node.children[z].dist];			
				*cost += (*child_cost) * weight;			
		}
	}

	//second pass: from root to leaf，从根节点到叶子节点
	memcpy(&costPtr(0), &bufferPtr(0), sizeof(double)); //根节点
	for (int i = 1; i < pixelsNum; i++) { //从1开始，需要用到fater节点
		TreeNode &node = m_tree[i];
		double *final_cost = &costPtr(node.id );
		double *cur_cost = &bufferPtr(node.id );
		double *father_cost = &costPtr(node.father.id );
		double weight = m_table[node.father.dist];
		
		*final_cost = weight * (*father_cost-weight*(*cur_cost)) + *cur_cost;
		
	}
}

CColorWeight::CColorWeight(cv::Mat &img_) {
	img = img_.clone();
	MeanFilter(img, img, 1); //中值滤波
	imgPtr = img;
	cannymask.create(img.size(), CV_8U);
	gray.create(img.size(), CV_8U);
	grayPtr = gray;
	cannyPtr = cannymask;
	cvtColor(img, gray, COLOR_BGR2GRAY); //注意img是BGR
	blur(gray, cannymask, Size(3, 3));
	Canny(cannymask, cannymask, 20, 60, 3); //注意Canny算子的参数设置
}

float CColorWeight::GetWeight(int x0, int y0, int x1, int y1) const {
	 
	//return (float)(std::abs(imgPtr(y0, x0)[0] - imgPtr(y1, x1)[0])+ std::abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1])+ std::abs(imgPtr(y0, x0)[2] - imgPtr(y1, x1)[2]))/3.0;	
	
	return (float)std::max(std::max(abs(imgPtr(y0, x0)[0] - imgPtr(y1, x1)[0]), abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1])), abs(imgPtr(y0, x0)[2] - imgPtr(y1, x1)[2])); //使用max()函数*/
	
	/*
	return (float)std::sqrt((std::abs(imgPtr(y0, x0)[0] - imgPtr(y1, x1)[0])*std::abs(imgPtr(y0, x0)[0] - imgPtr(y1, x1)[0]) +
		std::abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1])*std::abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1]) +
		std::abs(imgPtr(y0, x0)[2] - imgPtr(y1, x1)[2]))*std::abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1])); // 3.0
    */
   
	/*
	float t = (float)std::max(
		std::max(abs(imgPtr(y0, x0)[0] - imgPtr(y1, x1)[0]), abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1])), abs(imgPtr(y0, x0)[2] - imgPtr(y1, x1)[2]) //使用max()函数
	);
	float w, thre = 4.0;
	if (cannyPtr(y0, x0) != cannyPtr(y1, x1)) {
		w = t;//abs(grayPtr(y0, x0) - grayPtr(y1, x1));
	}
	else {
		w = std::min(t, thre);//float(abs(grayPtr(y0, x0) - grayPtr(y1, x1)))
	}	
	return w;
	*/
}

CColorDepthWeight::CColorDepthWeight(cv::Mat &img_, cv::Mat &disp_, cv::Mat& mask_, int maxLevel) :
	disp(disp_), mask(mask_) {
	img = img_.clone();
	MeanFilter(img, img, 1);
	imgPtr = img;
	level = (float)maxLevel;
}

float CColorDepthWeight::GetWeight(int x0, int y0, int x1, int y1) const {
#define ALPHA_DEP_SEG 0.5f
	if (mask(y0, x0) && mask(y1, x1)) {
		float dispValue = abs(disp(y0, x0) - disp(y1, x1)) / level;
		float colorValue = std::max(
			std::max(abs(imgPtr(y0, x0)[0] - imgPtr(y1, x1)[0]), abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1])),
			abs(imgPtr(y0, x0)[2] - imgPtr(y1, x1)[2])
			) / 255.0f;
		return (ALPHA_DEP_SEG * dispValue + (1.0f - ALPHA_DEP_SEG) * colorValue);
	}
	else {
		return (float)std::max(
			std::max(abs(imgPtr(y0, x0)[0] - imgPtr(y1, x1)[0]), abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1])),
			abs(imgPtr(y0, x0)[2] - imgPtr(y1, x1)[2])
			) / 255.0f;
	}
}