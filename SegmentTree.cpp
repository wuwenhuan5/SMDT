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

void CSegmentTree::BuildTree_V(cv::Size size, float sigma,  CWeightProvider &weightProvider) //������
{
	//UpdateTable(sigma); //���±��
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2];// �ߵ�����Ϊ height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int x = 0; x < m_imgSize.width; x++) {
		for (int y = 0; y < m_imgSize.height; y++) {
			if (y >= 1) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y - 1) * m_imgSize.width + x;
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x, y - 1);// 100000;
				edgeNum++;
			}

			
			if (y == 0 && x < m_imgSize.width - 1) {  // ���϶˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + (x + 1);
				edges[edgeNum].w = 100000; // weightProvider.GetWeight(x, y, x + 1, y);  // 100000;
				edgeNum++;
			}
			
		}
	}

	//printf("edgeNum=%d\n",edgeNum);
	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // ��0
	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // ���ﲻ�ָ�


	//���ò�ͬ��ɫ����ָ�ͼ��_______	
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
	//___�������__________

	//���ñ߽��ߵķ�ʽ����ָ�ͼ��
	/*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//�߽���ɫ
	*/

	/* //����ͼ�ָ����ɵ���ͨͼ
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing���������ͼ������
	CV_Assert(1 == u->num_sets());// ��һ��
	*/


	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //���нڵ㣬��i��������Ϊ��i���ڵ�

	for (int i = 0; i < edgeNum; i++) {  //��ÿ���ڵ�װ�����ӽڵ�
										 // if(!edges_mask[i]) continue;
										 //�����������ӱߣ���edges_mask[]=255ʱ
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa������
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb������

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb��pa���ӽڵ�
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa��pb���ӽڵ�
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //����

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	//m_tree[0] = AdjTable[0]; //��0�����أ�����0���ڵ�
	int center = 0; // ((int)(m_imgSize.width / 2.0 + 0.5) - 1);
	m_tree[0] = AdjTable[center];//�����[height/2-1,0]
	isVisited[center] = true;//isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) 
	{ //����
		TreeNode &p = m_tree[start++]; //����
		for (int i = 0; i < p.childrenNum; i++) { //������ȱ���
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;
			
			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //װ�����ڵ㣬����ڵ�c��Ҫ�����ó�������Ϊp�Ѿ���Ϊ�丸�ڵ㣬������������Ϊc���ӽڵ���

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {				
				 if (t.children[j].id != p.id) {  //������ֻ�if(t.children[j].id != true){ 
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //���
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);
	//printf("pixelsNum=%d\n", pixelsNum);

	// CV_Assert(start == pixelsNum && end == pixelsNum); //���ԣ���ͷ���β�������

	cout << "BuildTree_V is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_H(cv::Size size, float sigma,  CWeightProvider &weightProvider) { // ������
	//UpdateTable(sigma); //���±��
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2];// �ߵ�����Ϊ height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = 0; x < m_imgSize.width; x++) {
			if (x < m_imgSize.width - 1) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + (x + 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x + 1, y); // �ߵ�Ȩ��
				edgeNum++;
			}

			// ���Բ�����ͨͼ
			if (y >= 1 && x == 0) {  // ����˵���
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y - 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1); //100000;
				edgeNum++;
			}
		}
	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // ��0
	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // ���ﲻ�ָ�

	//���ò�ͬ��ɫ����ָ�ͼ��_______	
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
	//___�������__________

	//���ñ߽��ߵķ�ʽ����ָ�ͼ��
	/*cv::Vec3b color = {255,255,255};
	 DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//�߽���ɫ
	*/

	/* //����ͼ�ָ����ɵ���ͨͼ
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing���������ͼ������
	CV_Assert(1 == u->num_sets());// ��һ��
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //���нڵ㣬��i��������Ϊ��i���ڵ�

	for (int i = 0; i < edgeNum; i++) {  //��ÿ���ڵ�װ�����ӽڵ�
		// if(!edges_mask[i]) continue;

		//�����������ӱߣ���edges_mask[]=255ʱ
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa������
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb������

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb��pa���ӽڵ�
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa��pb���ӽڵ�
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //����

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	//m_tree[0] = AdjTable[0]; //��0�����أ�����0���ڵ�
	int center = 0; // ((int)(m_imgSize.height / 2.0 + 0.5) - 1)*m_imgSize.width + 0;
	m_tree[0] = AdjTable[center];//�����[height/2-1,0]
	isVisited[center] = true;;//isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //����
		TreeNode &p = m_tree[start++]; //����

		for (int i = 0; i < p.childrenNum; i++) { //������ȱ���
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //װ�����ڵ㣬����ڵ�c��Ҫ�����ó�������Ϊp�Ѿ���Ϊ�丸�ڵ㣬������������Ϊc���ӽڵ���

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //������ֻ�
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //���
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);
	//printf("pixelsNum=%d\n", pixelsNum);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //���ԣ���ͷ���β�������
	cout << "BuildTree_H is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_DU(cv::Size size, float sigma,  CWeightProvider &weightProvider) { // ����-���� �Խ�����
	//UpdateTable(sigma); //���±��
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2]; // �ߵ�����Ϊ height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = 0; x < m_imgSize.width; x++) {
			if ((x+1 <= m_imgSize.width - 1) && ( y+1<=m_imgSize.height-1) ) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y+1) * m_imgSize.width + (x + 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x + 1, y+1); // �ߵ�Ȩ��
				edgeNum++;
			}

			// ���Բ�����ͨͼ
			if ( y + 1 <= m_imgSize.height - 1 && x == 0) {   //��������˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y + 1);//100000;
				edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == 0) {   //�������϶˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x + 1, y); //100000;
				edgeNum++;
			}
		}
	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // ��0

	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // ���ﲻ�ָ�

    //���ò�ͬ��ɫ����ָ�ͼ��_______	
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
	//___�������__________

   //���ñ߽��ߵķ�ʽ����ָ�ͼ��
   /*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//�߽���ɫ
   */

	/* //����ͼ�ָ����ɵ���ͨͼ
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing���������ͼ������
	CV_Assert(1 == u->num_sets());// ��һ��
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //���нڵ㣬��i��������Ϊ��i���ڵ�

	for (int i = 0; i < edgeNum; i++) {  //��ÿ���ڵ�װ�����ӽڵ�
										 // if(!edges_mask[i]) continue;

										 //�����������ӱߣ���edges_mask[]=255ʱ
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa������
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb������

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb��pa���ӽڵ�
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa��pb���ӽڵ�
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //����

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	m_tree[0] = AdjTable[0]; //��0�����أ�����0���ڵ�
	isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //����
		TreeNode &p = m_tree[start++]; //����

		for (int i = 0; i < p.childrenNum; i++) { //������ȱ���
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //װ�����ڵ㣬����ڵ�c��Ҫ�����ó�������Ϊp�Ѿ���Ϊ�丸�ڵ㣬������������Ϊc���ӽڵ���

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //������ֻ�
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //���
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //���ԣ���ͷ���β�������
	cout << "BuildTree_DU is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_DD(cv::Size size, float sigma,  CWeightProvider &weightProvider) { // ����-���� �Խ�����
	//UpdateTable(sigma); //���±��
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2]; // �ߵ�����Ϊ height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = m_imgSize.width-1; x >=0; x--) {
			if (( x-1 >= 0 ) && (y + 1 <= m_imgSize.height - 1)) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + (x - 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x - 1, y + 1);  // �ߵ�Ȩ��
				edgeNum++;
			}

		/*	if (y + 1 <= m_imgSize.height - 1 && x == m_imgSize.width-1) {   //�������Ҷ˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 1000; //weightProvider.GetWeight(x, y, x, y - 1);
				edgeNum++;
			}

	    if (x + 1 <= m_imgSize.width - 1 && y == 0) {   //�������϶˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 1000; //weightProvider.GetWeight(x, y, x, y - 1);
				edgeNum++;
			}
		}   */

			/* ���Բ�����ͨͼ*/
			if (y + 1 <= m_imgSize.height - 1 && x == 0) {   //��������˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y + 1); //100000;
				edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == m_imgSize.height - 1) {   //�������¶˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x + 1, y); //100000;
				edgeNum++;
			}
			
		}

	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // ��0

	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // ���ﲻ�ָ�

	//���ò�ͬ��ɫ����ָ�ͼ��_______	
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
	//___�������__________

	//���ñ߽��ߵķ�ʽ����ָ�ͼ��
	/*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//�߽���ɫ
	*/

	/* //����ͼ�ָ����ɵ���ͨͼ
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing���������ͼ������
	CV_Assert(1 == u->num_sets());// ��һ��
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //���нڵ㣬��i��������Ϊ��i���ڵ�

	for (int i = 0; i < edgeNum; i++) {  //��ÿ���ڵ�װ�����ӽڵ�
										 // if(!edges_mask[i]) continue;

										 //�����������ӱߣ���edges_mask[]=255ʱ
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa������
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb������

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb��pa���ӽڵ�
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa��pb���ӽڵ�
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //����

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	//m_tree[0] = AdjTable[0]; //��0�����أ�����0���ڵ�
	int center = 0;// (int)(m_imgSize.height - 1)*m_imgSize.width + 0;
	m_tree[0] = AdjTable[center];//�����[height/2-1,0]
	isVisited[center] = true;;//isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //����
		TreeNode &p = m_tree[start++]; //����

		for (int i = 0; i < p.childrenNum; i++) { //������ȱ���
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //װ�����ڵ㣬����ڵ�c��Ҫ�����ó�������Ϊp�Ѿ���Ϊ�丸�ڵ㣬������������Ϊc���ӽڵ���

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //������ֻ�
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //���
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //���ԣ���ͷ���β�������

	cout << "BuildTree_DD is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_xyx2y1(cv::Size size, float sigma,  CWeightProvider &weightProvider) { // ����-��������Խ�������������(x,y)��(x+2,y+1),����5����sigma
	//UpdateTable(sigma); //���±��
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();
	
	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2]; // �ߵ�����Ϊ height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = 0; x < m_imgSize.width; x++) {
			if ((x + 2 <= m_imgSize.width - 1) && (y+1<= m_imgSize.height - 1)) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y+1) * m_imgSize.width + (x + 2);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x + 2, y+1); // �ߵ�Ȩ��
				edgeNum++;
			}

			/* ���Բ�����ͨͼ?*/

			
			if (y + 1 <= m_imgSize.height - 1 && x == 0) {   //��������˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; // weightProvider.GetWeight(x, y, x, y + 1);//1000; 
				edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == 0) {   //�������϶˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000;// weightProvider.GetWeight(x, y, x + 1, y); //1000;
				edgeNum++;
			}

			//���ӵ�2�����1��֮��Ŀ�϶��������(x,y)��(x+2,y+1)���ڵĿ�϶��Ϊ�˱���[0,0]��[1,0]�ظ���2��
			if (x == 0 && y >= 1) {
				edges[edgeNum].a = y * m_imgSize.width + x; //��1��
				edges[edgeNum].b = y * m_imgSize.width + x + 1; //��2��
				edges[edgeNum].w = 100000; // weightProvider.GetWeight(x, y, x, y + 1);//1000; 
				edgeNum++;
			}
			
		}
	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // ��0

	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // ���ﲻ�ָ�

	//���ò�ͬ��ɫ����ָ�ͼ��_______	
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
	//___�������__________

	//���ñ߽��ߵķ�ʽ����ָ�ͼ��
	/*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//�߽���ɫ
	*/

	/* //����ͼ�ָ����ɵ���ͨͼ
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing���������ͼ������
	CV_Assert(1 == u->num_sets());// ��һ��
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //���нڵ㣬��i��������Ϊ��i���ڵ�

	for (int i = 0; i < edgeNum; i++) {  //��ÿ���ڵ�װ�����ӽڵ�
										 // if(!edges_mask[i]) continue;

										 //�����������ӱߣ���edges_mask[]=255ʱ
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa������
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb������

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb��pa���ӽڵ�
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa��pb���ӽڵ�
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //����

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	m_tree[0] = AdjTable[0]; //��0�����أ�����0���ڵ�
	isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //����
		TreeNode &p = m_tree[start++]; //����

		for (int i = 0; i < p.childrenNum; i++) { //������ȱ���
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //װ�����ڵ㣬����ڵ�c��Ҫ�����ó�������Ϊp�Ѿ���Ϊ�丸�ڵ㣬������������Ϊc���ӽڵ���

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //������ֻ�
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //���
		}
	}

	//printf("start=%d\n", pixelsNum);
	//printf("end=%d\n", edgeNum);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //���ԣ���ͷ���β�������
	cout << "BuildTree_xyx2y1 is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_xyx1y2(cv::Size size, float sigma,  CWeightProvider &weightProvider) { // ����-��������Խ�������������(x,y)��(x+1,y+2),����5����sigma
	//UpdateTable(sigma); //���±��
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2]; // �ߵ�����Ϊ height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = 0; x < m_imgSize.width; x++) {
			if ((x + 1 <= m_imgSize.width - 1) && (y + 2 <= m_imgSize.height - 1)) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 2) * m_imgSize.width + (x + 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x + 1, y + 2); // �ߵ�Ȩ��
				edgeNum++;
			}

			/* ���Բ�����ͨͼ?*/
			if (y + 1 <= m_imgSize.height - 1 && x == 0) {   //��������˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; // weightProvider.GetWeight(x, y, x, y + 1);//1000; 
				edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == 0) {   //�������϶˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000;// weightProvider.GetWeight(x, y, x + 1, y); //1000;
				edgeNum++;
			}

			//���ӵ�1�к͵�2�У�������(x,y)��(x+1,y+2)���ֵķ�϶
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
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // ��0

	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // ���ﲻ�ָ�

	//���ò�ͬ��ɫ����ָ�ͼ��_______	
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
	//___�������__________

	//���ñ߽��ߵķ�ʽ����ָ�ͼ��
	/*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//�߽���ɫ
	*/

	/* //����ͼ�ָ����ɵ���ͨͼ
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing���������ͼ������
	CV_Assert(1 == u->num_sets());// ��һ��
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //���нڵ㣬��i��������Ϊ��i���ڵ�

	for (int i = 0; i < edgeNum; i++) {  //��ÿ���ڵ�װ�����ӽڵ�
										 // if(!edges_mask[i]) continue;

										 //�����������ӱߣ���edges_mask[]=255ʱ
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa������
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb������

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb��pa���ӽڵ�
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa��pb���ӽڵ�
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //����

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	m_tree[0] = AdjTable[0]; //��0�����أ�����0���ڵ�
	isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //����
		TreeNode &p = m_tree[start++]; //����

		for (int i = 0; i < p.childrenNum; i++) { //������ȱ���
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //װ�����ڵ㣬����ڵ�c��Ҫ�����ó�������Ϊp�Ѿ���Ϊ�丸�ڵ㣬������������Ϊc���ӽڵ���

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //������ֻ�
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //���
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //���ԣ���ͷ���β�������
	cout << "BuildTree_xyx1y2 is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_xyx_1y2(cv::Size size, float sigma, CWeightProvider &weightProvider) { // ����-���϶Խ�������������(x,y)��(x-1,y+2),����5����sigma
	//UpdateTable(sigma); //���±��
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2]; // �ߵ�����Ϊ height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = m_imgSize.width - 1; x >= 0; x--) {
			if ((x - 1 >= 0) && (y + 2 <= m_imgSize.height - 1)) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 2) * m_imgSize.width + (x - 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x - 1, y + 2);  // �ߵ�Ȩ��
				edgeNum++;
			}

			if (y + 1 <= m_imgSize.height - 1 && x == m_imgSize.width-1) {   //�������Ҷ˵ı�
			edges[edgeNum].a = y * m_imgSize.width + x;
			edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
			edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1);
			edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == 0) {   //�������϶˵ı�
			edges[edgeNum].a = y * m_imgSize.width + x;
			edges[edgeNum].b = y * m_imgSize.width + x + 1;
			edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1);
			edgeNum++;
			}

			//���ӵ�1�к͵�2�У�����϶��x <m_imgSize.width - 1��Ϊ�˱���[m_imgSize.width - 1,0]��[m_imgSize.width - 1,1]���ظ�
			if (y ==0 && x < m_imgSize.width - 1) {  
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1);
				edgeNum++;
			}
			
		  

			/*
			// ���Բ�����ͨͼ
			if (y + 1 <= m_imgSize.height - 1 && x == 0) {   //��������˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000;// weightProvider.GetWeight(x, y, x, y + 1); //1000;
				edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == m_imgSize.height - 1) {   //�������¶˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000;// weightProvider.GetWeight(x, y, x + 1, y); //1000;
				edgeNum++;
			}
			*/
			
		} 

	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // ��0

	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // ���ﲻ�ָ�

	//���ò�ͬ��ɫ����ָ�ͼ��_______	
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
	//___�������__________

	//���ñ߽��ߵķ�ʽ����ָ�ͼ��
	/*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//�߽���ɫ
	*/

	/* //����ͼ�ָ����ɵ���ͨͼ
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing���������ͼ������
	CV_Assert(1 == u->num_sets());// ��һ��
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //���нڵ㣬��i��������Ϊ��i���ڵ�

	for (int i = 0; i < edgeNum; i++) {  //��ÿ���ڵ�װ�����ӽڵ�
										 // if(!edges_mask[i]) continue;

										 //�����������ӱߣ���edges_mask[]=255ʱ
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa������
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb������

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb��pa���ӽڵ�
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa��pb���ӽڵ�
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //����

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	//m_tree[0] = AdjTable[0]; //��0�����أ�����0���ڵ�
	int center = 0;// (int)(m_imgSize.width - 1); //���Ͻ�
	m_tree[0] = AdjTable[center];//�����[height/2-1,0]
	isVisited[center] = true;;//isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //����
		TreeNode &p = m_tree[start++]; //����

		for (int i = 0; i < p.childrenNum; i++) { //������ȱ���
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //װ�����ڵ㣬����ڵ�c��Ҫ�����ó�������Ϊp�Ѿ���Ϊ�丸�ڵ㣬������������Ϊc���ӽڵ���

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //������ֻ�
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //���
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //���ԣ���ͷ���β�������

	cout<<"BuildTree_xyx_1y2 is over"<<endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildTree_xyx_2y1(cv::Size size, float sigma, CWeightProvider &weightProvider) { // ����-���϶Խ�������������(x,y)��(x-2,y+1),����5����sigma
	//UpdateTable(sigma); //���±��
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2]; // �ߵ�����Ϊ height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = m_imgSize.width - 1; x >= 0; x--) {
			if ((x - 2 >= 0) && (y + 1 <= m_imgSize.height - 1)) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + (x - 2);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x - 2, y + 1);  // �ߵ�Ȩ��
				edgeNum++;
			}

			if (y + 1 <= m_imgSize.height - 1 && x == m_imgSize.width-1) {   //�������Ҷ˵ı�
			edges[edgeNum].a = y * m_imgSize.width + x;
			edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
			edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1);
			edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == 0) {   //�������϶˵ı�
			edges[edgeNum].a = y * m_imgSize.width + x;
			edges[edgeNum].b = y * m_imgSize.width + x + 1;
			edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1);
			edgeNum++;
			}
			 
			//����(x,y)��(x-2,y+1)����ַ�϶�����ӵ�����2�к����1�еķ�϶��y != 0�����[m_imgSize.width - 2,0]��[m_imgSize.width - 1,0]�ظ�����
			if (x + 1 == m_imgSize.width - 1 && y != 0) {   //�������϶˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000; //weightProvider.GetWeight(x, y, x, y - 1);
				edgeNum++;
			}

			/* ���Բ�����ͨͼ
			if (y + 1 <= m_imgSize.height - 1 && x == 0) {   //��������˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = (y + 1) * m_imgSize.width + x;
				edges[edgeNum].w = 100000;// weightProvider.GetWeight(x, y, x, y + 1); //1000;
				edgeNum++;
			}

			if (x + 1 <= m_imgSize.width - 1 && y == m_imgSize.height - 1) {   //�������¶˵ı�
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + x + 1;
				edges[edgeNum].w = 100000;// weightProvider.GetWeight(x, y, x + 1, y); //1000;
				edgeNum++;
			}
			*/
		}

	}

	//uchar *edges_mask = new uchar[edgeNum];
	//memset(edges_mask, 0, sizeof(uchar) * edgeNum); // ��0

	//universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // ���ﲻ�ָ�

	//���ò�ͬ��ɫ����ָ�ͼ��_______	����ͼ�ķָ��㷨�л��ƵĲ�ͬ��ɫ��ָ�
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
	//___�������__________

	//���ñ߽��ߵķ�ʽ����ָ�ͼ��
	/*cv::Vec3b color = {255,255,255};
	DrawContoursAroundSegments(img,u, m_imgSize.width, m_imgSize.height, color );//�߽���ɫ
	*/

	/* //����ͼ�ָ����ɵ���ͨͼ
	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing���������ͼ������
	CV_Assert(1 == u->num_sets());// ��һ��
	*/

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //���нڵ㣬��i��������Ϊ��i���ڵ�

	for (int i = 0; i < edgeNum; i++) {  //��ÿ���ڵ�װ�����ӽڵ�
										 // if(!edges_mask[i]) continue;

										 //�����������ӱߣ���edges_mask[]=255ʱ
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa������
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb������

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb��pa���ӽڵ�
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa��pb���ӽڵ�
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //����

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	//m_tree[0] = AdjTable[0]; //��0�����أ�����0���ڵ�
	int center = 0;// m_imgSize.width - 1; //(int)(m_imgSize.height - 1)*m_imgSize.width + 0;
	m_tree[0] = AdjTable[center];//�����[height/2-1,0]
	isVisited[center] = true;;//isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //����
		TreeNode &p = m_tree[start++]; //����

		for (int i = 0; i < p.childrenNum; i++) { //������ȱ���
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //װ�����ڵ㣬����ڵ�c��Ҫ�����ó�������Ϊp�Ѿ���Ϊ�丸�ڵ㣬������������Ϊc���ӽڵ���

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //������ֻ�
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //���
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	//CV_Assert(start == pixelsNum && end == pixelsNum); //���ԣ���ͷ���β�������
	cout<<"BuildTree_xyx_2y1 is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	//delete u;
	//delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildSegmentTree(cv::Size size, float sigma, float tau, CWeightProvider &weightProvider)  // �ָ�����ϵ���С������
{
	UpdateTable(sigma); //���±��
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2];// �ߵ�����Ϊ height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = 0; x < m_imgSize.width; x++) {
			if (x < m_imgSize.width - 1) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + (x + 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x + 1, y); // �ߵ�Ȩ��
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
	memset(edges_mask, 0, sizeof(uchar) * edgeNum); // ��0
	universe * u = segment_graph(pixelsNum, edgeNum, edges, tau, edges_mask); // �ָ�ɶ����ͼ

	//���ñ߽��ߵķ�ʽ����ָ�ͼ��
	cv::Vec3b color = { 255,255,255 };
	DrawContoursAroundSegments(img, u, m_imgSize.width, m_imgSize.height, color /*�߽���ɫ*/);


	u = segment_graph_one(u, edgeNum, edges, edges_mask); // Mei Xing���������ͼ�����������������еı�ѡ�����ˣ����Ҹ���Ϊ���ظ���-1
	CV_Assert(1 == u->num_sets());// ��һ��ͼ

	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //���нڵ㣬��i��������Ϊ��i���ڵ�

	for (int i = 0; i < edgeNum; i++) {  //��ÿ���ڵ�װ�����ӽڵ�
		if (!edges_mask[i]) continue;

		//�����������ӱߣ���edges_mask[]=255ʱ
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa������
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb������

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb��pa���ӽڵ�
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa��pb���ӽڵ�
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //����

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	m_tree[0] = AdjTable[0]; //��0�����أ�����0���ڵ�
	isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //����
		TreeNode &p = m_tree[start++]; //����

		for (int i = 0; i < p.childrenNum; i++) { //������ȱ���
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //װ�����ڵ㣬����ڵ�c��Ҫ�����ó�������Ϊp�Ѿ���Ϊ�丸�ڵ㣬������������Ϊc���ӽڵ���

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //������ֻ�
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //���
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	CV_Assert(start == pixelsNum && end == pixelsNum); //���ԣ���ͷ���β�������
	cout << "BuildSegmentTree is over!" << endl;

	delete[]isVisited;
	delete[]AdjTable;
	delete u;
	delete[]edges_mask;
	delete[]edges;
}

void CSegmentTree::BuildMSTree(cv::Size size, float sigma,  CWeightProvider &weightProvider)  //��С������
{
	UpdateTable(sigma); //���±��
	m_imgSize = size;
	int pixelsNum = m_imgSize.area();

	//step 1: build segment tree
	edge *edges = new edge[m_imgSize.area() * NUM_NEIGHBOR / 2];// �ߵ�����Ϊ height*(width-1)+width*(height-1)
	int edgeNum = 0;
	for (int y = 0; y < m_imgSize.height; y++) {
		for (int x = 0; x < m_imgSize.width; x++) {
			if (x < m_imgSize.width - 1) {
				edges[edgeNum].a = y * m_imgSize.width + x;
				edges[edgeNum].b = y * m_imgSize.width + (x + 1);
				edges[edgeNum].w = weightProvider.GetWeight(x, y, x + 1, y); // �ߵ�Ȩ��
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
	memset(edges_mask, 0, sizeof(uchar) * edgeNum);  // ��0
	universe *u = MST_graph(pixelsNum, edgeNum, edges, edges_mask); //�Ѿ�ѡ���˱�

																	//step 2: build node based graph
	TreeNode *AdjTable = new TreeNode[pixelsNum];
	for (int i = 0; i < pixelsNum; i++)
		AdjTable[i].id = i; //���нڵ㣬��i��������Ϊ��i���ڵ�

	for (int i = 0; i < edgeNum; i++) {  //��ÿ���ڵ�װ�����ӽڵ�
		if (!edges_mask[i]) continue;

		//�����������ӱߣ���edges_mask[]=255ʱ
		int pa = edges[i].a;
		int pb = edges[i].b;
		int dis = std::min(int(edges[i].w * weightProvider.GetScale() + 0.5f), 255);

		int x0, y0, x1, y1;
		x0 = pa % m_imgSize.width; y0 = pa / m_imgSize.width; // pa������
		x1 = pb % m_imgSize.width; y1 = pb / m_imgSize.width; // pb������

		TreeNode &nodeA = AdjTable[pa];
		TreeNode &nodeB = AdjTable[pb];

		nodeA.children[nodeA.childrenNum].id = pb;  // pb��pa���ӽڵ�
		nodeA.children[nodeA.childrenNum].dist = (uchar)dis;
		nodeA.childrenNum++;

		nodeB.children[nodeB.childrenNum].id = pa;  // pa��pb���ӽڵ�
		nodeB.children[nodeB.childrenNum].dist = (uchar)dis;
		nodeB.childrenNum++;
	}

	//step 3: build ordered tree
	if (!m_tree.empty()) m_tree.clear();
	m_tree.resize(pixelsNum); //����

	bool *isVisited = new bool[pixelsNum];
	memset(isVisited, 0, sizeof(bool) * pixelsNum);

	m_tree[0] = AdjTable[0]; //��0�����أ�����0���ڵ�
	isVisited[0] = true;
	int start = 0, end = 1;

	while (start < end) { //����
		TreeNode &p = m_tree[start++]; //����

		for (int i = 0; i < p.childrenNum; i++) { //������ȱ���
			if (isVisited[p.children[i].id]) continue;

			isVisited[p.children[i].id] = true;

			TreeNode c;
			c.id = p.children[i].id;
			c.father.id = p.id;
			c.father.dist = p.children[i].dist; //װ�����ڵ㣬����ڵ�c��Ҫ�����ó�������Ϊp�Ѿ���Ϊ�丸�ڵ㣬������������Ϊc���ӽڵ���

			TreeNode &t = AdjTable[c.id];
			for (int j = 0; j < t.childrenNum; j++) {
				if (t.children[j].id != p.id) { //������ֻ�
					c.children[c.childrenNum++] = t.children[j];
				}
			}
			m_tree[end++] = c; //���
		}
	}

	//printf("start=%d\n", start);
	//printf("end=%d\n", end);

	CV_Assert(start == pixelsNum && end == pixelsNum); //���ԣ���ͷ���β�������
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

void CSegmentTree::Filter(cv::Mat costVol, int channel) { // channel���Ӳ�����
	cv::Mat costBuffer = costVol.clone();

	KIdx_<double, 3>  costPtr((double *)costVol.data, m_imgSize.height, m_imgSize.width, channel);
	KIdx_<double, 3>  bufferPtr((double *)costBuffer.data, m_imgSize.height, m_imgSize.width, channel);


	int pixelsNum = m_imgSize.area();
	//first pass: from leaf to root����Ҷ�ӽڵ㵽���ڵ�,���ﱣ֤���һ������Ҷ�ӽڵ㣬0Ϊ���ڵ�
	for (int i = pixelsNum - 1; i >= 0; i--) {
		TreeNode &node = m_tree[i];
		double *cost = &bufferPtr(node.id * channel);
		for (int z = 0; z < node.childrenNum; z++) { //�ж��Ƿ����ӽڵ�			
			double *child_cost = &bufferPtr(node.children[z].id * channel);
			double weight = m_table[node.children[z].dist];
			for (int k = 0; k < channel; k++) {
				cost[k] += child_cost[k] * weight;
			}
		}
	}

	
	//second pass: from root to leaf���Ӹ��ڵ㵽Ҷ�ӽڵ�
	memcpy(&costPtr(0), &bufferPtr(0), sizeof(double) * channel); //���ڵ�
	for (int i = 1; i < pixelsNum; i++) { //��1��ʼ����Ҫ�õ�fater�ڵ�
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

void CSegmentTree::Filter_gray(cv::Mat costVol) {  //���ԻҶ�ͼ�� channel=1��ͨ������
	cv::Mat costBuffer = costVol.clone();
	
	// KIdx_<float, 3>  costPtr((float *)costVol.data, m_imgSize.height, m_imgSize.width, channel);
	// KIdx_<float, 3>  bufferPtr((float *)costBuffer.data, m_imgSize.height, m_imgSize.width, channel);
     	
	KIdx_<double, 2>  costPtr((double *)costVol.data, m_imgSize.height, m_imgSize.width);
	KIdx_<double, 2>  bufferPtr((double *)costBuffer.data, m_imgSize.height, m_imgSize.width);
	

	int pixelsNum = m_imgSize.area();
	//first pass: from leaf to root����Ҷ�ӽڵ㵽���ڵ�
	for (int i = pixelsNum - 1; i >= 0; i--) {
		TreeNode &node = m_tree[i];
		double *cost = &bufferPtr(node.id );
		for (int z = 0; z < node.childrenNum; z++) { //�ж��Ƿ����ӽڵ�
			double *child_cost = &bufferPtr(node.children[z].id );
			double weight = m_table[node.children[z].dist];			
				*cost += (*child_cost) * weight;			
		}
	}

	//second pass: from root to leaf���Ӹ��ڵ㵽Ҷ�ӽڵ�
	memcpy(&costPtr(0), &bufferPtr(0), sizeof(double)); //���ڵ�
	for (int i = 1; i < pixelsNum; i++) { //��1��ʼ����Ҫ�õ�fater�ڵ�
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
	MeanFilter(img, img, 1); //��ֵ�˲�
	imgPtr = img;
	cannymask.create(img.size(), CV_8U);
	gray.create(img.size(), CV_8U);
	grayPtr = gray;
	cannyPtr = cannymask;
	cvtColor(img, gray, COLOR_BGR2GRAY); //ע��img��BGR
	blur(gray, cannymask, Size(3, 3));
	Canny(cannymask, cannymask, 20, 60, 3); //ע��Canny���ӵĲ�������
}

float CColorWeight::GetWeight(int x0, int y0, int x1, int y1) const {
	 
	//return (float)(std::abs(imgPtr(y0, x0)[0] - imgPtr(y1, x1)[0])+ std::abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1])+ std::abs(imgPtr(y0, x0)[2] - imgPtr(y1, x1)[2]))/3.0;	
	
	return (float)std::max(std::max(abs(imgPtr(y0, x0)[0] - imgPtr(y1, x1)[0]), abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1])), abs(imgPtr(y0, x0)[2] - imgPtr(y1, x1)[2])); //ʹ��max()����*/
	
	/*
	return (float)std::sqrt((std::abs(imgPtr(y0, x0)[0] - imgPtr(y1, x1)[0])*std::abs(imgPtr(y0, x0)[0] - imgPtr(y1, x1)[0]) +
		std::abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1])*std::abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1]) +
		std::abs(imgPtr(y0, x0)[2] - imgPtr(y1, x1)[2]))*std::abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1])); // 3.0
    */
   
	/*
	float t = (float)std::max(
		std::max(abs(imgPtr(y0, x0)[0] - imgPtr(y1, x1)[0]), abs(imgPtr(y0, x0)[1] - imgPtr(y1, x1)[1])), abs(imgPtr(y0, x0)[2] - imgPtr(y1, x1)[2]) //ʹ��max()����
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