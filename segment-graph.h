/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

//modified by X.Sun, 2012

#ifndef SEGMENT_GRAPH
#define SEGMENT_GRAPH

#define PENALTY_CROSS_SEG 5

#include <vector>
using namespace std;

#include <algorithm>
#include <cmath>
#include "disjoint-set.h"
#include "SegmentTree.h"
#include <opencv2/opencv.hpp>
// threshold function
#define THRESHOLD(size, c) (c/size)
#define MIN_SIZE_SEG 50

/*
 * Segment a graph
 *
 * Returns a disjoint-set forest representing the segmentation.
 *
 * num_vertices: number of vertices in graph.
 * num_edges: number of edges in graph
 * edges: array of edges.
 * c: constant for threshold function.
 */
universe *segment_graph(int num_vertices, int num_edges, edge *edges, 
			float c, unsigned char *edges_mask=NULL) { 
  // sort edges by weight
  std::sort(edges, edges + num_edges); //

  // make a disjoint-set forest
  universe *u = new universe(num_vertices);

  // init thresholds
  float *threshold = new float[num_vertices];
  for (int i = 0; i < num_vertices; i++)
    threshold[i] = THRESHOLD(1,c);
  
  // for each edge, in non-decreasing weight order...�����Ѿ�����������
  for (int i = 0; i < num_edges; i++) {
    edge *pedge = &edges[i];
    
    // components connected by this edge
    int a = u->find(pedge->a);
    int b = u->find(pedge->b);
    if (a != b) 
	{
		if (pedge->w <= threshold[a] && pedge->w <= threshold[b]) //��ʽ6
		{
			edges_mask[i]=255; // ��ʾ����ͨ�����ڵı�
			u->join(a, b);
			a = u->find(a);	
			
			threshold[a]  = pedge->w + THRESHOLD(u->size(a), c);
		}
    }
  }

  /*
  //�ϲ�С�ķָ�
  int num = u->num_sets();
  printf("�ϲ�ǰ�ķָ���%d",num);

  
  //���Խ����ʾ���ϲ����Ӳ�ͼ����ֺܶ��ɫ�ߵ�
 int min_size = 50;
  for (int i = 0; i < num_edges; i++) {
	  int a = u->find(edges[i].a);
	  int b = u->find(edges[i].b);
	  if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
	  {
		  edges_mask[i] = 255; //ÿ���ϲ��ı߶�Ҫ���
		  u->join(a, b);
	  }
		 
  } 
  num = u->num_sets();
  printf("�ϲ���ķָ���%d", num);
  */
  

  // free up
  delete []threshold;
  return u;
}

universe *segment_graph_one(universe *u, int num_edges, edge *edges, unsigned char *edges_mask) //�Լ������segment_graph�����
{
	//added by X. Sun: re-organizing the structures to be a single tree����һ���ı߸���Ϊ(������-1)
	for (int i = 0; i < num_edges; i++)
	{
		int a = u->find(edges[i].a);
		int b = u->find(edges[i].b);
		if (a != b)
		{
			int size_min = MIN(u->size(a), u->size(b));
			u->join(a, b);

			//record
			edges_mask[i] = 255;
			if (size_min > MIN_SIZE_SEG)
				edges[i].w += PENALTY_CROSS_SEG;
		}
	}
	
  return u;
}

//ѡȡ������С�������ı�
universe *MST_graph(int num_vertices, int num_edges, edge *edges, unsigned char *edges_mask)
{
	// sort edges by weight
	std::sort(edges, edges + num_edges);

	// make a disjoint-set forest
	universe *u = new universe(num_vertices);
	// for each edge, in non-decreasing weight order...�����Ѿ�����������
	int nMST=0;
	for (int i = 0; i < num_edges; i++) {
		edge *pedge = &edges[i];

		// components connected by this edge
		int a = u->find(pedge->a);
		int b = u->find(pedge->b);
		if (a != b)
		{		
			u->set(a,b);
			edges_mask[i] = 255; // ��ʾ��ѡ��MST�ı�
			nMST = nMST + 1;
		}
	}
	//cout << "��С�������ı���Ϊ" << nMST << endl;
	return u;
}
//=================================================================================
/// DrawContoursAroundSegments,����ָ����ɫ���Ʒָ�߽磬�����泬���طָ�ͼ���޸�SLIC����Ļ�ͼ����
//=================================================================================
void DrawContoursAroundSegments(
	cv::Mat image,
	//const int*			labels,
	universe *u,
	const int&				width,
	const int&				height,
	cv::Vec3b				color  //�߽���ɫ
	)
{
	

	const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 }; //������

	int sz = width*height;

	vector<bool> istaken(sz, false);

	int mainindex(0);
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			int np(0);
			int lable1 = u->find(mainindex);
			for (int i = 0; i < 8; i++) //ͳ������(j,k)�İ˸��������أ������ǲ�ͬ�����ظ����������np��
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if ((x >= 0 && x < width) && (y >= 0 && y < height))
				{
					int index = y*width + x;

					if (false == istaken[index])
					{
						int lable2 = u->find(index);
						if (lable1 != lable2) np++;
						//if (labels[mainindex] != labels[index]) np++;
					}
				}
			}
			if (np >1)
			{
				int r, c;
				r = mainindex / width;
				c = mainindex % width;
				image.at<cv::Vec3b>(r, c)[0] = color[0];
				image.at<cv::Vec3b>(r, c)[1] = color[1];
				image.at<cv::Vec3b>(r, c)[2] = color[2];

				istaken[mainindex] = true;
			}
			mainindex++;
		}
	}
	cv::namedWindow("SLIC");
	cv::imshow("SLIC", image); //��ʾ�����طָ�ͼ��
	//cv::imwrite("Superpixeltest.jpg", image); //���泬���طָ�ͼ��
	cv::waitKey(30);
}
#endif