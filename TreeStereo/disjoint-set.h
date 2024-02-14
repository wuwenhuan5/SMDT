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

#ifndef DISJOINT_SET
#define DISJOINT_SET

// disjoint-set forests using union-by-rank and path compression (sort of).

typedef struct {
	int rank;
	int p;
	int size;
} uni_elt;

class universe {
public:
	universe(int elements);
	~universe();
	int find(int x);  
	void join(int x, int y);
	void set(int x,int y);
	int size(int x) const { return elts[x].size; }
	int num_sets() const { return num; }

private:
	uni_elt *elts;
	int num;
};

universe::universe(int elements) {
	elts = new uni_elt[elements];
	num = elements;
	for (int i = 0; i < elements; i++) { //每个点是一个连通分量
		elts[i].rank = 0;
		elts[i].size = 1;
		elts[i].p = i;
	}
}

universe::~universe() {
	delete [] elts;
}

int universe::find(int x) { //查x所在的连通分量
	int y = x;
	while (y != elts[y].p) //表示y被合并过
		y = elts[y].p;  //这是连通分量中y=elts[y].p的元素
	elts[x].p = y;
	return y;
}

void universe::join(int x, int y) { //并
	if(x != elts[x].p) x = find(x);
	if(y != elts[y].p) y = find(y);

	if(x == y) return;

	if (elts[x].rank > elts[y].rank) { //rank表示合并的次数或深度吗？越往根节点，rank越大？或合并的次数越多，rank越大？
		elts[y].p = x;
		elts[x].size += elts[y].size;
	} else {
		elts[x].p = y;
		elts[y].size += elts[x].size;
		if (elts[x].rank == elts[y].rank) //rank相等时，y合并x，后者合并前者，x是y的子集
			elts[y].rank++; //？
	}
	num--;
}

void universe::set(int x,int y) //假设x和y是一条边的两端点，若该边是即将要选入MST中，则设置x,y为同一个域
{
	if (x != elts[x].p) x = find(x);
	if (y != elts[y].p) y = find(y);

	if (x == y) return;

	elts[y].p = x; //设置x与y是同一个域

}

#endif
