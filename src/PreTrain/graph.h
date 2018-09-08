#ifndef GRAPH_H
#define GRAPH_H
#include <vector>
#include <set>
#include <cassert>
#include "label.h"
using namespace std;
// #define DEBUG

class Graph {
  public:
	Graph() = default;
	Graph(int node_count) : node_count_(node_count), edge_count_(0)
		, out_edges_(node_count), in_edges_(node_count) {
			SetNodeCount(node_count);
	}
	~Graph() = default;

	void AddEdge(int from, int to);
	void SetNodeCount(int node_count);
	void SampleNegativeEdges(vector<Label> &label, double in_ratio = 1, double out_ratio = 1);

	int GetNodeCount() {
		return node_count_;
	}
	int GetEdgeCount() {
		return edge_count_;
	}
	inline vector<int> &GetInNeighbour(int index) {
#ifdef DEBUG
		assert((0 <= index && index < in_edges_.size()) && "GetInNeighbour: index out of range");
#endif
		return in_edges_[index];
	}

	inline vector<int> &GetOutNeighbour(int index) {
#ifdef DEBUG
		assert((0 <= index && index < out_edges_.size()) && "GetOutNeighbour: index out of range");
#endif
		return out_edges_[index];
	}
	inline vector<int> &GetInNegNeighbour(int index) {
#ifdef DEBUG
		assert((0 <= index && index < in_neg_edges_.size()) && "GetInNeighbour: index out of range");
#endif
		return in_neg_edges_[index];
	}

	inline vector<int> &GetOutNegNeighbour(int index) {
#ifdef DEBUG
		assert((0 <= index && index < out_neg_edges_.size()) && "GetOutNeighbour: index out of range");
#endif
		return out_neg_edges_[index];
	}
  private:
	vector< vector<int> > out_edges_, in_edges_;
	vector< vector<int> > out_neg_edges_, in_neg_edges_;
	int node_count_; // number of nodes in the graph
	int edge_count_; // number of edges in the graph
};
#endif
