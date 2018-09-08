#include "graph.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>

void Graph::AddEdge(int from, int to) {
	++edge_count_;
	vector<int>::iterator it;
	it = find(out_edges_[from].begin(), out_edges_[from].end(), to);
	if (it == out_edges_[from].end())
		out_edges_[from].push_back(to);
	it = find(in_edges_[to].begin(), in_edges_[to].end(), from);
	if (it == in_edges_[to].end())
		in_edges_[to].push_back(from);
}

void Graph::SetNodeCount(int node_count) {
	edge_count_ = 0;
	node_count_ = node_count;
	out_edges_.clear();
	out_edges_.resize(node_count_);
	in_edges_.clear();
	in_edges_.resize(node_count_);
	out_neg_edges_.clear();
	out_neg_edges_.resize(node_count_);
	in_neg_edges_.clear();
	in_neg_edges_.resize(node_count_);
}

void Graph::SampleNegativeEdges(vector<Label> &label, double in_ratio, double out_ratio) {
	out_neg_edges_.clear();
	out_neg_edges_.resize(node_count_);
	in_neg_edges_.clear();
	in_neg_edges_.resize(node_count_);
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
#ifdef _WIN32
	std::mt19937 gen(time(0) + clock()); //Standard mersenne_twister_engine seeded with rd()
#else
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
#endif
	int w_count = 0;
	for (; label[w_count].date == 0; ++w_count);
	std::uniform_int_distribution<> word_dis(0, w_count - 1);
	std::uniform_int_distribution<> node_dis(w_count, node_count_ - 1);
	vector<double> in_candidates;
	// for word nodes
	for (int u = 0; u < w_count; ++u) {
		set<int> node_set(out_edges_[u].begin(), out_edges_[u].end());
		for(int i = 0; i < out_edges_[u].size(); ++i){
			int node = node_dis(gen);
			if (node_set.find(node) == node_set.end()){
				out_neg_edges_[u].push_back(node);
				in_neg_edges_[node].push_back(u);
				node_set.insert(node);
			}
		}
	}
	//for standard nodes
	set< pair<int, int> > edge_set;
	for(int u = 0; u < node_count_; ++u)
		for(int i = 0; i < out_edges_[u].size(); ++i)
			edge_set.insert(pair<int, int>(u, out_edges_[u][i]));
	for(int i = 0; i < edge_count_ * in_ratio; ++i){
		int node1 = node_dis(gen);
		int node2 = node_dis(gen);
		if (edge_set.find(pair<int, int>(node1, node2)) == edge_set.end()){
			out_neg_edges_[node1].push_back(node2);
			in_neg_edges_[node2].push_back(node1);
			edge_set.insert(pair<int, int>(node1, node2));
		}
	}
}

// void Graph::SampleNegativeEdges(vector<Label> &label, double in_ratio, double out_ratio) {
// 	out_neg_edges_.clear();
// 	out_neg_edges_.resize(node_count_);
// 	in_neg_edges_.clear();
// 	in_neg_edges_.resize(node_count_);
// 	std::random_device rd;  //Will be used to obtain a seed for the random number engine
// #ifdef _WIN32
// 	std::mt19937 gen(time(0) + clock()); //Standard mersenne_twister_engine seeded with rd()
// #else
// 	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
// #endif
// 	vector<int> in_neg_edge_count(node_count_, 0);
// 	std::uniform_int_distribution<int> dis(0, node_count_ - 1);
// 	for(int i = 0; i < edge_count_ * in_ratio; ++i)
// 		++in_neg_edge_count[dis(gen)];
// 	int candidates_count = 0, w_count = 0;
// 	for(; label[w_count].date == 0; ++w_count);
// 		std::uniform_int_distribution<> word_dis(0, w_count - 1);
// 	vector<double> in_candidates;
// 	in_candidates.reserve(2 * edge_count_ + node_count_ + 10000);
// 	for (int i = w_count + 1; i < node_count_; ++i) {
// 		if (label[i].date != label[i - 1].date) {
// 			for (int j = candidates_count + w_count; j < i; ++j)
// 				for (int k = 0; k < out_edges_[j].size() * 2 + 1; ++k)
// 					in_candidates.push_back(j);
// 			candidates_count = i - w_count;
// 		}
// 		int sampled = 0;
// 		set<int> in_edges_set(in_edges_[i].begin(), in_edges_[i].end());
// 		// while ((sampled + 1) * 4.9 < in_neg_edge_count[i]) {
// 		// 	int node = word_dis(gen);
// 		// 	if (in_edges_set.find(node) == in_edges_set.end()) {
// 		// 		++sampled;
// 		// 		in_edges_set.insert(node);
// 		// 		in_neg_edges_[i].push_back(node);
// 		// 		out_neg_edges_[node].push_back(i);
// 		// 	}
// 		// }
// 		if (in_candidates.size() != 0) {
// 			std::uniform_int_distribution<> dis(0, in_candidates.size() - 1);
// 			// set<int> in_edges_set(in_edges_[i].begin(), in_edges_[i].end());
// 			while (sampled < in_neg_edge_count[i] && in_edges_set.size() < candidates_count) {
// 				int node = in_candidates[dis(gen)];
// 				if (in_edges_set.find(node) == in_edges_set.end()) {
// 					++sampled;
// 					in_edges_set.insert(node);
// 					in_neg_edges_[i].push_back(node);
// 					out_neg_edges_[node].push_back(i);
// 				}
// 			}
// 		}
// 	}
// }




// 0.866
// void Graph::SampleNegativeEdges(vector<Label> &label, double in_ratio, double out_ratio) {
// 	out_neg_edges_.clear();
// 	out_neg_edges_.resize(node_count_);
// 	in_neg_edges_.clear();
// 	in_neg_edges_.resize(node_count_);
// 	std::random_device rd;  //Will be used to obtain a seed for the random number engine
// #ifdef _WIN32
// 	std::mt19937 gen(time(0) + clock()); //Standard mersenne_twister_engine seeded with rd()
// #else
// 	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
// #endif
// 	int candidates_count = 0;
// 	vector<double> in_candidates;
// 	vector<double> out_candidates;
// 	in_candidates.reserve(2 * edge_count_ + node_count_ + 10000);
// 	out_candidates.reserve(2 * edge_count_ + node_count_ + 10000);
// 	for (int i = 1; i < node_count_; ++i) {
// 		if (label[i].date != label[i - 1].date) {
// 			for (int j = candidates_count; j < i; ++j)
// 				for (int k = 0; k < out_edges_[j].size() * 2 + 1; ++k)
// 					in_candidates.push_back(j);
// 			candidates_count = i;
// 		}
// 		if (in_candidates.size() != 0) {
// 			std::uniform_int_distribution<> dis(0, in_candidates.size() - 1);
// 			set<int> in_edges_set(in_edges_[i].begin(), in_edges_[i].end());
// 			int sampled = 0;
// 			while (sampled < in_edges_[i].size() * in_ratio && in_edges_set.size() < candidates_count) {
// 				int node = in_candidates[dis(gen)];
// 				if (in_edges_set.find(node) == in_edges_set.end()) {
// 					++sampled;
// 					in_edges_set.insert(node);
// 					in_neg_edges_[i].push_back(node);
// 					out_neg_edges_[node].push_back(i);
// 				}
// 			}
// 		}
// 	}
// 	candidates_count = 0;
// 	for (int i = node_count_ - 2; i >= 0; --i) {
// 		if (label[i].date != label[i + 1].date) {
// 			for (int j = node_count_ - candidates_count - 1; j > i; --j)
// 				for (int k = 0; k < in_edges_[j].size() * 2 + 1; ++k)
// 					out_candidates.push_back(j);
// 			candidates_count = node_count_ - i - 1;
// 		}
// 		if (out_candidates.size() != 0) {
// 			std::uniform_int_distribution<> dis(0, out_candidates.size() - 1);
// 			set<int> out_edges_set(out_edges_[i].begin(), out_edges_[i].end());
// 			int sampled = 0;
// 			while (sampled < out_edges_[i].size() * out_ratio && out_edges_set.size() < candidates_count) {
// 				int node = out_candidates[dis(gen)];
// 				if (out_edges_set.find(node) == out_edges_set.end()) {
// 					++sampled;
// 					out_edges_set.insert(node);
// 					out_neg_edges_[i].push_back(node);
// 					in_neg_edges_[node].push_back(i);
// 				}
// 			}
// 		}
// 	}
// }

// int main(int argc, char const *argv[])
// {
// 	Graph g(10);
// 	return 0;
// }
