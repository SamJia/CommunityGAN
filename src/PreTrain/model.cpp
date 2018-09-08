#include "model.h"
#include "common.h"
#ifndef __clang__
#include <omp.h>
#endif
#include <fstream>
#include <sstream>
#include <iterator>
#include <iostream>
#include <ctime>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cfloat>
#include <functional>
#include <cstdlib>
#define OMP

void Model::Init(Utils &utils) {
#ifdef __clang__
	output_file_perfix_ = utils.ParseArgumentString("-o", "../AGMtime/output_abstract/saved_", "Output Graph data prefix");
	input_graph_filename_ = utils.ParseArgumentString("-i", "../data/test_graph.txt", "Input edgelist file name");
	input_label_filename_ = utils.ParseArgumentString("-l", "../data/test_time_label.txt", "Input file name for node names (Node ID, Node label) ");
	thread_number_ = utils.ParseArgumentInt("-nt",  4, "Number of threads for parallelization");
#else
//	TODO: Please Change the directory here --> To Jia
	output_file_perfix_ = utils.ParseArgumentString("-o", "A:\\Users\\Sam\\OneDrive - sjtu.edu.cn\\Lab\\AAAI2019\\CDGAN\\data\\community_detection\\com-amazon_pretrain_", "Output Graph data prefix");
	input_graph_filename_ = utils.ParseArgumentString("-i", "A:\\Users\\Sam\\OneDrive - sjtu.edu.cn\\Lab\\AAAI2019\\CDGAN\\data\\community_detection\\com-amazon_agm.txt", "Input edgelist file name");
	input_label_filename_ = utils.ParseArgumentString("-l", "none", "Input file name for node dates (Node ID, Node date) ");
	input_text_filename_ = utils.ParseArgumentString("-t", "none", "Input file name for node' text (Node ID, Node texts), 'none' means do not load text ");
	thread_number_ = utils.ParseArgumentInt("-nt",  omp_get_num_procs(), "Number of threads for parallelization");
#endif
	omp_set_num_threads(thread_number_);
	optimal_community_count_ = utils.ParseArgumentInt("-c", 500, "The number of communities to detect (-1 detect automatically)");
	min_community_count_ = utils.ParseArgumentInt("-mc", 5, "Minimum number of communities to try");
	max_community_count_ = utils.ParseArgumentInt("-xc", 500, "Maximum number of communities to try");
	divide_community_count_ = utils.ParseArgumentInt("-nc", 10, "How many trials for the number of communities");
	step_alpha_ = utils.ParseArgumentDouble("-sa", 0.05, "Alpha for backtracking line search");
	step_beta_ = utils.ParseArgumentDouble("-sb", 0.1, "Beta for backtracking line search");
	same_time_ref_ = utils.ParseArgumentInt("-st", 0, "Allow reference between two same time node or not (0: don't allow, 1: allow)");
	without_eta_ = utils.ParseArgumentInt("-woe", 1, "Disable Eta or not (0: enable eta, 1: disable eta, 2: symmetric eta)");
	same_eta_ = utils.ParseArgumentInt("-se", 1, "same Eta or not (0: different eta, 1: same eta)");
	max_iteration_times_ = utils.ParseArgumentInt("-mi", 500, "Maximum number of update iteration");
	save_iteration_times_ = utils.ParseArgumentInt("-si", 5000, "How many iterations for once save");
	resample_iteration_times_ = utils.ParseArgumentInt("-rsi", 10, "How many iterations for once negative sampling");
	zero_threshold_ = utils.ParseArgumentDouble("-sa", 1e-4, "Zero Threshold for F and eta");
	largest_n_f_ = utils.ParseArgumentDouble("-lnf", 0, "Remain only largest how many elements for F");
	MIN_F_ = 0.0;
	MAX_F_ = 1000.0;
	MIN_ETA_ = 0.0;
	MAX_ETA_ = 1000.0;
	MIN_P_ = 0.0001;
	MAX_P_ = 0.9999;
	MIN_GRAD_ = -10;
	MAX_GRAD_ = 10;
	cerr << "Model initiate done" << TimeString() << endl;
}

void Model::LoadGraph() {
	// system("dir");
	ifstream fin(input_graph_filename_);
	assert(fin && "open graph file failed");
	string s, doc_name, ref_name;
	int doc_id, ref_id;
	map<string, int>::iterator it;
	while (getline(fin, s, '\n')) {
		istringstream iss(s);
		iss >> doc_name; // read in the node name, which is the key of name2id_
		it = name2id_.find("d" + doc_name); // find its corresponding order
		assert((it != name2id_.end()) && "No such node in label");
		doc_id = it->second; // set this doc_id with its corresponding order
		while (iss >> ref_name) {
			it = name2id_.find("d" + ref_name);
			assert((it != name2id_.end()) && "No such node in label");
			ref_id = it->second;
			if (label_[ref_id].date < label_[doc_id].date || (label_[ref_id].date == label_[doc_id].date && same_time_ref_))
				graph_.AddEdge(ref_id, doc_id);
		}
	}
	fin.close();

	if (input_text_filename_ != "none") {
		fin.open(input_text_filename_);
		assert(fin && "open text file failed");
		string word_name;
		int word_id;
		while (fin >> doc_name) {
			it = name2id_.find("d" + doc_name); // find its corresponding order
			assert((it != name2id_.end()) && "No such paper in label file");
			doc_id = it->second; // set this doc_id with its corresponding order
			getline(fin, s, '\n');
			istringstream iss(s);
			while (iss >> word_name) {
				it = name2id_.find("w" + word_name);
				assert((it != name2id_.end()) && "No such paper in label file");
				word_id = it->second;
				graph_.AddEdge(word_id, doc_id);
			}
		}
	}

	for (int i = 0; i < graph_.GetNodeCount(); ++i) {
		sort(graph_.GetInNeighbour(i).begin(), graph_.GetInNeighbour(i).end());
		sort(graph_.GetOutNeighbour(i).begin(), graph_.GetOutNeighbour(i).end());
	}
	// graph_.SampleNegativeEdges(label_, 3);
	// int pos_in_count = 0, pos_out_count = 0, neg_in_count = 0, neg_out_count = 0;
	// for (int u = 0; u < graph_.GetNodeCount(); ++u) {
	// 	pos_in_count += graph_.GetInNeighbour(u).size();
	// 	pos_out_count += graph_.GetOutNeighbour(u).size();
	// 	neg_in_count += graph_.GetInNegNeighbour(u).size();
	// 	neg_out_count += graph_.GetOutNegNeighbour(u).size();
	// }
	// cout << pos_in_count << '\t' << pos_out_count << '\t' << neg_in_count << '\t' << neg_out_count << endl;
	cerr << "Load Graph Done. Totally " << graph_.GetNodeCount() << " nodes and "
	     << graph_.GetEdgeCount() << " edges. " << TimeString() << endl;
}

void Model::LoadLabel() {
	int date;
	string name;
	ifstream fin;
	if (input_label_filename_ != "none") {
		fin.open(input_label_filename_);
		assert(fin && "open label file failed");
		while (fin >> name) {
			fin >> date;
			name = "d" + name;
			label_.push_back(Label(name, date));
		}
		fin.close();
	} else {
		fin.open(input_graph_filename_);
		assert(fin && "open graph file failed");
		set<string> name_set;
		date = 1;
		while (fin >> name) {
			name = "d" + name;
			if (name_set.find(name) == name_set.end()) {
				name_set.insert(name);
				label_.push_back(Label(name, date++));
			}
		}
		fin.close();
	}

	if (input_text_filename_ != "none") {
		fin.open(input_text_filename_);
		assert(fin && "open text file failed");
		set<string> word_set;
		while (fin >> name) {
			getline(fin, name, '\n');
			istringstream iis(name);
			while (iis >> name) {
				name = "w" + name;
				if (word_set.find(name) == word_set.end()) {
					label_.push_back(Label(name, 0));
					word_set.insert(name);
				}
			}
		}
		fin.close();
	}

	sort(label_.begin(), label_.end()); // Time increasing order
	for (int i = 0; i < label_.size(); ++i) {
		name2id_.insert(pair<string, int>(label_[i].name, i));
	}
	graph_.SetNodeCount(label_.size());
	cerr << "Load Label Done. " << TimeString() << endl;
}

int Model::DetectCommunityCount(double hold_fraction) {
	return 100;
}

int Model::GetOptimalCommunityCount() {
	if (optimal_community_count_ == -1)
		return DetectCommunityCount();
	return optimal_community_count_;
}

int Model::CountDifferent(vector<int> v1, vector<int> v2) {
	int count = 0, i1, i2;
	// In graph_init process, neighbours have been sorted. So it need not sort now.
	for (i1 = 0, i2 = 0; i1 < v1.size() && i2 < v2.size();)
		if (v1[i1] < v2[i2]) {
			++i1;
			++count;
		} else {
			if (v1[i1] == v2[i2])
				++i1;
			++i2;
		}
	count += v1.size() - i1;
	return count;
}

void Model::Resize() {
	F_.assign(graph_.GetNodeCount(), NNVector(community_count_));
	newF_.assign(graph_.GetNodeCount(), NNVector(community_count_));
	Eta_.assign(community_count_, NNVector(community_count_));
	newEta_.assign(community_count_, NNVector(community_count_));
	EtaF_.assign(graph_.GetNodeCount(), NNVector(community_count_));
	FEta_.assign(graph_.GetNodeCount(), NNVector(community_count_));
}

double Model::Conductance(int cut, int vol) {
	//if (cut == 0)
	//	return 5;
	if (vol == 0)
		return 10;
	return cut / (double)vol;
}

void Model::InitNeighbourCommunity(int community_count) {
	community_count_ = community_count;
	Resize();

	// compute conductance of neighborhood community
	vector<pair<double, int> > conductances;
	conductances.reserve(graph_.GetNodeCount());
	for (int u = 0; u < graph_.GetNodeCount(); ++u) {
		if (label_[u].name.substr(0, 1) == "w")
			continue;
		vector<int> S;
		S.push_back(u);
		S.insert(S.end(), graph_.GetInNeighbour(u).begin(), graph_.GetInNeighbour(u).end());
		if (S.size() < 3)
			continue;
		int pos;
		for (pos = 1; pos < S.size() && S[pos] < u; ++pos)
			S[pos - 1] = S[pos];
		S[pos - 1] = u;
		int vol = 0, cut = 0;
		for (int i = 0; i < S.size(); ++i) {
			int v = S[i];
			vol += graph_.GetOutNeighbour(v).size();
			cut += CountDifferent(graph_.GetOutNeighbour(v), S);
		}
		if (input_label_filename_ == "none") {
			vector<int> S;
			S.insert(S.end(), graph_.GetInNeighbour(u).begin(), graph_.GetInNeighbour(u).end());
			S.insert(S.end(), graph_.GetOutNeighbour(u).begin(), graph_.GetOutNeighbour(u).end());
			S.push_back(u);
			if (S.size() < 6)
				continue;
			sort(S.begin(), S.end());
			vol = 0, cut = 0;
			for (int i = 0; i < S.size(); ++i) {
				int v = S[i];
				if (u == v)
					continue;
				vol += graph_.GetOutNeighbour(v).size() + graph_.GetInNeighbour(v).size();
				cut += CountDifferent(graph_.GetOutNeighbour(v), S);
				cut += CountDifferent(graph_.GetInNeighbour(v), S);
			}
		}
		conductances.push_back(pair<double, int>(Conductance(cut, vol), u));
	}
	sort(conductances.begin(), conductances.end());
	cout << "conductance computation completed " << TimeString() << endl;
	ofstream fout("conductance.txt");
	for (int i = 0; i < conductances.size(); ++i)
		fout << conductances[i].first << '\t' << conductances[i].second << '\n';
	fout.close();

	// choose nodes with local minimum in conductance
	vector<bool> not_local_min(graph_.GetNodeCount(), false);
	int community_id = 0;
	for (int i = 0; i < conductances.size(); i++) {
		int u = conductances[i].second;
		if (not_local_min[u]) {
			continue;
		}
		//add the node and its neighbors to the current community
		F_[u].set(community_id, 1);
		vector<int> &in_neighbour = graph_.GetInNeighbour(u);
		for (int i = 0; i < in_neighbour.size(); ++i)
			F_[in_neighbour[i]].set(community_id, 1);
		if (input_label_filename_ == "none") {
			vector<int> &out_neighbour = graph_.GetOutNeighbour(u);
			for (int i = 0; i < out_neighbour.size(); ++i)
				F_[out_neighbour[i]].set(community_id, 1);
		}
		// exclude its neighbors from the next considerations
		for (int i = 0; i < in_neighbour.size(); ++i)
			not_local_min[in_neighbour[i]] = true;
		++community_id;
		if (community_id >= community_count) {
			break;
			// community_id = 0;
		}
	}
	if (community_count > community_id) {
		printf("%d communities needed to fill randomly\n", community_count - community_id);
	}
	//assign a member to zero-member community (if any)
	default_random_engine e(time(0));
	uniform_int_distribution<int> uniform_int(0, graph_.GetNodeCount() - 1);
	uniform_real_distribution<double> uniform_double(0, 1);
	for (; community_id < community_count; ++community_id)
		for (int i = 0; i < 10; ++i)
			F_[uniform_int(e)].set(community_id, uniform_double(e));

	//TODO : Initiate eta.
	// for (int i = 0; i < Eta_[0].size(); ++i)
	// 	for (int j = 0; j < Eta_[0].size(); ++j)
	// 		Eta_[i].set(j, 0.3);
	for (int i = 0; i < Eta_[0].size(); ++i)
		Eta_[i].set(i, 0.9);
	if (without_eta_ == 1) {
		for (int i = 0; i < Eta_[0].size(); ++i)
			Eta_[i].set(i, 1);
	}
	// srand(time(0));
	// for(int i = 0; i < Eta_[0].size(); ++i)
	// 	for(int j = 0; j < Eta_[0][0].size(); ++j)
	// 		Eta_[0][i][j] = rand() * 1.0 / RAND_MAX;
}

void Model::CalculateEtaF(vector< NNVector > &eta, vector< NNVector > &etaf) {
	if (without_eta_ == 1) {
		etaf = F_;
		return;
	}
#ifdef OMP
	#pragma omp parallel for num_threads(thread_number_) schedule(dynamic, 128)
#endif
	for (int u = 0; u < F_.size(); ++u)
		for (int i = 0; i < eta.size(); ++i) {
			double tmp = eta[i].dot(F_[u]);
			if (tmp > 1e-6)
				etaf[u].set(i, tmp);
		}
}

void Model::CalculateFEta(vector< NNVector > &eta, vector< NNVector > &feta) {
	if (without_eta_ == 1) {
		feta = F_;
		return;
	}
	vector< NNVector > eta_t = Transpose(eta);
// 	vector<vector<double> > eta_t(community_count_, vector<double>(community_count_));
// 	// #pragma omp parallel for num_threads(thread_number_)
// 	for (int i = 0; i < community_count_; ++i)
// 		for (int j = 0; j < community_count_; ++j)
// 			eta_t[i][j] = eta[j][i];
#ifdef OMP
	#pragma omp parallel for num_threads(thread_number_) schedule(dynamic, 128)
#endif
	for (int u = 0; u < F_.size(); ++u)
		for (int i = 0; i < eta_t.size(); ++i) {
			double tmp = F_[u].dot(eta_t[i]);
			if (tmp > 1e-6)
				feta[u].set(i, tmp);
		}
}

double Model::Norm2(vector<double> &v) {
	return sqrt(VectorDot(v, v));
}

double Model::Norm2(vector<vector<double> > &m) {
	double result = 0.0;
	for (int i = 0; i < m.size(); ++i)
		result += VectorDot(m[i], m[i]);
	return sqrt(result);
}

double Model::Norm2(NNVector &v) {
	return sqrt(v.dot(v));
}

double Model::Norm2(vector<NNVector > &m) {
	double result = 0.0;
	for (int i = 0; i < m.size(); ++i)
		result += m[i].dot(m[i]);
	return sqrt(result);
}

void Model::FGradientForRow(int u, NNVector &pos_gradient, NNVector &neg_gradient) {
	pos_gradient.clear();
	neg_gradient.clear();

	for (int i = 0; i < graph_.GetInNeighbour(u).size(); ++i) {
		int v = graph_.GetInNeighbour(u)[i];
		double fv_eta_fu = FEta_[v].dot(F_[u]);
		double exp_vetau = min(max(exp(-fv_eta_fu), MIN_P_), MAX_P_);
		double argument = exp_vetau / (1 - exp_vetau);
		pos_gradient += FEta_[v] * argument;
	}
	for (int i = 0; i < graph_.GetInNegNeighbour(u).size(); ++i) {
		int v = graph_.GetInNegNeighbour(u)[i];
		neg_gradient += FEta_[v];
	}

	for (int i = 0; i < graph_.GetOutNeighbour(u).size(); ++i) {
		int v = graph_.GetOutNeighbour(u)[i];
		double fu_eta_fv = F_[u].dot(EtaF_[v]);
		double exp_uetav = min(max(exp(-fu_eta_fv), MIN_P_), MAX_P_);
		double argument = exp_uetav / (1 - exp_uetav);
		pos_gradient += EtaF_[v] * argument;
	}
	for (int i = 0; i < graph_.GetOutNegNeighbour(u).size(); ++i) {
		int v = graph_.GetOutNegNeighbour(u)[i];
		neg_gradient += EtaF_[v];
	}
}

double Model::FLikelihoodForRow(int u, NNVector &Fu) {
	double L = 0.0;
	for (int i = 0; i < graph_.GetInNeighbour(u).size(); ++i) {
		int v = graph_.GetInNeighbour(u)[i];
		double fv_eta_fu = FEta_[v].dot(Fu);
		double exp_vetau = min(max(exp(-fv_eta_fu), MIN_P_), MAX_P_);
		L += log(1 - exp_vetau);
	}
	for (int i = 0; i < graph_.GetInNegNeighbour(u).size(); ++i) {
		int v = graph_.GetInNegNeighbour(u)[i];
		L -= FEta_[v].dot(Fu);
	}
	for (int i = 0; i < graph_.GetOutNeighbour(u).size(); ++i) {
		int v = graph_.GetOutNeighbour(u)[i];
		double fu_eta_fv = Fu.dot(EtaF_[v]);
		double exp_uetav = min(max(exp(-fu_eta_fv), MIN_P_), MAX_P_);
		L += log(1 - exp_uetav);
	}
	for (int i = 0; i < graph_.GetOutNegNeighbour(u).size(); ++i) {
		int v = graph_.GetOutNegNeighbour(u)[i];
		L -= Fu.dot(EtaF_[v]);
	}
	return L;
}

bool Model::FGoStepSizeByLineSearch(int u, NNVector &pos_gradient, NNVector &neg_gradient, const int MAX_ITER) {
	double step_size = 1.0;
	double init_likelihood = FLikelihoodForRow(u, F_[u]);
	vector<double> tmp(community_count_, 0);
	VectorAddTo(pos_gradient, tmp);
	VectorSubTo(neg_gradient, tmp);
	double slope = sqrt(VectorDot(tmp, tmp));
	// PrintVector("F_[u]", F_[u]);
	for (int iter = 0; iter < MAX_ITER; iter++, step_size *= step_beta_) {
		// bool exceed = false;
		// for (int c = 0; c < community_count_; c++) {
		// 	double new_value = F_[u][c] + step_size * gradient[c];
		// 	if (new_value < MIN_F_)
		// 		// exceed = true;
		// 		new_value = F_[u][c] / 2;
		// 	if (new_value > MAX_F_)
		// 		// exceed = true;
		// 		new_value = MAX_F_;
		// 	newF_[u][c] = new_value;
		// }
		// if (exceed)
		// 	continue;
		newF_[u] = (F_[u] + (pos_gradient * step_size)) - (neg_gradient * step_size);
		newF_[u].clear_with_threshold(zero_threshold_);
		newF_[u].clip(MIN_F_, MAX_F_);
		if (largest_n_f_ > 0)
			newF_[u].clear_to_largest_n(largest_n_f_);
		if (label_[u].name[0] == 'w'){
			// newF_[u].clear_to_largest_n(2);
			// newF_[u].clip(0, 1);
		}
		//cout << "step size: " << step_size << '\n';
		//PrintVector("newF[u]", newF_[u]);
		//system("PAUSE");
		if (FLikelihoodForRow(u, newF_[u]) >= init_likelihood + step_alpha_ * step_size * slope)
			return true;
	}
	newF_[u] = F_[u];
	return 0;
}

void Model::UpdateF(vector<int> &order) {
	CalculateEtaF(Eta_, EtaF_);
	CalculateFEta(Eta_, FEta_);

#ifdef OMP
	#pragma omp parallel
#endif
	{
		NNVector pos_gradient(community_count_);
		NNVector neg_gradient(community_count_);
#ifdef OMP
		#pragma omp for schedule(dynamic, 64)
#endif
		for (int i = 0; i < order.size(); ++i) {
			int u = i;
			FGradientForRow(u, pos_gradient, neg_gradient);
			if (Norm2(pos_gradient) < 1e-4)
				newF_[u] = F_[u];
			else
				FGoStepSizeByLineSearch(u, pos_gradient, neg_gradient);
		}
	}
	F_ = newF_;
}

void Model::EtaGradient(vector< vector<double> > &pos_gradient, vector< vector<double> > &neg_gradient) {
	pos_gradient.assign(community_count_, vector<double>(community_count_, 0));
	neg_gradient.assign(community_count_, vector<double>(community_count_, 0));
	omp_lock_t lock;
	omp_init_lock(&lock);
#ifdef OMP
	#pragma omp parallel
#endif
	{
		NNVector pos_tmp(community_count_), neg_tmp(community_count_);
#ifdef OMP
		#pragma omp for schedule(dynamic, 64)
#endif
		for (int u = 0; u < graph_.GetNodeCount(); ++u) {
			pos_tmp.clear();
			neg_tmp.clear();
			for (int i = 0; i < graph_.GetOutNeighbour(u).size(); ++i) {
				int v = graph_.GetOutNeighbour(u)[i];
				double fu_eta_fv = F_[u].dot(EtaF_[v]);
				double exp_uetav = min(max(exp(-fu_eta_fv), MIN_P_), MAX_P_);
				double argument = exp_uetav / (1 - exp_uetav);
				pos_tmp += F_[v] * argument;
			}

			for (int i = 0; i < graph_.GetOutNegNeighbour(u).size(); ++i) {
				int v = graph_.GetOutNegNeighbour(u)[i];
				neg_tmp += F_[v];
			}
			//times F_u
			omp_set_lock(&lock);
			int idx_i, idx_j;
			for (int i = 0; i < F_[u].idxs_.size(); ++i) {
				idx_i = F_[u].idxs_[i];
				for (int j = 0; j < pos_tmp.idxs_.size(); ++j) {
					idx_j = pos_tmp.idxs_[j];
					pos_gradient[idx_i][idx_j] += F_[u].values_[i] * pos_tmp.values_[j];
				}
				for (int j = 0; j < neg_tmp.idxs_.size(); ++j) {
					idx_j = neg_tmp.idxs_[j];
					pos_gradient[idx_i][idx_j] += F_[u].values_[i] * neg_tmp.values_[j];
				}
			}
			omp_unset_lock(&lock);
		}
	}
	omp_destroy_lock(&lock);
}

double Model::EtaLikelihood(vector< NNVector > &eta) {
	CalculateEtaF(eta, EtaF_);
	double L = 0.0;
#ifdef OMP
	#pragma omp parallel for reduction(+:L) schedule(dynamic, 64)
#endif
	for (int u = 0; u < graph_.GetNodeCount(); ++u) {
		for (int i = 0; i < graph_.GetOutNeighbour(u).size(); ++i) {
			int v = graph_.GetOutNeighbour(u)[i];
			double fu_eta_fv = F_[u].dot(EtaF_[v]);
			double exp_uetav = min(max(exp(-fu_eta_fv), MIN_P_), MAX_P_);
			L += log(1 - exp_uetav);
		}
		for (int i = 0; i < graph_.GetOutNegNeighbour(u).size(); ++i) {
			int v = graph_.GetOutNegNeighbour(u)[i];
			L -= F_[u].dot(EtaF_[v]);
		}
	}


	return L;
}



double Model::EtaGoStepSizeByLineSearch(vector<vector<double> > &pos_gradient, vector<vector<double> > &neg_gradient, const int MAX_ITER) {
	static double init_step_size = 1;
	double step_size = init_step_size;
	// cout << "initial step_size is " << step_size << endl;
	double init_likelihood = EtaLikelihood(Eta_);
	double slope = 0.0;
	// rolling gradient to a vector
	vector< vector<double> > gradient(community_count_, vector<double>(community_count_));
	for (int i = 0; i < community_count_; ++i)
		for (int j = 0; j < community_count_; ++j) {
			gradient[i][j] = pos_gradient[i][j] - neg_gradient[i][j];
			slope += gradient[i][j] * gradient[i][j];
		}
	slope = sqrt(slope);
	for (int iter = 0; iter < MAX_ITER; iter++, step_size *= step_beta_) {
		bool exceed = false;
#ifdef OMP
		#pragma omp parallel for
#endif
		for (int i = 0; i < community_count_; ++i)
			for (int j = 0; j < community_count_; ++j)
				gradient[i][j] = (pos_gradient[i][j] - neg_gradient[i][j]) * step_size;
#ifdef OMP
		#pragma omp parallel for
#endif
		for (int i = 0; i < community_count_; ++i) {
			newEta_[i] = Eta_[i] + gradient[i];
			newEta_[i].clear_with_threshold(zero_threshold_);
			// for (int j = 0; j < community_count_; ++j) {
			// 	double new_value = Eta_[i][j] + step_size * gradient[i][j];
			// 	if (new_value < MIN_ETA_)
			// 		// exceed = true;
			// 		new_value = Eta_[update_type][i][j] / 2;
			// 	if (new_value > MAX_ETA_)
			// 		// exceed = true;
			// 		new_value = MAX_ETA_;
			// 	newEta_[i][j] = new_value;
			// }
		}
		if (exceed)
			continue;

		double new_likelihood = EtaLikelihood(newEta_);
		double old_likelihood = init_likelihood + step_alpha_ * step_size * slope;
//		cout << new_likelihood << '\n' << old_likelihood << '\n';
		if ( new_likelihood >= old_likelihood ) {
			// newEta_[update_type] = Eta_[update_type]; //reset newEta to Eta
			if (iter >= 3)
				for (int i = 0; i < iter; i += 2)
					init_step_size *= step_beta_;
			else if (iter <= 2)
				init_step_size /= step_beta_;
			// cout << "Return step_size for Eta: " << step_size << endl;
			return step_size;
//			return true;
		}
	}
	newEta_ = Eta_; //reset newEta to Eta
//	return false;
	step_size = 0.0;
	init_step_size *= step_beta_;
	// cout << "Return step_size for Eta: " << step_size << endl;
	return step_size;
}

void Model::UpdateEta() {
	CalculateEtaF(Eta_, EtaF_);

	vector<vector<double> > pos_gradient(community_count_, vector<double>(community_count_, 0));
	vector<vector<double> > neg_gradient(community_count_, vector<double>(community_count_, 0));
	EtaGradient(pos_gradient, neg_gradient); //get search direction: Eta Gradient

	// if (Norm2(pos_gradient) < 1e-4) { return; }
	double step_size = EtaGoStepSizeByLineSearch(pos_gradient, neg_gradient);
	Eta_ = newEta_;
}

double Model::Likelihood() {
	CalculateEtaF(Eta_, EtaF_);
	double L = 0.0;
#ifdef OMP
	#pragma omp parallel for reduction(+:L) schedule(dynamic, 64)
#endif
	for (int u = 0; u < graph_.GetNodeCount(); ++u) {
		for (int i = 0; i < graph_.GetOutNeighbour(u).size(); ++i) {
			int v = graph_.GetOutNeighbour(u)[i];
			double fu_eta_fv = F_[u].dot(EtaF_[v]);
			double exp_uetav = min(max(exp(-fu_eta_fv), MIN_P_), MAX_P_);
			L += log(1 - exp_uetav);
		}
		for (int i = 0; i < graph_.GetOutNegNeighbour(u).size(); ++i) {
			int v = graph_.GetOutNegNeighbour(u)[i];
			L -= F_[u].dot(EtaF_[v]);
		}
	}
	return L;
}

double Model::SaveEdgeProbability(string model_name) {
	CalculateEtaF(Eta_, EtaF_);
	double L = 0.0;
	ofstream fpos(output_file_perfix_ + model_name + ".pos.txt");
	ofstream fneg(output_file_perfix_ + model_name + ".neg.txt");
	for (int u = 0; u < graph_.GetNodeCount(); ++u) {
		for (int i = 0; i < graph_.GetOutNeighbour(u).size(); ++i) {
			int v = graph_.GetOutNeighbour(u)[i];
			double fu_eta_fv = F_[u].dot(EtaF_[v]);
			double exp_uetav = exp(-fu_eta_fv);
			fpos << u << '\t' << v << '\t' << (1 - exp_uetav) << '\n';
		}
		for (int i = 0; i < graph_.GetOutNegNeighbour(u).size(); ++i) {
			int v = graph_.GetOutNegNeighbour(u)[i];
			double fu_eta_fv = F_[u].dot(EtaF_[v]);
			double exp_uetav = exp(-fu_eta_fv);
			fneg << u << '\t' << v << '\t' << (1 - exp_uetav) << '\n';
		}
	}
	fpos.close();
	fneg.close();

	return L;
}

string Model::GenerateModelName(int iteration) {
	assert((iteration >= 0) && "iteration should be greater than zero");
	ostringstream ss;
	ss << setw(5) << iteration;
	string s = ss.str();
	replace(s.begin(), s.end(), ' ', '0');
	return s;
}

void Model::SaveF(string & model_name) {
	string filename = output_file_perfix_ + model_name + ".f.txt";
	cout << filename << endl;
	ofstream fout;
	fout.open(filename);
	assert(fout && "open f ouput file failed");
	fout << F_.size() << '\t' << F_[0].size() << '\n';
	for (int i = 0; i < F_.size(); ++i) {
		fout << label_[i].name << '\t' << F_[i] << '\n';
	}
	fout.close();

	// filename = output_file_perfix_ + model_name + ".name.txt";
	// fout.open(filename);
	// assert(fout && "open name ouput file failed");
	// fout << F_.size() << '\t' << F_[0].size() << '\n';
	// for (int i = 0; i < F_.size(); ++i) {
	// 	fout << label_[i].name << '\n';
	// }
	// fout.close();

	// double varepsilon = (2.0 * graph_.GetEdgeCount()) / graph_.GetNodeCount() / (graph_.GetNodeCount() - 1);
	// double delta = sqrt(-log(1 - varepsilon));
	// double delta = sqrt((2.0 * graph_.GetEdgeCount()) / graph_.GetNodeCount() / graph_.GetNodeCount());
	double delta = sqrt(-log(1 - 1.0 / graph_.GetNodeCount()));
	vector<vector<pair<double, string> > > contents;
	contents.resize(community_count_);
	int type;
	for (int i = 0; i < F_.size(); ++i) {
		for (int j = 0; j < F_[i].idxs_.size(); ++j) {
			double delta = sqrt(-log(1 - 1.0 / graph_.GetNodeCount()) / Eta_[j].get(j));
			if (F_[i].values_[j] > delta)
				contents[j].push_back(pair<double, string>(F_[i].values_[j], label_[i].name));
		}
	}
	for (int i = 0; i < community_count_; ++i) {
		sort(contents[i].begin(), contents[i].end());
	}

	filename = output_file_perfix_ + model_name + ".cmty.txt";
	fout.open(filename);
	assert(fout && "open cmty ouput file failed");
	for (int i = 0; i < community_count_; ++i) {
		for (int j = 0; j < contents[i].size(); ++j)
			fout << contents[i][j].second << '\t';
		fout << '\n';
	}
	fout.close();

	// vector<vector<pair<double, string> > > etaed_contents;
	// etaed_contents.resize(community_count_);
	// CalculateFEta(Eta_[0], FEta_[0]);
	// CalculateEtaF(Eta_[0], EtaF_[0]);
	// for (int i = 0; i < F_.size(); ++i) {
	// 	for (int j = 0; j < F_[i].size(); ++j) {
	// 		double delta = sqrt(-log(1 - 1.0 / graph_.GetNodeCount()) / Eta_[j][j]);
	// 		if (((FEta_[0][i][j] + EtaF_[0][i][j]) / 2) >= delta)
	// 			etaed_contents[j].push_back(pair<double, string>((FEta_[0][i][j] + EtaF_[0][i][j]) / 2, label_[i].name.substr(1, label_[i].name.length() - 1)));
	// 	}
	// }
	// filename = output_file_perfix_ + model_name + ".etaed_snapout.txt";
	// fout.open(filename);
	// assert(fout && "open etaed_snapout ouput file failed");
	// for (int i = 0; i < community_count_; ++i) {
	// 	for (int j = 0; j < etaed_contents[i].size(); ++j)
	// 		fout << etaed_contents[i][j].second << '\t';
	// 	fout << '\n';
	// }
	// fout.close();

	// filename = output_file_perfix_ + model_name + ".contentd.txt";
	// fout.open(filename);
	// assert(fout && "open contentd ouput file failed");
	// for (int i = 0; i < community_count_; ++i) {
	// 	fout << contents[0][i].size() << ';';
	// 	for (int j = 0; j < contents[0][i].size(); ++j)
	// 		fout << contents[0][i][j].first << ':' << contents[0][i][j].second << '\t';
	// 	fout << '\n';
	// }
	// fout.close();

	// filename = output_file_perfix_ + model_name + ".contentt.txt";
	// fout.open(filename);
	// assert(fout && "open contentt ouput file failed");
	// for (int i = 0; i < community_count_; ++i) {
	// 	fout << contents[1][i].size() << ';';
	// 	for (int j = 0; j < contents[1][i].size(); ++j)
	// 		fout << contents[1][i][j].first << ':' << contents[1][i][j].second << '\t';
	// 	fout << '\n';
	// }
	// fout.close();
}

void Model::SaveEta(string & model_name) {
	string filename = output_file_perfix_ + model_name + ".etad.txt";
	ofstream fout(filename);
	assert(fout && "open etad ouput file failed");
	fout << Eta_.size() << '\t' << Eta_[0].size() << '\n';
	for (int i = 0; i < Eta_.size(); ++i) {
		fout << Eta_[i] << '\n';
	}
	fout.close();
}

void Model::SaveLikelihood(string &model_name) {
	string filename = output_file_perfix_ + model_name + ".likelihood.txt";
	ofstream fout(filename);
	assert(fout && "open likelihood ouput file failed");
	fout << likelihoods_.size() << '\n';
	for (int i = 0; i < likelihoods_.size(); ++i) {
		fout << likelihoods_[i] << '\n';
	}
	fout.close();
}


void Model::SaveData(string model_name) {
	SaveF(model_name);
	SaveEta(model_name);
	// SaveEdgeProbability(model_name);
	// SaveLikelihood(model_name);
}


void Model::MLEGradAscent(int MAX_ITER, int RECORD_PERIOD) {
	ofstream fout(output_file_perfix_ + ".F_gradient.txt");
	fout.close();
	fout.open(output_file_perfix_ + ".etaw_gradient.txt");
	fout.close();
	fout.open(output_file_perfix_ + ".etad_gradient.txt");
	fout.close();
	fout.open(output_file_perfix_ + ".L_twopart.txt");
	fout.close();
	if (MAX_ITER == -1)
		MAX_ITER = max_iteration_times_;
	if (RECORD_PERIOD == -1)
		RECORD_PERIOD = save_iteration_times_;
	int iter = 0;
	vector<int> order(graph_.GetNodeCount());
	for (int i = 0; i < order.size(); ++i)
		order[i] = i;
	default_random_engine e(time(0));
	graph_.SampleNegativeEdges(label_, 5, 5);
	for (iter = 0; iter < MAX_ITER; ++iter) {
		int pos_in_count = 0, pos_out_count = 0, neg_in_count = 0, neg_out_count = 0;
		for (int u = 0; u < graph_.GetNodeCount(); ++u) {
			pos_in_count += graph_.GetInNeighbour(u).size();
			pos_out_count += graph_.GetOutNeighbour(u).size();
			neg_in_count += graph_.GetInNegNeighbour(u).size();
			neg_out_count += graph_.GetOutNegNeighbour(u).size();
		}
		cout << pos_in_count << '\t' << pos_out_count << '\t' << neg_in_count << '\t' << neg_out_count << endl;
		likelihoods_.push_back(Likelihood());
		cout << "Likelihood at iterations-" << iter << ": " << likelihoods_.back() << endl;

		if (resample_iteration_times_ > 0 && (iter % resample_iteration_times_ == 0)) {
			graph_.SampleNegativeEdges(label_, 5, 5);
		}
		
		if (RECORD_PERIOD > 0 && (iter % RECORD_PERIOD == 0)) {
			cout << iter << " iterations. " << TimeString() << endl;
			SaveData(GenerateModelName(iter));
		}

//		shuffle(order.begin(), order.end(), e);
		UpdateF(order);
		// cout << "Update F done. " << TimeString() << endl;
		if (without_eta_ != 1)
			UpdateEta();


	}
	likelihoods_.push_back(Likelihood());
}


// int main(int argc, char const *argv[]) {
// 	return 0;
// }
