#ifndef MODEL_H
#define MODEL_H
#include <vector>
#include <set>
#include "graph.h"
#include "utils.h"
#include "nnvector.h"
#include "label.h"
using namespace std;


class Model {
public:
	Model() = default;
	~Model() = default;
	void Init(Utils & utils);
	void LoadLabel();
	void LoadGraph();
	int DetectCommunityCount(double hold_fraction = 0.1);
	int GetOptimalCommunityCount();
	int CountDifferent(vector<int> v1, vector<int> v2);
	void Resize();
	double Conductance(int cut, int vol);
	void InitNeighbourCommunity(int community_count);
	void CalculateEtaF(vector< NNVector > &eta, vector< NNVector > &etaf);
	void CalculateFEta(vector< NNVector > &eta, vector< NNVector > &feta);

	double Norm2(vector<double> &v);
	double Norm2(vector<vector<double> > &m);
	double Norm2(NNVector &v);
	double Norm2(vector<NNVector > &m);
	void FGradientForRow(int u, NNVector &pos_gradient, NNVector &neg_gradient);
	double FLikelihoodForRow(int u, NNVector &Fu);
	bool FGoStepSizeByLineSearch(int u, NNVector &pos_gradient, NNVector &neg_gradient, const int MAX_ITER = 10);
	void UpdateF(vector<int> &order);
	void EtaGradient(vector< vector<double> > &pos_gradient, vector< vector<double> > &neg_gradient);
	double EtaLikelihood(vector< NNVector > &eta);
	double EtaGoStepSizeByLineSearch(vector<vector<double> > &pos_gradient, vector<vector<double> > &neg_gradient, const int MAX_ITER = 100);
	void UpdateEta();
	double Likelihood();
	double SaveEdgeProbability(string model_name);
	void MLEGradAscent(const int MAX_ITER = -1, const int RECORD_PERIOD = -1);
	string GenerateModelName(int iteration);
	void SaveF(string &model_name);
	void SaveEta(string &model_name);
	void SaveLikelihood(string &model_name);
	void SaveData(string model_name);

private:
	//parameters
	string output_file_perfix_;	// Output Graph data prefix
	string input_graph_filename_;	// Input edgelist file name
	string input_label_filename_;	// Input file name for node labels (Node ID, Node label)
	string input_text_filename_;	// Input file name for node' text (Node ID, Node text)
	int optimal_community_count_;	// The number of communities to detect (-1: detect automatically)
	int min_community_count_;	// Minimum number of communities to try
	int max_community_count_;	// Maximum number of communities to try
	int divide_community_count_;	// How many trials for the number of communities
	int thread_number_;	// Number of threads for parallelization
	bool same_time_ref_; // Determine whether a citation between two papers published at the same time is allowed
	int without_eta_; // if true, eta = I
	bool same_eta_; // if true, etad = etaw
	int max_iteration_times_; // Maximum number of update iteration
	int save_iteration_times_; // How many iterations for once save
	int resample_iteration_times_; // How many iterations for once negative sampling
	double step_alpha_;	// Alpha for backtracking line search
	double step_beta_;	// Beta for backtracking line search
	double MIN_F_; // max value for F
	double MAX_F_; // min value for F
	double MIN_ETA_; // max value for Eta
	double MAX_ETA_; // min value for Eta
	double MIN_P_; // min probability for P(u, v)
	double MAX_P_; // max probability for P(u, v)
	double MIN_GRAD_; // min gradient
	double MAX_GRAD_; // max gradient
	double zero_threshold_; // zero threshold for F and eta
	int largest_n_f_; // remain only largest how many elements for F

	int community_count_; // Number of Community
	Graph graph_;	// Input graph data
	vector< NNVector > F_; // Strength of affiliation edge between community and node, Nodes * Coms
	vector< NNVector > newF_; // new F_, Nodes * Coms
	vector< NNVector > Eta_; // Coms * Coms;
	vector< NNVector > newEta_; // Coms * Coms
	vector< NNVector > EtaF_; // eta times Fv, Nodes * Coms
	vector< NNVector > FEta_; // eta^T times Fv == Fv^T * eta, Nodes * Coms

	vector<Label> label_; // message about each node, Nodes
	map<string, int> name2id_; // convert name to id;

	vector<double> likelihoods_; // record the likelihood after each iteration.
};
#endif
