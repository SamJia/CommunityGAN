#ifndef COMMON_H
#define COMMON_H
#include <string>
#include <cstring>
#include <vector>
#include <ctime>
#include "nnvector.h"
using namespace std;
inline string TimeString() {
	time_t timep;
	time(&timep);
#ifdef _MSC_VER
	char buffer[100];
	ctime_s(buffer, 100, &timep);
	buffer[strlen(buffer) - 1] = '\0';
	return buffer;
#else
	string s = ctime(&timep);
	return s.substr(0, s.length() - 1);
#endif
}

template<typename T>
T VectorDot(const vector<T> &v1, const vector<T> &v2) {
	assert((v1.size() == v2.size()) && "vector's length not match");
	T result = 0;
	for (int i = 0; i < v1.size(); ++i)
		result += v1[i] * v2[i];
	return result;
}

template<typename T>
vector<T> VectorAdd(const vector<T> &v1, const vector<T> &v2) {
	assert((v1.size() == v2.size()) && "vector's length not match");
	vector<T> result(v1.size());
	for (int i = 0; i < v1.size(); ++i)
		result[i] = v1[i] + v2[i];
	return result;
}

template<typename T>
vector<T> VectorAdd(const vector<T> &v1, const vector<T> &v2, vector<T> &v3) {
	assert((v1.size() == v2.size()) && "vector's length not match");
	for (int i = 0; i < v1.size(); ++i)
		v3[i] = v1[i] + v2[i];
}

template<typename T>
void VectorAddTo(const vector<T> &v1, vector<T> &v2) {
	assert((v1.size() == v2.size()) && "vector's length not match");
	for (int i = 0; i < v1.size(); ++i)
		v2[i] += v1[i];
}

template<typename T>
void VectorSubTo(const vector<T> &v1, vector<T> &v2) {
	assert((v1.size() == v2.size()) && "vector's length not match");
	for (int i = 0; i < v1.size(); ++i)
		v2[i] -= v1[i];
}

template<typename T>
void VectorAddTo(const NNVector &v1, vector<T> &v2) {
	assert((v1.size() == v2.size()) && "vector's length not match");
	for (int i = 0; i < v1.idxs_.size(); ++i)
		v2[v1.idxs_[i]] += v1.values_[i];
}

template<typename T>
void VectorSubTo(const NNVector &v1, vector<T> &v2) {
	assert((v1.size() == v2.size()) && "vector's length not match");
	for (int i = 0; i < v1.idxs_.size(); ++i)
		v2[v1.idxs_[i]] -= v1.values_[i];
}

template<typename T>
void PrintVector(string name, vector<T> v) {
	cout << name << ' ';
	for (int i = 0; i < v.size(); ++i)
		cout << v[i] << '\t';
	cout << endl;
}

vector< NNVector > Transpose(const vector< NNVector > &mat);


#endif