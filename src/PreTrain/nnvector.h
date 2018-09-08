#ifndef NNVECTOR_H
#define NNVECTOR_H

#include <algorithm>
#include <vector>
#include <iostream>
#include <functional>
using namespace std;

// Non-negative vector, all the values saved in such vector must be non-negative number.
// Stored by sparse structure.
class NNVector {
  public:
	explicit NNVector(int size) : size_(size) {
		idxs_.reserve(10);
		values_.reserve(10);
	}
	~NNVector() = default;
	inline int size() const {
		return size_;
	}
	void set(int idx, double value);
	double get(int idx) const;
	void clip(double a_min, double a_max);
	void clear();
	void clear_with_threshold(double threshold);
	void clear_to_largest_n(int n);
	vector<double> to_vector() const;

	NNVector operator*(double v) const;
	NNVector operator-(double v) const;
	NNVector & operator*=(double v);
	NNVector & operator-=(double v);

	NNVector operator+(const NNVector &v2) const;
	NNVector operator-(const NNVector &v2) const;
	NNVector operator*(const NNVector &v2) const;
	NNVector operator+(const vector<double> &v2) const;
	NNVector operator-(const vector<double> &v2) const;
	NNVector operator*(const vector<double> &v2) const;
	NNVector & operator+=(const NNVector &v2);
	NNVector & operator-=(const NNVector &v2);
	NNVector & operator*=(const NNVector &v2);
	NNVector & operator+=(const vector<double> &v2);
	NNVector & operator-=(const vector<double> &v2);
	NNVector & operator*=(const vector<double> &v2);

	double dot(const NNVector &v2) const;
	double dot(const vector<double> &v2) const;

	friend ostream & operator<<(ostream &os, const NNVector &v2);
	// friend vector< NNVector > Transpose(const vector< NNVector > &mat);
	// template<typename T>
	// friend void VectorAddTo(const NNVector &v1, vector<T> &v2);
	// template<typename T>
	// friend void VectorSubTo(const NNVector &v1, vector<T> &v2);

	vector<int> idxs_;
	vector<double> values_;
	int size_;
};
#endif