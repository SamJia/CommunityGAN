#include "nnvector.h"
// #define DEBUG

void NNVector::set(int idx, double value) {
#ifdef DEBUG
	if (idx >= size_)
		throw out_of_range("set value out of range");
#endif
	if (value <= 0)
		return;
	if (idxs_.size() == 0 || idxs_.back() < idx) {
		idxs_.push_back(idx);
		values_.push_back(value);
		return;
	}
	int i;
	for (i = 0; i < idxs_.size(); ++i)
		if (idxs_[i] == idx)
			break;
	if (i == idxs_.size()) {
		idxs_.push_back(0);
		values_.push_back(0);
		int j;
		for (j = idxs_.size() - 2; j >= 0 && idxs_[j] > idx; --j) {
			idxs_[j + 1] = idxs_[j];
			values_[j + 1] = values_[j];
		}
		idxs_[j + 1] = idx;
		values_[j + 1] = value;
	} else
		values_[i] = value;
}

double NNVector::get(int idx) const {
#ifdef DEBUG
	if (idx >= size_)
		throw out_of_range("get value out of range");
#endif
	int i;
	for (i = 0; i < idxs_.size(); ++i)
		if (idxs_[i] == idx)
			break;
	return i == idxs_.size() ? 0 : values_[i];
}


void NNVector::clip(double a_min, double a_max) {
	for(int i = 0; i < values_.size(); ++i){
		if (values_[i] < a_min)
			values_[i] = a_min;
		else if(values_[i] > a_max)
			values_[i] = a_max;
	}
}


void NNVector::clear() {
	idxs_.clear();
	values_.clear();
}


void NNVector::clear_with_threshold(double threshold) {
	int i, j;
	for (i = 0, j = 0; i < idxs_.size(); ++i)
		if (values_[i] >= threshold) {
			idxs_[j] = idxs_[i];
			values_[j] = values_[i];
			++j;
		}
	idxs_.resize(j);
	values_.resize(j);
}

void NNVector::clear_to_largest_n(int n) {
#ifdef DEBUG
	if (n <= 0)
		throw domain_error("can not clear to largest 0");
#endif
	if(values_.size() <= n)
		return;
	vector<double> tmp(values_);
	nth_element(tmp.begin(), tmp.begin() + n - 1, tmp.end(), greater<double>());
	clear_with_threshold(tmp[n - 1]);
}

vector<double> NNVector::to_vector() const {
	vector<double> result(size_, 0);
	for (int i = 0; i < idxs_.size(); ++i)
		result[idxs_[i]] = values_[i];
	return result;
}

NNVector NNVector::operator-(double v) const {
#ifdef DEBUG
	if (v < 0)
		throw domain_error("can only minus with non-negative value");
#endif
	if (v == 0)
		return NNVector(*this);
	NNVector result(size_);
	result.idxs_.reserve(idxs_.size());
	result.values_.reserve(idxs_.size());
	for (int i = 0; i < idxs_.size(); ++i)
		if (values_[i] > v) {
			result.idxs_.push_back(idxs_[i]);
			result.values_.push_back(values_[i] - v);
		}
	return result;
}

NNVector NNVector::operator*(double v) const {
#ifdef DEBUG
	if (v < 0)
		throw domain_error("can only multiply with non-negative value");
#endif
	if (v == 0)
		return NNVector(size_);
	NNVector result(size_);
	result.idxs_.reserve(idxs_.size());
	result.values_.reserve(idxs_.size());
	for (int i = 0; i < idxs_.size(); ++i) {
		result.idxs_.push_back(idxs_[i]);
		result.values_.push_back(values_[i] * v);
	}
	return result;
}


NNVector & NNVector::operator-=(double v) {
#ifdef DEBUG
	if (v < 0)
		throw domain_error("can only minus with non-negative value");
#endif
	if (v == 0) {
		idxs_.clear();
		values_.clear();
		return *this;
	}
	int i, j;
	for (i = 0, j = 0; i < idxs_.size(); ++i)
		if (values_[i] > v) {
			idxs_[j] = idxs_[i];
			values_[j] = values_[i] - v;
			++j;
		}
	idxs_.resize(j);
	values_.resize(j);
	return *this;
}

NNVector & NNVector::operator*=(double v) {
#ifdef DEBUG
	if (v < 0)
		throw domain_error("can only multiply with non-negative value");
#endif
	if (v == 0) {
		idxs_.clear();
		values_.clear();
		return *this;
	}
	for (int i = 0; i < values_.size(); ++i) {
		values_[i] *= v;
	}
	return *this;
}

NNVector NNVector::operator+(const NNVector &v2) const {
#ifdef DEBUG
	if (size_ != v2.size_)
		throw invalid_argument("the size of two vectors does not match");
#endif
	NNVector result(size_);
	int i, j;
	result.idxs_.reserve(min(int(idxs_.size() + v2.idxs_.size()), size_));
	result.values_.reserve(min(int(idxs_.size() + v2.idxs_.size()), size_));
	for (i = 0, j = 0; i < idxs_.size() && j < v2.idxs_.size();) {
		if (idxs_[i] < v2.idxs_[j]) {
			result.idxs_.push_back(idxs_[i]);
			result.values_.push_back(values_[i]);
			++i;
		} else if (v2.idxs_[j] < idxs_[i]) {
			result.idxs_.push_back(v2.idxs_[j]);
			result.values_.push_back(v2.values_[j]);
			++j;
		} else {
			result.idxs_.push_back(idxs_[i]);
			result.values_.push_back(values_[i] + v2.values_[j]);
			++i;
			++j;
		}
	}
	result.idxs_.insert(result.idxs_.end(), idxs_.begin() + i, idxs_.end());
	result.values_.insert(result.values_.end(), values_.begin() + i, values_.end());
	// for (; i < idxs_.size(); ++i) {
	// 	result.idxs_.push_back(idxs_[i]);
	// 	result.values_.push_back(values_[i]);
	// }
	for (; j < v2.idxs_.size(); ++j) {
		result.idxs_.push_back(v2.idxs_[j]);
		result.values_.push_back(v2.values_[j]);
	}
	return result;
}

NNVector NNVector::operator-(const NNVector &v2) const {
#ifdef DEBUG
	if (size_ != v2.size_)
		throw invalid_argument("the size of two vectors does not match");
#endif
	NNVector result(size_);
	int i, j;
	for (i = 0, j = 0; i < idxs_.size() && j < v2.idxs_.size();) {
		if (idxs_[i] < v2.idxs_[j]) {
			result.idxs_.push_back(idxs_[i]);
			result.values_.push_back(values_[i]);
			++i;
		} else if (v2.idxs_[j] < idxs_[i]) {
			// result.idxs_.push_back(v2.idxs_[j]);
			// result.values_.push_back(v2.values_[j]);
			++j;
		} else {
			if (values_[i] > v2.values_[j]) {
				result.idxs_.push_back(idxs_[i]);
				result.values_.push_back(values_[i] - v2.values_[j]);
			}
			++i;
			++j;
		}
	}
	for (; i < idxs_.size(); ++i) {
		result.idxs_.push_back(idxs_[i]);
		result.values_.push_back(values_[i]);
	}
	// for (; j < v2.idxs_.size(); ++j) {
	// 	result.idxs_.push_back(v2.idxs_[j]);
	// 	result.values_.push_back(v2.values_[j]);
	// }
	return result;
}

NNVector NNVector::operator*(const NNVector &v2) const {
#ifdef DEBUG
	if (size_ != v2.size_)
		throw invalid_argument("the size of two vectors does not match");
#endif
	NNVector result(size_);
	int i, j;
	for (i = 0, j = 0; i < idxs_.size() && j < v2.idxs_.size();) {
		if (idxs_[i] < v2.idxs_[j]) {
			// result.idxs_.push_back(idxs_[i]);
			// result.values_.push_back(values_[i]);
			++i;
		} else if (v2.idxs_[j] < idxs_[i]) {
			// result.idxs_.push_back(v2.idxs_[j]);
			// result.values_.push_back(v2.values_[j]);
			++j;
		} else {
			result.idxs_.push_back(idxs_[i]);
			result.values_.push_back(values_[i] * v2.values_[j]);
			++i;
			++j;
		}
	}
	// for (; i < idxs_.size(); ++i) {
	// 	result.idxs_.push_back(idxs_[i]);
	// 	result.values_.push_back(values_[i]);
	// }
	// for (; j < v2.idxs_.size(); ++j) {
	// 	result.idxs_.push_back(v2.idxs_[j]);
	// 	result.values_.push_back(v2.values_[j]);
	// }
	return result;
}

NNVector NNVector::operator+(const vector<double> &v2) const {
#ifdef DEBUG
	if (size_ != v2.size())
		throw invalid_argument("the size of two vectors does not match");
#endif
	NNVector result(size_);
	int i, j;
	for (i = 0, j = 0; i < idxs_.size() && j < v2.size();) {
		if (idxs_[i] < j) {
			result.idxs_.push_back(idxs_[i]);
			result.values_.push_back(values_[i]);
			++i;
		} else if (j < idxs_[i]) {
			if (v2[j] > 0) {
				result.idxs_.push_back(j);
				result.values_.push_back(v2[j]);
			}
			++j;
		} else {
			if (values_[i] + v2[j] > 0) {
				result.idxs_.push_back(idxs_[i]);
				result.values_.push_back(values_[i] + v2[j]);
			}
			++i;
			++j;
		}
	}
	for (; i < idxs_.size(); ++i) {
		result.idxs_.push_back(idxs_[i]);
		result.values_.push_back(values_[i]);
	}
	for (; j < v2.size(); ++j) {
		if (v2[j] > 0) {
			result.idxs_.push_back(j);
			result.values_.push_back(v2[j]);
		}
	}
	return result;
}

NNVector NNVector::operator-(const vector<double> &v2) const {
#ifdef DEBUG
	if (size_ != v2.size())
		throw invalid_argument("the size of two vectors does not match");
#endif
	NNVector result(size_);
	int i, j;
	for (i = 0, j = 0; i < idxs_.size() && j < v2.size();) {
		if (idxs_[i] < j) {
			result.idxs_.push_back(idxs_[i]);
			result.values_.push_back(values_[i]);
			++i;
		} else if (j < idxs_[i]) {
			if (v2[j] < 0) {
				result.idxs_.push_back(j);
				result.values_.push_back(-v2[j]);
			}
			++j;
		} else {
			if (values_[i] - v2[j] > 0) {
				result.idxs_.push_back(idxs_[i]);
				result.values_.push_back(values_[i] - v2[j]);
			}
			++i;
			++j;
		}
	}
	for (; i < idxs_.size(); ++i) {
		result.idxs_.push_back(idxs_[i]);
		result.values_.push_back(values_[i]);
	}
	for (; j < v2.size(); ++j) {
		if (v2[j] < 0) {
			result.idxs_.push_back(j);
			result.values_.push_back(-v2[j]);
		}
	}
	return result;
}

NNVector NNVector::operator*(const vector<double> &v2) const {
#ifdef DEBUG
	if (size_ != v2.size())
		throw invalid_argument("the size of two vectors does not match");
#endif
	NNVector result(size_);
	int i;
	for (i = 0; i < idxs_.size(); ++i) {
		if (v2[idxs_[i]] > 0) {
			result.idxs_.push_back(idxs_[i]);
			result.values_.push_back(values_[i] * v2[idxs_[i]]);
		}
	}
	return result;
}


NNVector & NNVector::operator+=(const NNVector &v2) {
	NNVector result = (*this) + v2;
	idxs_.swap(result.idxs_);
	values_.swap(result.values_);
	return *this;
}

NNVector & NNVector::operator-=(const NNVector &v2) {
	NNVector result = (*this) - v2;
	idxs_.swap(result.idxs_);
	values_.swap(result.values_);
	return *this;
}

NNVector & NNVector::operator*=(const NNVector &v2) {
	NNVector result = (*this) * v2;
	idxs_.swap(result.idxs_);
	values_.swap(result.values_);
	return *this;
}


NNVector & NNVector::operator+=(const vector<double> &v2) {
	NNVector result = (*this) + v2;
	idxs_.swap(result.idxs_);
	values_.swap(result.values_);
	return *this;
}

NNVector & NNVector::operator-=(const vector<double> &v2) {
	NNVector result = (*this) - v2;
	idxs_.swap(result.idxs_);
	values_.swap(result.values_);
	return *this;
}

NNVector & NNVector::operator*=(const vector<double> &v2) {
	NNVector result = (*this) * v2;
	idxs_.swap(result.idxs_);
	values_.swap(result.values_);
	return *this;
}

double NNVector::dot(const vector<double> &v2) const {
#ifdef DEBUG
	if (size_ != v2.size())
		throw invalid_argument("the size of two vectors does not match");
#endif
	double result = 0;
	int i;
	for (i = 0; i < idxs_.size(); ++i)
		result += values_[i] + v2[idxs_[i]];
	return result;
}

double NNVector::dot(const NNVector &v2) const {
#ifdef DEBUG
	if (size_ != v2.size_)
		throw invalid_argument("the size of two vectors does not match");
#endif
	double result = 0;
	int size1 = idxs_.size(), size2 = v2.idxs_.size();
	for (int i = 0, j = 0; i < size1 && j < size2;) {
		if (idxs_[i] < v2.idxs_[j])
			++i;
		else if (v2.idxs_[j] < idxs_[i])
			++j;
		else {
			result += values_[i] * v2.values_[j];
			++i;
			++j;
		}
	}
	return result;
}


ostream & operator<<(ostream &os, const NNVector &v2) {
	os << '[';
	for (int i = 0; i < v2.idxs_.size(); ++i) {
		os << '(' << v2.idxs_[i] << ',' << v2.values_[i] << ')';
	}
	os << ']';
	// os << endl;
	return os;
}

// int main(int argc, char const *argv[]) {
// 	NNVector a(4);
// 	a.set(2, 3);
// 	a.set(0, 1);
// 	NNVector b = a - 2;
// 	// cout << a.idxs_.size() << endl;
// 	cout << a << endl;
// 	cout << (a - 1) << endl;
// 	cout << (a * 2) << endl;
// 	cout << (a - b) << endl;
// 	cout << (b - a) << endl;
// 	return 0;
// }
