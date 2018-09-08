#include "common.h"

vector< NNVector > Transpose(const vector< NNVector > &mat) {
	vector< NNVector > result(mat[0].size(), NNVector(mat.size()));
	for (int i = 0; i < mat.size(); ++i) {
		for (int j = 0; j < mat[i].idxs_.size(); ++j)
			result[mat[i].idxs_[j]].set(i, mat[i].values_[j]);
	}
	return result;
}