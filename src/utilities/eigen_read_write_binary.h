#ifndef _EIGEN_READ_WRITE_BINARY
#define _EIGEN_READ_WRITE_BINARY

#include <fstream>

// Convenience functions for saving/loading an Eigen matrix to/from a binary file
template<class Matrix>
void write_binary(const char* filename, const Matrix& matrix) {
	std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
	typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
	out.write((char*)(&rows), sizeof(typename Matrix::Index));
	out.write((char*)(&cols), sizeof(typename Matrix::Index));
	out.write((char*)matrix.data(), rows*cols * sizeof(typename Matrix::Scalar));
	out.close();
}

template<class Matrix>
void read_binary(const char* filename, Matrix& matrix) {
	std::ifstream in(filename, std::ios::in | std::ios::binary);
	typename Matrix::Index rows = 0, cols = 0;
	in.read((char*)(&rows), sizeof(typename Matrix::Index));
	in.read((char*)(&cols), sizeof(typename Matrix::Index));
	matrix.resize(rows, cols);
	in.read((char *)matrix.data(), rows*cols * sizeof(typename Matrix::Scalar));
	in.close();
}


template <typename T, int I, typename IND>
void write_binary(const char* filename, Eigen::SparseMatrix<T, I, IND>& m) {
	std::vector<Eigen::Triplet<int>> res;
	int sz = m.nonZeros();
	m.makeCompressed();

	std::ofstream writeFile;
	writeFile.open(filename, std::ios::binary);

	if (writeFile.is_open()) {
		IND rows, cols, nnzs, outS, innS;
		rows = m.rows();
		cols = m.cols();
		nnzs = m.nonZeros();
		outS = m.outerSize();
		innS = m.innerSize();

		writeFile.write((const char *)&(rows), sizeof(IND));
		writeFile.write((const char *)&(cols), sizeof(IND));
		writeFile.write((const char *)&(nnzs), sizeof(IND));
		writeFile.write((const char *)&(outS), sizeof(IND));
		writeFile.write((const char *)&(innS), sizeof(IND));

		writeFile.write((const char *)(m.valuePtr()), sizeof(T) * m.nonZeros());
		writeFile.write((const char *)(m.outerIndexPtr()), sizeof(IND) * m.outerSize());
		writeFile.write((const char *)(m.innerIndexPtr()), sizeof(IND) * m.nonZeros());

		writeFile.close();
	}
}

template <typename T, int I, typename IND>
void read_binary(const char* filename, Eigen::SparseMatrix<T, I, IND>& m) {
	std::ifstream readFile;
	readFile.open(filename, std::ios::binary);

	if (readFile.is_open()) {
		IND rows, cols, nnz, inSz, outSz;
		readFile.read((char*)&rows, sizeof(IND));
		readFile.read((char*)&cols, sizeof(IND));
		readFile.read((char*)&nnz, sizeof(IND));
		readFile.read((char*)&inSz, sizeof(IND));
		readFile.read((char*)&outSz, sizeof(IND));

		m.resize(rows, cols);
		m.makeCompressed();
		m.resizeNonZeros(nnz);

		readFile.read((char*)(m.valuePtr()), sizeof(T) * nnz);
		readFile.read((char*)(m.outerIndexPtr()), sizeof(IND) * outSz);
		readFile.read((char*)(m.innerIndexPtr()), sizeof(IND) * nnz);

		m.finalize();
		readFile.close();
	} 
}

#endif