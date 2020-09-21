#include "spectral_signature.h"

#ifdef _WIN32
#include <io.h> 
#define access    _access_s
#else
#include <unistd.h>
#endif

#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <spectra/SymGEigsSolver.h>
#include <spectra/MatOp/DenseSymMatProd.h>
#include <spectra/MatOp/SparseCholesky.h>

#include <geometry/mesh.h>
#include <utilities/eigen_read_write_binary.h>

using namespace Spectra;

SpectralSignature::SpectralSignature(std::shared_ptr<ParameterOptimization> param_opt, const std::shared_ptr<Mesh> mesh, int k): 
	ShapeSignature(mesh, param_opt),
	_laplacian_matrix(0,0),
	_mass_matrix(0,0),
	_eigen_vecs_lap(Eigen::MatrixXd::Zero(0,0)),
	_eigen_vals_lap(Eigen::MatrixXd::Zero(0,0)),
	_k(-1),
	_max_k(-1) {

	if (mesh == nullptr) {
		// Empty model, so empty signature
		return;
	}

	if (!mesh->loaded()) {
		// The mesh failed to load
		return;
	}

	if (mesh->vertices().size() <= 0 || mesh->faces().size() <= 0) {
		// Mesh is insufficient for the purposes of this class of signatures
		return;
	}

	// igl::cotmatrix returns negated Laplacian, so negate it again
	igl::cotmatrix(mesh->vertices(), mesh->faces(), _laplacian_matrix);
	_laplacian_matrix = (-1.0) * _laplacian_matrix;

	igl::massmatrix(mesh->vertices(), mesh->faces(), igl::MASSMATRIX_TYPE_VORONOI, _mass_matrix);

	// Eigenvalues smallest to largest and corresponding vectors
	calc_laplacian_eigen_pairs(mesh->resource_dir(), mesh->name(), k);
}

SpectralSignature::~SpectralSignature() {
}

void SpectralSignature::calc_laplacian_eigen_pairs(std::string resource_dir, std::string model_name, int k) {
	// Attempt to load precalculated file
	std::string evals_path = resource_dir + "//cache//" + model_name + ".lap_evals";
	std::string evecs_path = resource_dir + "//cache//" + model_name + ".lap_evecs";
	
	bool calculated = false;
	if (!(access(evals_path.c_str(), 0) == 0) || !(access(evecs_path.c_str(), 0) == 0)) {
		// Files don't exist, or are inaccessible
		if (k <= 0 || k >= origin_mesh()->vertices().rows()) {
			calculated = calc_new_full_pairs(evals_path, evecs_path);
		} else {
			calculated = calc_new_partial_pairs(evals_path, evecs_path, k);
		}

		if (calculated) {
			_k = _max_k = _eigen_vals_lap.rows();

			write_binary(evals_path.c_str(), _eigen_vals_lap);
			write_binary(evecs_path.c_str(), _eigen_vecs_lap);
		} else {
			_k = _max_k = 0;
		}
	} else {
		// Else, use precalculated matrices
		std::cout << "Loading saved eigenstructure..." << std::endl;
		read_binary(evals_path.c_str(), _eigen_vals_lap);
		read_binary(evecs_path.c_str(), _eigen_vecs_lap);

		_max_k = _eigen_vals_lap.rows();

		if ((_max_k < _k && _k <= origin_mesh()->vertices().rows()) || 
			((_k <= 0 || _k > origin_mesh()->vertices().rows()) && _max_k < origin_mesh()->vertices().rows())) {
			// Eigendecomposition is insufficient for the requested k
			if (k <= 0 || k >= origin_mesh()->vertices().rows()) {
				calculated = calc_new_full_pairs(evals_path, evecs_path);
			} else {
				calculated = calc_new_partial_pairs(evals_path, evecs_path, k);
			}

			if (calculated) {
				_k = _max_k = _eigen_vals_lap.rows();

				write_binary(evals_path.c_str(), _eigen_vals_lap);
				write_binary(evecs_path.c_str(), _eigen_vecs_lap);
			} else {
				_k = _max_k = 0;
			}
		} else if (_max_k < _k || _k <= 0) {
			_k = _max_k;
		}
	}
}

bool SpectralSignature::calc_new_partial_pairs(std::string evals_path, std::string evecs_path, int k) {
	// Partial decomposition only up to k pairs
	// We are going to calculate the eigenvalues of (M^-1 L)
	Eigen::MatrixXd L_dense(_laplacian_matrix);

	// Construct matrix operation object using the wrapper class DenseSymMatProd
	// Construct matrix operation object using the wrapper class DenseSymMatProd
	DenseSymMatProd<double> op(L_dense);
	SparseCholesky<double> Bop(_mass_matrix);

	// Construct eigen solver object, requesting the largest k eigenvalues
	SymGEigsSolver<double, LARGEST_ALGE, DenseSymMatProd<double>, SparseCholesky<double>, GEIGS_CHOLESKY> eigs(&op, &Bop, k, std::min(static_cast<int>(std::ceil(2 * k)), static_cast<int>(L_dense.rows())));

	// Initialize and compute
	auto start = std::chrono::system_clock::now();

	std::cout << "Calculating partial eigenstructure..." << std::endl;

	eigs.init();
	int nconv = eigs.compute();

	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed = end - start;

	std::cout << "Done! Elapsed time: " << elapsed.count() << std::endl;

	// Retrieve results
	if (eigs.info() == SUCCESSFUL) {
		_eigen_vals_lap = eigs.eigenvalues();
		_eigen_vecs_lap = eigs.eigenvectors();
	}

	return (eigs.info() == SUCCESSFUL);
}

bool SpectralSignature::calc_new_full_pairs(std::string evals_path, std::string evecs_path) {
	// Full decomposition
	Eigen::MatrixXd L_dense(_laplacian_matrix);
	Eigen::MatrixXd M_dense(_mass_matrix);
	Eigen::MatrixXd M_dense_inv = M_dense.inverse();

	auto start = std::chrono::system_clock::now();

	std::cout << "Calculating full eigenstructure..." << std::endl;

	Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(L_dense, M_dense_inv, Eigen::ComputeEigenvectors | Eigen::BAx_lx);
	
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed = end - start;

	std::cout << "Done! Elapsed time: " << elapsed.count() << std::endl;

	_eigen_vecs_lap = es.eigenvectors().real();
	_eigen_vals_lap = es.eigenvalues().real();

	return true;
}

const Eigen::Block<const Eigen::MatrixXd> SpectralSignature::eigenvectors() const {
	if (_k > 0 && _k < _max_k) {
		return _eigen_vecs_lap.block(0, _eigen_vecs_lap.cols() - _k, _eigen_vecs_lap.rows(), _k);
	} 
	
	return _eigen_vecs_lap.block(0, 0, _eigen_vecs_lap.rows(), _eigen_vecs_lap.cols());
}

const Eigen::Block<const Eigen::MatrixXd> SpectralSignature::eigenvalues() const {
	if (_k > 0 && _k < _max_k) {
		return _eigen_vals_lap.block(_eigen_vals_lap.rows() - _k, 0, _k, 1);
	}

	return _eigen_vals_lap.block(0, 0, _eigen_vals_lap.rows(), _eigen_vals_lap.cols());
}

void SpectralSignature::resample_k(int k) {
	_k = k;

	if (_k > _max_k || _k <= 0) {
		calc_laplacian_eigen_pairs(origin_mesh()->resource_dir(), origin_mesh()->name(), k);
	}
}

int SpectralSignature::get_current_k() {
	return _k;
}