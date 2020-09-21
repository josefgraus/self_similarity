#include "eigen_decomp.h"

#include <chrono>

#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <spectra/SymGEigsSolver.h>
#include <spectra/MatOp/DenseSymMatProd.h>
#include <spectra/MatOp/SparseCholesky.h>

#include <benchmarks/model_manifest.h>
#include <utilities/eigen_read_write_binary.h>

using namespace Spectra;

EigenDecomp::EigenDecomp(std::string benchmark_dir, std::string output_dir, int granularity): _benchmark_dir(benchmark_dir), _output_dir(output_dir), _granularity(granularity) {
}

EigenDecomp::~EigenDecomp() {
}

void EigenDecomp::benchmark() {
	std::vector<std::string> model_paths = obj_manifest_from_directory(_benchmark_dir);

	for (const std::string& model_path : model_paths) {
		std::shared_ptr<Mesh> model = Mesh::instantiate(model_path);

		if (!model->loaded()) {
			continue;
		}

		std::string model_out = _output_dir + model->name() + "\\";

		CreateDirectory(model_out.c_str(), nullptr);

		std::stringstream ss; ss << model_out << "\\benchmark.m";

		std::ofstream benchmark(ss.str(), std::ofstream::out);

		if (!benchmark.is_open()) {
			return;
		}

		benchmark << "close all;" << std::endl;
		benchmark << "clear all;" << std::endl << std::endl;

		benchmark << "decomposition_times = [ ..." << std::endl;;

		// Time partial Spectra eigen-decomposion on each model at each [k]
		int n = model->vertices().rows();

		// igl::cotmatrix returns negated Laplacian, so negate it again
		Eigen::SparseMatrix<double> LM, MM;
		igl::cotmatrix(model->vertices(), model->faces(), LM);
		LM = (-1.0) * LM;

		igl::massmatrix(model->vertices(), model->faces(), igl::MASSMATRIX_TYPE_VORONOI, MM);

		Eigen::MatrixXd L_dense(LM);
		Eigen::MatrixXd M_dense(MM);
		Eigen::MatrixXd M_dense_inv = M_dense.inverse();

		// Time full Eigen eigen-decomposition
		Eigen::MatrixXd full_evecs;
		Eigen::MatrixXd full_evals;
		Eigen::MatrixXd full_hks;
		std::chrono::duration<double> full_elapsed;

		{
			std::cout << "Calculating full eigenstructure... ";

			auto start = std::chrono::system_clock::now();

			Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(L_dense, M_dense_inv, Eigen::ComputeEigenvectors | Eigen::BAx_lx);

			auto end = std::chrono::system_clock::now();

			full_evecs = es.eigenvectors().real();
			full_evals = es.eigenvalues().real();

			std::string evals_path = model_out + "full.lap_evals";
			std::string evecs_path = model_out + "full.lap_evecs";

			write_binary(evals_path.c_str(), full_evals);
			write_binary(evecs_path.c_str(), full_evecs);

			full_elapsed = end - start;

			std::cout << full_elapsed.count() << " s" << std::endl;
		}

		Eigen::VectorXd t_step = (Eigen::VectorXd(1) << t_lower_bound(full_evals)).finished();
		calculate_hks(full_evals, full_evecs, t_step, full_hks);

		std::vector<Eigen::DenseIndex> k_used;
		Eigen::MatrixXd hks_residuals = Eigen::MatrixXd::Zero(n, n);
		int inc = static_cast<int>(std::ceil(static_cast<double>(n) / static_cast<double>(_granularity)));
		for (int k = 1; k < n-1; k+=inc) {
			Eigen::MatrixXd evals;
			Eigen::MatrixXd evecs;

			std::cout << "Calculating partial eigenstructure [ " << k << " ]... ";

			// Partial decomposition only up to k pairs
			// We are going to calculate the eigenvalues of (Lx = )

			// Construct matrix operation object using the wrapper class DenseSymMatProd
			DenseSymMatProd<double> op(L_dense);
			SparseCholesky<double> Bop(MM);
			
			auto start = std::chrono::system_clock::now();

			// Construct eigen solver object, requesting the largest k eigenvalues
			SymGEigsSolver<double, LARGEST_ALGE, DenseSymMatProd<double>, SparseCholesky<double>, GEIGS_CHOLESKY> eigs(&op, &Bop, k, std::min(static_cast<int>(std::ceil(2 * k)), n));

			// Initialize and compute
			eigs.init();
			int nconv = eigs.compute(); 

			auto end = std::chrono::system_clock::now();
				 
			// Retrieve results
			if (eigs.info() == SUCCESSFUL) {
				evals = eigs.eigenvalues().real();
				evecs = eigs.eigenvectors().real();

				std::string evals_path = model_out + std::to_string(k) + ".lap_evals";
				std::string evecs_path = model_out + std::to_string(k) + ".lap_evecs";

				write_binary(evals_path.c_str(), evals);
				write_binary(evecs_path.c_str(), evecs);

				std::chrono::duration<double> elapsed = end - start;

				std::cout << elapsed.count() << " s" << std::endl;

				benchmark << k << ", " << elapsed.count() << "; ..." << std::endl;

				Eigen::MatrixXd partial_hks;
				calculate_hks(evals, evecs, t_step, partial_hks);

				hks_residuals.col(k_used.size()) = (full_hks - partial_hks);

				k_used.push_back(k);
			} else {
				std::cout << "Failed!!!" << std::endl;
			}
		}

		// The full Eigen provided solver will be encoded as (n+1) for accounting purposes
		benchmark << (n) << ", " << full_elapsed.count() << "; ..." << std::endl;
		benchmark << "];" << std::endl;

		hks_residuals.col(k_used.size()) = (full_hks - full_hks);
		k_used.push_back(n);

		hks_residuals.conservativeResize(Eigen::NoChange, k_used.size());

		// Matlab graph the difference of the heat kernel signature as partial approaches full eigendecomposition
		// Surface visualization where Z_kp = |P_kp - F_p|,  P_kp is the HKS value based on k-partial eigencomposition for point p, F_p is the HKS value based on the full eigedecomposition at p
		benchmark << "X = [ ..." << std::endl;
		for (Eigen::DenseIndex i = 0; i < hks_residuals.rows(); ++i) {
			for (Eigen::DenseIndex j = 0; j < k_used.size(); ++j) {
				benchmark << k_used[j];

				if (j + 1 < k_used.size()) {
					benchmark << ", ";
				}
			}
			benchmark << "; ..." << std::endl;
		}
		benchmark << "];" << std::endl;
		 
		benchmark << "Y = transpose(meshgrid(0:" << (n-1) << ", 0:" << (k_used.size() - 1) << "));" << std::endl;

		benchmark << "hks_residuals = [ ..." << std::endl;		
		for (Eigen::DenseIndex i = 0; i < hks_residuals.rows(); ++i) {
			for (Eigen::DenseIndex j = 0; j < hks_residuals.cols(); ++j) {
				benchmark << hks_residuals(i, j);

				if (j + 1 < hks_residuals.cols()) {
					benchmark << ", ";
				} 
			}
			benchmark << "; ..." << std::endl;
		}
		benchmark << "];" << std::endl;

		benchmark << "figure;" << std::endl;
		benchmark << "subplot(2,1,1); % decomposition timing" << std::endl;
		benchmark << "hold on;" << std::endl; 
		benchmark << "plot(decomposition_times(:,1), decomposition_times(:,2));" << std::endl;
		benchmark << "scatter(decomposition_times(:, 1), decomposition_times(:, 2), 'ro');" << std::endl;
		benchmark << "plot([0, decomposition_times(end, 1)], [decomposition_times(end, 2), decomposition_times(end, 2)]);" << std::endl;
		benchmark << "line(max(decomposition_times(:,2)) * 1.20, [0,0]);" << std::endl;
		benchmark << "title('Time taken by eigendecomposition');" << std::endl;
		benchmark << "xlabel('k - number of eigenpairs used');" << std::endl;
		benchmark << "ylabel('time (s)');" << std::endl << std::endl;

		benchmark << "subplot(2,1,2); % HKS residuals" << std::endl;
		benchmark << "hold on;" << std::endl;
		benchmark << "s = surf(X, Y, hks_residuals);" << std::endl;
		benchmark << "s.EdgeColor = 'none';" << std::endl;
		benchmark << "title('HKS residuals between full and k-partial');" << std::endl;
		benchmark << "xlabel('k - number of eigenpairs used');" << std::endl;
		benchmark << "ylabel('vertex ID');" << std::endl;
		benchmark << "zlabel('difference (HKS)');" << std::endl;
		benchmark << "zlim([-5.0, 5.0]);" << std::endl;
		benchmark << "caxis([-5.0, 5.0]);" << std::endl;
		benchmark << "colorbar;" << std::endl;
		benchmark << "colormap cool;" << std::endl;
		benchmark << "view(0,0)" << std::endl << std::endl;

		benchmark.close();
	}
}

void EigenDecomp::calculate_hks(const Eigen::MatrixXd& eigenvalues, const Eigen::MatrixXd& eigenvectors, const Eigen::VectorXd& t_steps, Eigen::MatrixXd& hks) {
	if (!(t_steps.size() > 0)) {
		return;
	}

	// TODO: The next two statements are the most expensive part of the entire operation -- can anything be done to avoid/reduce/optimize them?
	Eigen::MatrixXd evals_t = ((-1.0) * eigenvalues.cwiseAbs() * t_steps.transpose()).array().exp();

	hks = eigenvectors.cwiseProduct(eigenvectors) * evals_t;

	// HKS scaling (TODO: implement Scale-invariant HKS instead of just dividing by the heat trace)
	Eigen::VectorXd heat_trace = evals_t.array().colwise().sum();
	for (unsigned int i = 0; i < heat_trace.size(); ++i) {
		hks.col(i) /= heat_trace(i);
	}
}

double EigenDecomp::t_lower_bound(const Eigen::MatrixXd& evals) const {
	// Heat Kernel Signature
	// [tmin, tmax] as suggested in SOG09: http://dl.acm.org/citation.cfm?id=1735603.1735621
	return 4.0 * std::log(10) / std::fabs(evals(evals.rows() - 1));
}

double EigenDecomp::t_upper_bound(const Eigen::MatrixXd& evals) const {
	// Heat Kernel Signature
	// [tmin, tmax] as suggested in SOG09: http://dl.acm.org/citation.cfm?id=1735603.1735621
	return 4.0 * std::log(10) / std::fabs(evals(1));
}