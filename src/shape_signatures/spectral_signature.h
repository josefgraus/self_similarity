#ifndef SPECTRAL_SIGNATURE_H_
#define SPECTRAL_SIGNATURE_H_

#include <memory>
#include <string>

#include <Eigen/Sparse>

#include <geometry/mesh.h>
#include <shape_signatures/shape_signature.h>

class SpectralSignature: public ShapeSignature {
	public:
		virtual ~SpectralSignature();

		virtual const Eigen::Block<const Eigen::MatrixXd> eigenvectors() const;
		virtual const Eigen::Block<const Eigen::MatrixXd> eigenvalues() const;

	protected:
		SpectralSignature(std::shared_ptr<ParameterOptimization> param_opt, const std::shared_ptr<Mesh> mesh, int k);

		void calc_laplacian_eigen_pairs(std::string resource_dir, std::string model_name, int k);
		bool calc_new_partial_pairs(std::string resource_dir, std::string model_name, int k);
		bool calc_new_full_pairs(std::string resource_dir, std::string model_name);

		void resample_k(int k);
		int get_current_k();

	private:
		Eigen::SparseMatrix<double> _laplacian_matrix, _mass_matrix;
		Eigen::MatrixXd _eigen_vecs_lap;
		Eigen::MatrixXd _eigen_vals_lap;
		int _k;
		int _max_k;
};

#endif