#ifndef EIGEN_DECOMP_H_
#define EIGEN_DECOMP_H_

#include <string>
#include <map>
#include <memory>

#include <shape_signatures/heat_kernel_signature.h>

struct BenchmarkMetrics {
	double _partial_timing;
	double _full_timing;
	std::shared_ptr<HeatKernelSignature> _residuals;
};

class EigenDecomp {
	public:
		EigenDecomp(std::string benchmark_dir, std::string output_dir, int granularity);
		~EigenDecomp();

		void benchmark();

	private:
		void calculate_hks(const Eigen::MatrixXd& eigenvalues, const Eigen::MatrixXd& eigenvectors, const Eigen::VectorXd& t_steps, Eigen::MatrixXd& hks);
		double t_lower_bound(const Eigen::MatrixXd& evals) const;
		double t_upper_bound(const Eigen::MatrixXd& evals) const;

		std::map<std::string, BenchmarkMetrics> _metrics;

		std::string _benchmark_dir;
		std::string _output_dir;
		int _granularity;
};

#endif