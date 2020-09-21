#ifndef WAVE_KERNEL_SIGNATURE
#define WAVE_KERNEL_SIGNATURE

#include <vector>
#include <unordered_map>

#include <shape_signatures/spectral_signature.h>
#include <matching/geodesic_fan.h>
#include <matching/threshold.h>

class WaveKernelSignature : public SpectralSignature {
	public:
		enum class Parameters {
			eMin,
			eMax,
			Steps
		};

		virtual ~WaveKernelSignature();

		static std::shared_ptr<WaveKernelSignature> instantiate(std::shared_ptr<Mesh> mesh, int k);
		static std::shared_ptr<WaveKernelSignature> instantiate(std::shared_ptr<Mesh> mesh, int steps, int k);
		static std::shared_ptr<WaveKernelSignature> instantiate(std::shared_ptr<Mesh> mesh, double eMin, double eMax, int k);
		static std::shared_ptr<WaveKernelSignature> instantiate(std::shared_ptr<Mesh> mesh, double eMin, double eMax, int steps, int k);

		void resample_at_e(double e, int k);
		friend std::shared_ptr<WaveKernelSignature> operator-(std::shared_ptr<WaveKernelSignature> lhs, std::shared_ptr<WaveKernelSignature> rhs);

		virtual const Eigen::VectorXd lerpable_coord(Eigen::DenseIndex fid, Eigen::DenseIndex vid);
		virtual Eigen::VectorXd lerpable_to_signature_value(const Eigen::VectorXd& lerped);

		virtual unsigned long feature_count();
		virtual unsigned long feature_dimension();
		virtual const Eigen::MatrixXd& get_signature_values();
		virtual Eigen::VectorXd get_signature_values(double index);
		int get_k_pairs_used();

		unsigned int get_steps();
		double get_emin();
		double get_emax();
		virtual Eigen::MatrixXd sig_steps();
		virtual const double step_width(double param);

		virtual double param_lower_bound();
		virtual double param_upper_bound();

		virtual void resample_at_param(double param);

	protected:
	class WKSParameterOptimization : public ParameterOptimization {
		public:
			WKSParameterOptimization();
			~WKSParameterOptimization();

			virtual double value(const TVector &x);
			virtual TVector upperBound() const;
			virtual TVector lowerBound() const;

			virtual bool callback(const cppoptlib::Criteria<Scalar> &state, const TVector &x);

			virtual Eigen::MatrixXd param_steps(unsigned int steps);
			virtual std::shared_ptr<GeodesicFan> geodesic_fan_from_relation(const Relation& r);

		protected:
			struct Metrics {
				Metrics() { };

				Eigen::DenseIndex _centroid_vid;
				double _geodesic_radius;

				std::shared_ptr<DiscreteExponentialMap> _dem;
			};

			std::unordered_map<std::shared_ptr<Patch>, Metrics> _metrics_map;
	};

	WaveKernelSignature(std::shared_ptr<Mesh> mesh, int k);
	WaveKernelSignature(std::shared_ptr<Mesh> mesh, int steps, int k);
	WaveKernelSignature(std::shared_ptr<Mesh> mesh, double eMin, double eMax, int k);
	WaveKernelSignature(std::shared_ptr<Mesh> mesh, double eMin, double eMax, int steps, int k);

	Eigen::MatrixXd calculate_wks(const Eigen::VectorXd& t_steps) const;
	Eigen::VectorXd wks_steps(double emin, double emax, int steps) const;
	double e_lower_bound() const;
	double e_upper_bound() const;

	Eigen::VectorXd _e_steps;
};

#endif
