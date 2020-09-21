#ifndef HEAT_KERNEL_SIGNATURE
#define HEAT_KERNEL_SIGNATURE

#include <vector>
#include <unordered_map>

#include <shape_signatures/spectral_signature.h>
#include <matching/geodesic_fan.h>
#include <matching/threshold.h>

class HeatKernelSignature: public SpectralSignature {
	public:
		enum class Parameters {
			tMin,
			tMax,
			Steps
		};

		virtual ~HeatKernelSignature();

		static std::shared_ptr<HeatKernelSignature> instantiate(std::shared_ptr<Mesh> mesh, int k);
		static std::shared_ptr<HeatKernelSignature> instantiate(std::shared_ptr<Mesh> mesh, int steps, int k);
		static std::shared_ptr<HeatKernelSignature> instantiate(std::shared_ptr<Mesh> mesh, double tMin, double tMax, int k);
		static std::shared_ptr<HeatKernelSignature> instantiate(std::shared_ptr<Mesh> mesh, double tMin, double tMax, int steps, int k);

		void resample_at_t(double t, int k);
		friend std::shared_ptr<HeatKernelSignature> operator-(std::shared_ptr<HeatKernelSignature> lhs, std::shared_ptr<HeatKernelSignature> rhs);

		virtual const Eigen::VectorXd lerpable_coord(Eigen::DenseIndex fid, Eigen::DenseIndex vid);
		virtual Eigen::VectorXd lerpable_to_signature_value(const Eigen::VectorXd& lerped);

		virtual unsigned long feature_count();
		virtual unsigned long feature_dimension();
		virtual const Eigen::MatrixXd& get_signature_values();
		virtual Eigen::VectorXd get_signature_values(double index);
		int get_k_pairs_used();

		unsigned int get_steps();
		double get_tmin();
		double get_tmax();
		virtual Eigen::MatrixXd sig_steps();
		virtual const double step_width(double param);

		virtual double param_lower_bound();
		virtual double param_upper_bound();

		virtual void resample_at_param(double param);

	protected:
		class HKSParameterOptimization : public ParameterOptimization {
			public:
				HKSParameterOptimization();
				~HKSParameterOptimization();

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

		HeatKernelSignature(std::shared_ptr<Mesh> mesh, int k);
		HeatKernelSignature(std::shared_ptr<Mesh> mesh, int steps, int k);
		HeatKernelSignature(std::shared_ptr<Mesh> mesh, double tMin, double tMax, int k);
		HeatKernelSignature(std::shared_ptr<Mesh> mesh, double tMin, double tMax, int steps, int k);

		Eigen::MatrixXd calculate_hks(const Eigen::VectorXd& t_steps) const;
		Eigen::VectorXd hks_steps(double tmin, double tmax, int steps) const;
		double t_lower_bound() const;
		double t_upper_bound() const;

		Eigen::VectorXd _t_steps;
};

#endif
