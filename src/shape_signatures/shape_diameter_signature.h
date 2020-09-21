#ifndef SHAPE_DIAMETER_SIGNATURE_H_
#define SHAPE_DIAMETER_SIGNATURE_H_

#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <igl/shape_diameter_function.h>

#include <shape_signatures/shape_signature.h>
#include <matching/parameterization/discrete_exponential_map.h>

class ShapeDiameterSignature: public ShapeSignature, std::enable_shared_from_this<ShapeDiameterSignature> {
	public:
		~ShapeDiameterSignature();

		static std::shared_ptr<ShapeDiameterSignature> instantiate(std::shared_ptr<Mesh> mesh, double t);

		virtual const Eigen::VectorXd lerpable_coord(Eigen::DenseIndex fid, Eigen::DenseIndex vid);
		virtual Eigen::VectorXd lerpable_to_signature_value(const Eigen::VectorXd& lerped);

		virtual const Eigen::MatrixXd& get_signature_values();
		virtual Eigen::VectorXd get_signature_values(double index);			// Get raw matrix of signature values
		virtual unsigned long feature_count();								// The number of "points" the signature defines values for across the mesh 
		virtual unsigned long feature_dimension();							// The dimensionality of the feature vector defined for each "point" the signature defines values for across the mesh 
		virtual Eigen::MatrixXd sig_steps();
		virtual const double step_width(double param);

		void resample_at_t(double t);
		double t_lower_bound();
		double t_upper_bound();
		virtual double param_lower_bound();
		virtual double param_upper_bound();

		virtual void resample_at_param(double param);

	protected:
		ShapeDiameterSignature(const std::shared_ptr<Mesh> mesh, double t);

		void calc_sdf_data(std::string resource_dir, std::string model_name, int k);

		Eigen::MatrixXd calculate_sdf(const Eigen::VectorXd& t_steps) const;
		Eigen::VectorXd sdf_steps(double tmin, double tmax, int steps) const;

		class SDFParameterOptimization : public ParameterOptimization {
			public:
				SDFParameterOptimization();
				~SDFParameterOptimization();

				virtual double value(const TVector &x);
				virtual TVector upperBound() const;
				virtual TVector lowerBound() const;

				virtual bool callback(const cppoptlib::Criteria<Scalar> &state, const TVector &x);

				virtual Eigen::MatrixXd param_steps(unsigned int steps);
				virtual std::shared_ptr<GeodesicFan> geodesic_fan_from_relation(const Relation& r);

			protected:
				struct Metrics {
					Metrics() {};

					Eigen::DenseIndex _centroid_vid;
					double _geodesic_radius;

					std::shared_ptr<DiscreteExponentialMap> _dem;
				};

				std::unordered_map<std::shared_ptr<Patch>, Metrics> _metrics_map;
		};

		//ShapeDiameterSignature(std::shared_ptr<Mesh> mesh);
		
		double _t;
		Eigen::VectorXd _t_steps;
		Eigen::MatrixXd _sdf_zero;
		
		Eigen::SparseMatrix<double> _M2;
		Eigen::SparseMatrix<double> _QH;
};

#endif
