#ifndef SHAPE_SIGNATURE_H_
#define SHAPE_SIGNATURE_H_

#include <memory>
#include <map>

#include <Eigen/Dense>
#include <cppoptlib/problem.h>

#include <matching/desire_set.h>

class Patch;
class Mesh;
class Threshold;
class CRSolver;
class GeodesicFan;
struct Relation;

template <class T>
class QuadraticBump;

class ShapeSignature {
	public:
		virtual ~ShapeSignature();

		std::shared_ptr<Mesh> origin_mesh();

		virtual const Eigen::VectorXd lerpable_coord(Eigen::DenseIndex fid, Eigen::DenseIndex vid) = 0;
		virtual Eigen::VectorXd lerpable_to_signature_value(const Eigen::VectorXd& lerped) = 0;

		virtual const Eigen::MatrixXd& get_signature_values();					// Get raw matrix of signature values
		virtual Eigen::VectorXd get_signature_values(double index) = 0;			// Get copy of column of raw matrix of signature values
		virtual unsigned long feature_count() = 0;								// The number of "points" the signature defines values for across the mesh 
		virtual unsigned long feature_dimension() = 0;							// The dimensionality of the feature vector defined for each "point" the signature defines values for across the mesh 
		virtual double lower_bound();
		virtual double upper_bound();
		virtual double param_lower_bound() = 0;
		virtual double param_upper_bound() = 0;
		virtual Eigen::MatrixXd sig_steps() = 0;
		virtual const double step_width(double param) = 0;

		virtual void resample_at_param(double param) = 0;

		void apply_quadratic_bump(QuadraticBump<double> bump);
		void clear_quadratic_bumps();

	protected:
		friend class CRSolver;
		friend void signature_value_proxy(int id, int index, std::shared_ptr<ShapeSignature> sig, Eigen::VectorXd x, std::shared_ptr<Eigen::MatrixXd> out);

		class ParameterOptimization : public cppoptlib::Problem<double> {
			public:
				using typename cppoptlib::Problem<double>::Scalar;
				using typename cppoptlib::Problem<double>::TVector;

				ParameterOptimization();
				~ParameterOptimization();

				void bind_signature(std::shared_ptr<ShapeSignature> optimizing_sig);

				void set_value_desire_set(std::vector<Relation> desire_set);

				virtual double value(const TVector &x) = 0;
				//virtual void gradient(const TVector &x, TVector& grad) = 0;	// Let cppoptlib use finite differencing for now until a gradient can be provided
				virtual TVector upperBound() const = 0;
				virtual TVector lowerBound() const = 0;

				virtual Eigen::MatrixXd param_steps(unsigned int steps) = 0;
				virtual std::shared_ptr<GeodesicFan> geodesic_fan_from_relation(const Relation& r) = 0;

			protected:
				std::weak_ptr<ShapeSignature> _optimizing_sig;
				std::vector<Relation> _value_desire_set;
		};

		ShapeSignature(std::shared_ptr<Mesh> mesh, std::shared_ptr<ParameterOptimization> param_opt);

		std::shared_ptr<Mesh> _mesh;
		Eigen::MatrixXd _sig;
		std::shared_ptr<ParameterOptimization> _param_opt;
		Eigen::MatrixXd _exception_map;
		Eigen::MatrixXd _exception_filtered;
		std::vector<QuadraticBump<double>> _bumps;
};

#endif