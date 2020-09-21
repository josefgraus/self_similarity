#ifndef GEODESIC_FAN_H_
#define GEODESIC_FAN_H_

#include <memory>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

class Patch;
class BarycentricCoord;
class ShapeSignature;
class DiscreteExponentialMap;

// A blade is just an opposed pair of spokes, used for assymetric fans
// As such, a blade is rotated from its center, but still requires a 2 * Pi magnitude rotation due to asymmetry
class GeodesicFanBlade {
	public:
		typedef Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic> SignatureTensor;

		GeodesicFanBlade() { };
		~GeodesicFanBlade() { };

		virtual const std::vector<BarycentricCoord>& blade_points() const = 0;
		virtual Eigen::MatrixXd parameterized_space_points_2d(std::shared_ptr<Patch>* cover_patch = nullptr, std::shared_ptr<DiscreteExponentialMap>* origin_map = nullptr, Eigen::DenseIndex center_vid = -1) const = 0;
		virtual SignatureTensor blade_values(double angle_step, std::shared_ptr<ShapeSignature> sig, std::shared_ptr<DiscreteExponentialMap>* dem = nullptr) = 0;
};

class GeodesicFan {
	public:
		GeodesicFan(const std::shared_ptr<DiscreteExponentialMap> dem, const std::shared_ptr<ShapeSignature> signature); 
		GeodesicFan(double angle_step, double radius, double radius_step, const std::shared_ptr<DiscreteExponentialMap> dem, const std::shared_ptr<ShapeSignature> signature, bool normalized = false);
		GeodesicFan(double angle_step, double radius, double radius_step, const std::shared_ptr<DiscreteExponentialMap> dem, const std::shared_ptr<ShapeSignature> signature, Eigen::Vector3d encode_up, bool normalized = false);
		GeodesicFan(std::shared_ptr<GeodesicFanBlade> custom_blade, double angle_step, const std::shared_ptr<ShapeSignature> signature);
		~GeodesicFan();

		const Eigen::VectorXd& operator()(Eigen::DenseIndex i, Eigen::DenseIndex j);
		friend std::ostream& operator<<(std::ostream &out, const GeodesicFan &fan);

		const Eigen::VectorXd& raw_fan_center() const { return _center; }
		const Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic>& raw_fan_data() const { return _fan; };

		double angle_step();
		double radius();
		double radius_step();

		unsigned int spokes() const;
		unsigned int levels() const;

		double lower_bound();
		double upper_bound();

		std::shared_ptr<DiscreteExponentialMap> origin_map() { return _origin_map; }

		Eigen::MatrixXd get_fan_vertices() const;
		Eigen::VectorXd get_fan_values(unsigned int layer) const;

		bool layer_over(std::shared_ptr<GeodesicFan> fan, double orientation = 0.0);

		double compare(const GeodesicFan& other, double& orientation);
		double compare(const GeodesicFanBlade::SignatureTensor& other, double& orientation);
		Eigen::VectorXd aligned_vector(double orientation);

		static std::shared_ptr<GeodesicFan> from_aligned_vector(Eigen::VectorXd vec, unsigned int spokes, unsigned int levels, unsigned int sig_dim);

		double l2_norm();
		double scaled_l2_norm();
		double geometric_mean();

	private:
		void populate(const std::shared_ptr<DiscreteExponentialMap> dem, const std::shared_ptr<ShapeSignature> signature, Eigen::Vector3d encode_up = { 0.0, 0.0, 0.0 });

		std::shared_ptr<GeodesicFanBlade> _custom_blade;

		Eigen::VectorXd _center;
		Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic> _fan;
		double _angle_step;
		double _radius;
		double _radius_step;
		Eigen::Matrix3d _TBN;
		std::shared_ptr<DiscreteExponentialMap> _origin_map;

		double _sig_min;
		double _sig_max;
		bool _normalized;
};

#endif
