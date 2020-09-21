#ifndef THRESHOLD_H_
#define THRESHOLD_H_

#include <array>

#include <Eigen/Dense>

class ShapeSignature;

class Threshold {
	public:
		Threshold(double demarc);
		Threshold(double interval_min, double interval_max);
		~Threshold();

		static Threshold from_clipped_normal_dist(double mean, double sigma, double symmetric_clip);
	
		virtual double min();
		virtual double max();
		double midpoint();
		double width();

		virtual bool contains(Eigen::VectorXd value);
		//virtual bool contains(std::shared_ptr<GeodesicFan> fan);
		virtual double distance(Eigen::VectorXd value);
		virtual std::array<Threshold, 2> split(double by_value);

		virtual std::shared_ptr<Threshold> padded_subinterval_about(Eigen::VectorXd value);

	protected:
		Threshold();

		double _interval_min;
		double _interval_max;
};

#endif