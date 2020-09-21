#ifndef SPHERICAL_THRESHOLD_H_
#define SPHERICAL_THRESHOLD_H_

#include <matching/threshold.h>

class GeodesicFan;

class SphericalThreshold : public Threshold {
	public:
		SphericalThreshold(Eigen::VectorXd disc_center, double max_angle, double min_angle = 0.0);
		~SphericalThreshold();

		virtual bool contains(Eigen::VectorXd value);
		virtual bool contains(std::shared_ptr<GeodesicFan> fan);
		virtual double distance(Eigen::VectorXd value);

		virtual std::shared_ptr<Threshold> padded_subinterval_about(Eigen::VectorXd value);

	protected:
		SphericalThreshold();

		Eigen::VectorXd _disc_center;
		std::shared_ptr<GeodesicFan> _disc_center_fan;
};

#endif