#include "spherical_threshold.h"

#include <matching/geodesic_fan.h>

SphericalThreshold::SphericalThreshold() {
}

SphericalThreshold::SphericalThreshold(Eigen::VectorXd disc_center, double max_angle, double min_angle):
	Threshold(min_angle, max_angle),
	_disc_center(disc_center) {

	//_disc_center_fan = GeodesicFan::from_aligned_vector(_disc_center);

	if (_disc_center_fan == nullptr) {
		throw std::domain_error("_disc_center_fan is invalid!");
	}
}

SphericalThreshold::~SphericalThreshold() {
}

bool SphericalThreshold::contains(Eigen::VectorXd value) {
	double angle = std::acos(std::min(1.0, std::max(0.0, _disc_center.dot(value.normalized()))));

	if (_interval_min <= angle && angle <= _interval_max) {
		return true;
	}

	return false;
}

bool SphericalThreshold::contains(std::shared_ptr<GeodesicFan> fan) {
	if (fan == nullptr) {
		return false;
	}

	double orientation;
	double comp = _disc_center_fan->compare(*fan, orientation);

	Eigen::VectorXd vec = fan->aligned_vector(orientation);

	return contains(vec);
}

double SphericalThreshold::distance(Eigen::VectorXd value) {
	double angle = std::acos(std::min(1.0, std::max(0.0, _disc_center.dot(value.normalized()))));

	if (_interval_min <= angle && angle <= _interval_max) {
		return 0.0;
	}

	if (angle < _interval_min) {
		return std::max(0.0, _interval_min - angle);
	}

	return std::min(2.0 * M_PI, angle - _interval_max);
}

std::shared_ptr<Threshold> SphericalThreshold::padded_subinterval_about(Eigen::VectorXd value) {
	throw std::logic_error("Not implemented!");
}