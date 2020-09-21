#include "threshold.h"

#include <algorithm>

#include <shape_signatures/shape_signature.h>

#undef min
#undef max

Threshold::Threshold() {
}

Threshold::Threshold(double demarc): 
	_interval_min(demarc), 
	_interval_max(demarc) {

}

Threshold::Threshold(double interval_min, double interval_max) {
	_interval_min = std::min(interval_min, interval_max);
	_interval_max = std::max(interval_min, interval_max);
}

Threshold::~Threshold() {
}

Threshold Threshold::from_clipped_normal_dist(double mean, double sigma, double symmetric_clip) {
	double w = (1.0 / std::sqrt(2 * M_PI * std::pow(sigma, 2.0))) * std::exp(-1.0 * std::pow(-1.0 * symmetric_clip, 2) / (2.0 * std::pow(sigma, 2.0)));
	//double upper_coeff = 1.0 - (1.0 / std::sqrt(2 * M_PI * std::pow(sigma, 2.0))) * std::exp(-1.0 * std::pow(symmetric_clip, 2) / (2.0 * std::pow(sigma, 2.0)));

	return Threshold(mean - w, mean + w);
}

double Threshold::min() {
	return _interval_min;
}

double Threshold::max() {
	return _interval_max;
}

double Threshold::midpoint() {
	return _interval_min + ((_interval_max - _interval_min) / 2.0);
}

double Threshold::width() {
	return std::fabs(max() - min());
}

bool Threshold::contains(Eigen::VectorXd value) {
	if (value(0) < _interval_min && std::fabs(value(0) - _interval_min) > std::numeric_limits<double>::epsilon()) {
		return false;
	}

	if (value(0) > _interval_max && std::fabs(value(0) - _interval_max) > std::numeric_limits<double>::epsilon()) {
		return false;
	}

	return true;
}

/*bool Threshold::contains(std::shared_ptr<GeodesicFan> fan) {
	throw std::exception("Not implemented!");
}*/

double Threshold::distance(Eigen::VectorXd value) {
	if (contains(value)) {
		return 0.0;
	} 

	return std::min(std::fabs(_interval_min - value(0)), std::fabs(_interval_max - value(0)));
}

std::array<Threshold, 2> Threshold::split(double by_value) {
	// clamp bisection value
	by_value = std::min(_interval_max, std::max(_interval_min, by_value));

	double a = std::min(_interval_min, by_value);
	double b = std::max(_interval_min, by_value);
	double c = std::min(_interval_max, by_value);
	double d = std::max(_interval_max, by_value);

	std::array<Threshold, 2> bisection = { Threshold(a, b), Threshold(c, d) };

	return bisection;
}

std::shared_ptr<Threshold> Threshold::padded_subinterval_about(Eigen::VectorXd value) {
	double width = std::min(std::fabs(value(0) - min()), std::fabs(value(0) - max())) / 2.0;

	return std::make_shared<Threshold>(value(0) - width, value(0) + width);
}