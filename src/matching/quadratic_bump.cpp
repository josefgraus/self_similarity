//#include "quadratic_bump.h"

#include <cmath>

template <class T>
QuadraticBump<T>::QuadraticBump(): _feature_index(-1), _mean(0.0), _width(1.0), _magnitude(0.0) {
}

template <class T>
QuadraticBump<T>::QuadraticBump(Eigen::DenseIndex feature_index, T mean, T width, T magnitude): _feature_index(feature_index), _mean(mean), _width(width), _magnitude(magnitude) {
}

template <class T>
QuadraticBump<T>::~QuadraticBump() {
}

template <class T>
Eigen::DenseIndex QuadraticBump<T>::feature_index() {
	return _feature_index;
}

template <class T>
double QuadraticBump<T>::energy_shift_by_parameter(T index) {
	return falloff(index) * _magnitude;
}

template <class T>
double QuadraticBump<T>::falloff(T t) {
	t -= _mean;

	double f;
	const double a = 1.0;
	t = t / _width;

	double p = std::fabs(t);

	if (p < a / 3.0) {
		f = 3.0 * (p*p) / (a*a);
	} else if (p < a) {
		f = -1.5 * (p*p) / (a*a) + 3.0 * p / a - 0.5;
	} else {
		f = 1.0;
	}

	return 1.0 - f;
}
