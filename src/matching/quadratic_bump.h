#ifndef QUADRATIC_BUMP_H_
#define QUADRATIC_BUMP_H_

#include <Eigen\Dense>

template <class T>
class QuadraticBump {
	public:
		QuadraticBump();
		QuadraticBump(Eigen::DenseIndex feature_index, T mean, T width, T magnitude);
		~QuadraticBump();

		Eigen::DenseIndex feature_index();

		double energy_shift_by_parameter(T index);

	private:
		double falloff(T t);

		Eigen::DenseIndex _feature_index;
		T _mean;
		T _width;
		T _magnitude;
};

#include <matching\quadratic_bump.cpp>

#endif
