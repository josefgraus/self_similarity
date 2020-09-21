#ifndef INTERSECTION_H_
#define INTERSECTION_H_

#include <Eigen/Dense>

namespace intersection {
	bool line_and_line_segment(const Eigen::Vector2d& p, const Eigen::Vector2d& r, const Eigen::Vector2d& q, const Eigen::Vector2d& s, Eigen::Vector2d& nearest_point);
}

#endif