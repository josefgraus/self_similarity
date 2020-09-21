#include "intersection.h"

namespace intersection {
	/* 2D vector cross product analog.
	   The cross product of 2D vectors results in a 3D vector with only a z component.
	   This function returns the magnitude of the z value.
	   */
	static inline double cross2d(const Eigen::Vector2d& v1, const Eigen::Vector2d& v2) {
		return v1.x() * v2.y() - v1.y() * v2.x();
	}

	/* Find 2D intersection of line and line segment
	   p -- first line segment start point
	   r -- first line segment vector (length encoded)
	   p -- second line segment start point
	   r -- second line segment vector (length encoded)
	   nearest_point -- a point where the two intersect, or the nearest point between the two
	   returns true if intersect, false if otherwise 
	   */
	bool line_and_line_segment(const Eigen::Vector2d& p, const Eigen::Vector2d& r, const Eigen::Vector2d& q, const Eigen::Vector2d& s, Eigen::Vector2d& nearest_point) {
		double rxs = cross2d(r, s);
		Eigen::Vector2d qp = q - p;

		if (std::fabs(rxs) < std::numeric_limits<double>::epsilon()) {
			double qpxr = cross2d(qp, r);

			if (std::fabs(qpxr) < std::numeric_limits<double>::epsilon()) {
				// collinear, so check if overlapping
				// Return midpoint of interval if overlapping, otherwise, return false
				double t0 = qp.dot(r / r.dot(r));
				double t1 = t0 + s.dot(r / r.dot(r));

				if (s.dot(r) < 0) {
					double swp = t0;
					t1 = t0;
					t0 = swp;
				}

				if ((t0 < 0.0 && t1 < 0.0) || (t0 > 1.0 && t1 > 0.0)) {
					return false;
				}

				double a = std::max(t0, 0.0);
				double b = std::min(t1, 0.0);

				assert(a <= b);

				double c = (a + b) / 2.0;

				nearest_point = p + c * r;

				return true;
			} else {
				// parallel and not colinear
				return false;
			}
		}

		double t = cross2d(qp, s) / rxs;
		double u = cross2d(qp, r) / rxs;

		if (t < 0.0 || t > 1.0 || u < 0.0 || u > 1.0) {
			return false;
		}
		
		nearest_point = p + t * r;

		assert((nearest_point - (q + u * s)).isZero(1e-7));

		return true;
	}
}