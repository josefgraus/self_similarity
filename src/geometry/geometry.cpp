#include "geometry.h"

#include <cmath>

#include <halfedge/trimesh.cpp>

Geometry::Geometry() {

}

Geometry::~Geometry() {

}

Eigen::Matrix3d basis_from_plane_normal(const Eigen::Vector3d plane_normal) {
	// Set up basis
	Eigen::Vector3d normal = plane_normal.normalized();

	Eigen::Vector3d tangent = normal.cross(Eigen::Vector3d::UnitY()).normalized();

	if (tangent.isZero() || std::isnan(tangent(0)) || std::isnan(tangent(1)) || std::isnan(tangent(2))) {
		// Just in case we're unlucky and plane_normal is (anti)parallel to UnitX(), then it can't also be (anti)parellel UnitY()
		tangent = normal.cross(Eigen::Vector3d::UnitX()).normalized();
	}

	Eigen::Vector3d bitangent = normal.cross(tangent).normalized();

	Eigen::Matrix3d TBN;
	TBN << tangent, bitangent, normal;

	assert(tangent.dot(bitangent) < std::numeric_limits<double>::epsilon());
	assert(tangent.dot(normal) < std::numeric_limits<double>::epsilon());
	assert(bitangent.dot(normal) < std::numeric_limits<double>::epsilon());

	return TBN;
}

// Quick point-in-triangle test taken from https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
double sign(Eigen::Vector2d p1, Eigen::Vector2d p2, Eigen::Vector2d p3) {
	return (p1(0) - p3(0)) * (p2(1) - p3(1)) - (p2(0) - p3(0)) * (p1(1) - p3(1));
}

bool point_in_triangle(Eigen::Vector2d pt, Eigen::Vector2d v1, Eigen::Vector2d v2, Eigen::Vector2d v3) {
	bool b1, b2, b3;

	b1 = sign(pt, v1, v2) < 0.0f;
	b2 = sign(pt, v2, v3) < 0.0f;
	b3 = sign(pt, v3, v1) < 0.0f;

	return ((b1 == b2) && (b2 == b3));
}

/* origin: Standard basis coordinate which is the origin of the local tangent plane which point is being mapped into
// origin_TBN: surface at origin
// point: Standard basis coordinate which is the point being rotated into origin's tangent plane
// output -> coordinate of mapping with in standard basis
*/
Eigen::Vector3d local_log_map(const Eigen::Vector3d& origin, const Eigen::Matrix3d& origin_TBN, const Eigen::Vector3d& point) {
	// Project the edge between the origin and the point onto the plane defined by the TBN matrix
	// Per paper
	Eigen::Vector3d qr = (point - origin);
	Eigen::Vector3d qr_planar;
	qr_planar << origin_TBN.col(0).dot(qr), origin_TBN.col(1).dot(qr), 0.0; 

	qr_planar.normalize();

	return qr.norm() * qr_planar;
}