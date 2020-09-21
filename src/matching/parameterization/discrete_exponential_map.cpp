#include "discrete_exponential_map.h"

#include <set>
#include <queue>

#include <Eigen/Dense>

#include <igl/hessian_energy.h>
#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>

#include <geometry/patch.h>
#include <shape_signatures/shape_signature.h>
#include <algorithms/shortest_path.h>
#include <matching/surface_stroke.h>

using namespace shortest_path;

DiscreteExponentialMap::DiscreteExponentialMap():
	_TBN(Eigen::Matrix3d::Identity()),
	_TBN_inv(Eigen::Matrix3d::Identity()),
	_geometry(nullptr) {

}

// Unintuitively, p_vid is actually the index of the vertex in relation to patch->origin_mesh()
DiscreteExponentialMap::DiscreteExponentialMap(std::shared_ptr<Patch> patch, Eigen::DenseIndex p_vid, const Eigen::MatrixXd* guide_points) {
	// Implementation of Discrete Exponential Map from,
	// "Part-Based Representation and Editing of 3D Surface Models", Ryan Schmidt, 2011

	auto mesh = patch->origin_mesh();

	const Eigen::MatrixXd& V = mesh->vertices();
	const Eigen::MatrixXd& N_orig = mesh->vertex_normals();
	Eigen::MatrixXd N(N_orig.rows(), N_orig.cols());

	if (V.size() <= 0 || N.size() <= 0) {
		return;
	}

	std::map<Eigen::DenseIndex, std::shared_ptr<DjikstraVertexNode>> nodes = djikstras_algorithm(patch, p_vid);

	if (nodes.size() == 0) {
		// djikstra's failed??
		return;
	}

	// Smooth the normals a bit before generating the map
	Eigen::SparseMatrix<double> M2;
	Eigen::SparseMatrix<double> QH;

	Eigen::MatrixXd V3 = V.leftCols<3>();
	Eigen::MatrixXi F3 = mesh->faces().leftCols<3>();

	igl::massmatrix(V3, F3, igl::MASSMATRIX_TYPE_BARYCENTRIC, M2);
	igl::hessian_energy(V3, F3, QH);

    // Smoothing -- 0.0 is no smoothing, 1.0 is full
	const double alpha = 0.25;

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> hessSolver(alpha * QH + (1.0 - alpha) * M2);
	N << hessSolver.solve((1.0 - alpha) * M2 * N_orig);

	Eigen::Vector3d p = V.row(p_vid).block<1, 3>(0, 0).transpose();
	Eigen::Vector3d Np = N.row(p_vid).block<1, 3>(0, 0).transpose().normalized();

	Eigen::Matrix3d Tp_TBN = basis_from_plane_normal(Np);

	// For each point in patch, create a chain of points also within the patch leading back to the center by the shortest path
	std::priority_queue<std::shared_ptr<DjikstraVertexNode>, std::vector<std::shared_ptr<DjikstraVertexNode>>, DjikstraDist> Q;

	for (auto it = nodes.begin(); it != nodes.end(); ++it) {
		Q.push(it->second);
	}

	// Create a set of planar undirected edges representing the discrete exponential map
	std::map<Eigen::DenseIndex, Eigen::Vector2d> DEM_points;
	DEM_points.insert(std::make_pair(Q.top()->_vid, Eigen::Vector2d(0.0, 0.0)));
	Q.pop();	

	while (!Q.empty()) {
		auto node = Q.top();
		Q.pop();

		assert(node->_prev != nullptr);
		assert(node->_dist > 0.0);
		assert(node->_vid >= 0);

		// Run from each neighbor already present in DEM_points, as an "Upwind Average"
		// Find all of q's neighbors already in the DEM that share locality with node->_prev
		std::vector<Eigen::DenseIndex> neighbors = patch->origin_mesh()->one_ring(node->_vid);
		std::vector<std::pair<Eigen::Vector2d, double>> upwind;
		Eigen::Vector2d from_parent;
		double parent_dist = 0.0;

		for (Eigen::DenseIndex prev_vid : neighbors) {
			auto prev = DEM_points.find(prev_vid);

			if (prev == DEM_points.end()) {
				continue;
			}

			Eigen::Vector3d q = V.row(node->_vid).block<1, 3>(0, 0).transpose();

			if (prev_vid == p_vid) {
				Eigen::Vector2d Tpq = local_log_map(p, Tp_TBN, q).topRows<2>();

				double weight = (q - p).squaredNorm() + 1e-7;

				upwind.push_back(std::make_pair(Tpq, weight));

				if (prev_vid == node->_prev->_vid) {
					// Store away for special locality test when averaging upwind points
					parent_dist = Tpq.norm();
					from_parent = Tpq;
				}

				continue;
			}

			Eigen::Vector3d r = V.row(prev_vid).block<1, 3>(0, 0).transpose();
			Eigen::Vector2d Tpr = prev->second;

			// A vector parallel to two planes is the cross product of the two normals. 
			Eigen::Vector3d Nr = N.row(prev_vid).block<1, 3>(0, 0).transpose().normalized();
			Eigen::Matrix3d Tr_TBN = basis_from_plane_normal(Nr);

			// Do not pass through the intermediate planes -- instead, just transform directly into Tp after log_r(q)
			Eigen::Vector2d Trq = local_log_map(r, Tr_TBN, q).topRows<2>();

			// 3D rotation Mn
			Eigen::Vector3d rot_axis = Nr.cross(Np);

			if (rot_axis.isZero(1e-7)) {
				rot_axis = Tp_TBN.col(0);
			}

			rot_axis.normalize();

			double rot_angle_3d = std::acos(std::min(1.0, std::max(-1.0, Nr.dot(Np))));
			Eigen::AngleAxis<double> R(rot_angle_3d, rot_axis);

			// 2D rotation for planar basis alignment
			Eigen::Vector3d er = (R * Tr_TBN.col(0)).normalized();
			Eigen::Vector3d ep = Tp_TBN.col(0);

			assert(er.dot(Np) < 1e-7);

			double rot_angle_2d = std::acos(std::min(1.0, std::max(-1.0, er.dot(ep))));

			if (!er.cross(ep).isZero(1e-7) && er.cross(ep).normalized().dot(Np) > 0.0) {
				rot_angle_2d *= -1.0;
			}

			Eigen::Rotation2D<double> E(rot_angle_2d);

			Eigen::Vector2d Tpq = Tpr + E * Trq;

			double weight = 1.0 / ((q - r).squaredNorm() + 1e-7);
			
			if (prev_vid == node->_prev->_vid) {
				// Store away for special locality test when averaging upwind points
				parent_dist = (Tpr - Tpq).norm();
				from_parent = Tpq;
			}

			upwind.push_back(std::make_pair(Tpq, weight));
		}

		assert(upwind.size() > 0);

		Eigen::Vector2d Tpq_avg(0.0, 0.0);

		double total_weight = 0.0;
		for (auto pt : upwind) {
			if ((pt.first - from_parent).norm() > parent_dist / 2.0) {
				continue;
			}

			total_weight += pt.second;
		}

		for (auto pt : upwind) {
			if ((pt.first - from_parent).norm() > parent_dist / 2.0) {
				continue;
			}

			Tpq_avg += pt.second * pt.first / total_weight;
		}

		DEM_points.insert(std::pair<Eigen::DenseIndex, Eigen::Vector2d>(node->_vid, Tpq_avg));
	}

	if (!init(p_vid, Tp_TBN, DEM_points, patch)) {
		throw std::invalid_argument("DEM arguments invalid!");
	}
}

DiscreteExponentialMap::DiscreteExponentialMap(const Eigen::DenseIndex center_vid, Eigen::Matrix3d& TBN, const std::map<Eigen::DenseIndex, Eigen::Vector2d>& vertices, std::shared_ptr<Patch> geometry) {
	if (!init(center_vid, TBN, vertices, geometry)) {
		throw std::invalid_argument("DEM arguments invalid!");
	}
}

DiscreteExponentialMap::~DiscreteExponentialMap() {

}

Eigen::MatrixXi DiscreteExponentialMap::get_reindexed_faces() const {
	Eigen::MatrixXi F = _faces;

	std::map<Eigen::DenseIndex, Eigen::DenseIndex> vid_remap;
	Eigen::DenseIndex i = 0;
	for (auto it = _vertices.cbegin(); it != _vertices.cend(); ++it, ++i) {
		vid_remap.insert(std::make_pair(it->first, i));
	}

	for (i = 0; i < F.size(); ++i) {
		F(i) = vid_remap.at(F(i));
	}

	return F;
}

bool DiscreteExponentialMap::init(const Eigen::DenseIndex center_vid, Eigen::Matrix3d& TBN, const std::map<Eigen::DenseIndex, Eigen::Vector2d>& vertices, std::shared_ptr<Patch> geometry) {
	_geometry = geometry;
	_TBN = TBN;
	_TBN_inv = TBN.inverse();
	_vertices = vertices;

	_center_vid = center_vid;

	// Only include faces made up of vertices in the map
	std::set<Eigen::DenseIndex> fids;
	std::shared_ptr<Mesh> mesh = _geometry->origin_mesh();
	const Eigen::MatrixXi& F = mesh->faces();
	const Eigen::MatrixXd& V = mesh->vertices();

	for (Eigen::DenseIndex i = 0; i < F.rows(); ++i) {
		bool included = true;

		for (Eigen::DenseIndex j = 0; j < F.cols(); ++j) {
			if (vertices.find(F(i, j)) == vertices.end()) {
				included = false;
				break;
			}
		}

		if (!included) {
			continue;
		}

		for (Eigen::DenseIndex j = 0; j < F.cols(); ++j) {
			// Check edge length (it should not be severely distorted
			Eigen::DenseIndex next = (j + 1) % F.cols();

			double dist = (V.row(F(i, j)).leftCols<3>() - V.row(F(i, next)).leftCols<3>()).norm();
			double dem_dist = (vertices.at(F(i, j)) - vertices.at(F(i, next))).norm();
			
			if (dem_dist > 2.0 * dist) {
				included = false;
				break;
			}
		}

		if (included) {
			fids.insert(i);
		}
	}

	_faces = Eigen::MatrixXi(fids.size(), F.cols());

	int fIndex = 0;
	for (auto fid : fids) {
		_fid_remap.insert(std::make_pair(fIndex, fid));
		_faces.row(fIndex++) << F.row(fid);
	}

	std::stringstream ss; ss << geometry->origin_mesh()->resource_dir() << "//matlab//dem_debug.m";
	to_matlab(ss.str());

	std::vector<std::pair<Eigen::DenseIndex, Eigen::Vector2d>> face_centers;

	_center_fid = -1;

	std::vector<Eigen::DenseIndex> center_fids;
	for (Eigen::DenseIndex i = 0; i < _faces.rows(); ++i) {
		for (Eigen::DenseIndex j = 0; j < _faces.cols(); ++j) {
			if (_faces(i, j) == _center_vid) {
				center_fids.push_back(i);
				break;
			}
		}
	}

	for (auto fid : center_fids) {
		Eigen::Vector2d fc = Eigen::Vector2d::Zero();

		for (Eigen::DenseIndex j = 0; j < _faces.cols(); ++j) {
			fc += vertices.at(_faces(fid, j)) / static_cast<double>(_faces.cols());
		}

		face_centers.push_back(std::pair<Eigen::DenseIndex, Eigen::Vector2d>(fid, fc));
	}

	Eigen::DenseIndex centroid_fid = -1;
	double dist = std::numeric_limits<double>::max();

	for (Eigen::DenseIndex i = 0; i < face_centers.size(); ++i) {
		double c_dist = (face_centers[i].second - vertices.at(_center_vid)).norm();

		if (c_dist < dist) {
			dist = c_dist;
			_center_fid = face_centers[i].first;
		}
	}

	if (_center_fid < 0 && _vertices.size() > 2) {
		throw std::domain_error("Invalid _center_fid!");
	}

	return true;
}

Eigen::DenseIndex DiscreteExponentialMap::get_center_vid() const {
	return _center_vid;
}

Eigen::DenseIndex DiscreteExponentialMap::get_center_fid() const {
	return _center_fid;
}

Eigen::MatrixXd DiscreteExponentialMap::get_3d_vertices() const {
	// get_reindexed_faces describes the faces for this vertex ordering
	Eigen::MatrixXd V_3d(_vertices.size(), 3);

	Eigen::DenseIndex index = 0;
	for (auto it = _vertices.cbegin(); it != _vertices.cend(); ++it) {
		Eigen::Vector3d tbn_point = (Eigen::Vector3d() << it->second, 0.0).finished();

		V_3d.row(index++) = _TBN * tbn_point;
	}

	return V_3d;
}

double DiscreteExponentialMap::get_radius() const {
	Eigen::MatrixXd V = get_3d_vertices();

	// the DEM is relative to the frame origin, so the radius is just the greatest magnitude norm
	double radius = 0.0;
	for (Eigen::DenseIndex i = 0; i < V.rows(); ++i) {
		radius = std::max(V.row(i).norm(), radius);
	}

	return radius;
}

Eigen::Vector3d DiscreteExponentialMap::get_normal() const {
	return _TBN.col(2);
}

Eigen::Vector3d DiscreteExponentialMap::get_tangent() const {
	return _TBN.col(0);
}

Eigen::Vector3d DiscreteExponentialMap::get_bitangent() const {
	return _TBN.col(1);
}

Eigen::Vector2d DiscreteExponentialMap::interpolated_polar(Eigen::Vector3d barycentric_coords, const std::vector<Eigen::DenseIndex>& vids) {
	Eigen::Vector2d polar;
	Eigen::MatrixXd points(2,3);

	for (Eigen::DenseIndex i = 0; i < 3; i++) {
		// Just gonna let it throw an exception if the vids are no in the map -- shame on the user!
		points.col(i) = _vertices[vids[i]];
	}

	Eigen::Vector2d xy = points * barycentric_coords;

	polar << std::sqrt(std::pow(xy(0), 2) + std::pow(xy(1), 2)), std::atan2(xy(1), xy(0));

	return polar;
}

Eigen::DenseIndex DiscreteExponentialMap::nearest_vertex_by_polar(const Eigen::Vector2d& polar_point) {
	Eigen::Vector2d xy_point;
	xy_point << polar_point(0) * std::cos(polar_point(1)),
				polar_point(0) * std::sin(polar_point(1));

	Eigen::DenseIndex vid = -1;
	double dist = std::numeric_limits<double>::max();
	for (auto v : _vertices) {
		double t_dist = (v.second - xy_point).norm();

		if (t_dist < dist) {
			vid = v.first;
			dist = t_dist;
		}
	}

	return vid;
}

Eigen::VectorXd DiscreteExponentialMap::query_map_value(const Eigen::Vector2d& xy_point, std::shared_ptr<ShapeSignature> sig) const {
	// Find triangle which contains (x, y)
	unsigned int i = 0;
	for (i = 0; i < _faces.rows(); ++i) {
		Eigen::DenseIndex r = _faces(i, 0);
		Eigen::DenseIndex s = _faces(i, 1);
		Eigen::DenseIndex t = _faces(i, 2);

		Eigen::Vector2d a = _vertices.at(_faces(i, 0));
		Eigen::Vector2d b = _vertices.at(_faces(i, 1));
		Eigen::Vector2d c = _vertices.at(_faces(i, 2));

		if (point_in_triangle(xy_point, a, b, c)) {
			break;
		}
	}

	// In case point isn't in any triangle, return some invalid result (-1.0?)
	if (i >= _faces.rows()) {
		// point is not within any triangle of the map, so return a vector packed with -1.0s
		return Eigen::VectorXd::Constant(sig->feature_dimension(), -1.0);
	}

	// Find barycentric coordinates of the point within the triangle
	// https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
	Eigen::Vector2d v0 = _vertices.at(_faces(i, 1)) - _vertices.at(_faces(i, 0));
	Eigen::Vector2d v1 = _vertices.at(_faces(i, 2)) - _vertices.at(_faces(i, 0));
	Eigen::Vector2d v2 = xy_point - _vertices.at(_faces(i, 0));
	double d00 = v0.dot(v0);
	double d01 = v0.dot(v1);
	double d11 = v1.dot(v1);
	double d20 = v2.dot(v0);
	double d21 = v2.dot(v1);
	double denom = d00 * d11 - d01 * d01;
	double v = (d11 * d20 - d01 * d21) / denom;
	double w = (d00 * d21 - d01 * d20) / denom;
	double u = 1.0f - v - w;

	if (u + v + w - 1.0 > std::numeric_limits<double>::epsilon()) {
		throw std::domain_error("query_map_value(): Invalid barycentric coordinates!");
	}

	// return linearly interpolated feature values from triangle vertices with respect to (x,y)
	//Eigen::VectorXd value = u * features.row(_faces(i, 0)) + v * features.row(_faces(i, 1)) + w * features.row(_faces(i, 2));
	Eigen::VectorXd u_coord = sig->lerpable_coord(_fid_remap.at(i), _faces(i, 0));
	Eigen::VectorXd v_coord = sig->lerpable_coord(_fid_remap.at(i), _faces(i, 1));
	Eigen::VectorXd w_coord = sig->lerpable_coord(_fid_remap.at(i), _faces(i, 2));

	Eigen::VectorXd value = u * u_coord
						  + v * v_coord
						  + w * w_coord;

	value = sig->lerpable_to_signature_value(value);

	return value;
}

Eigen::VectorXd DiscreteExponentialMap::query_map_value_polar(const Eigen::Vector2d& polar_point, std::shared_ptr<ShapeSignature> sig) const {
	Eigen::Vector2d xy_point;
	xy_point << polar_point(0) * std::cos(polar_point(1)),
				polar_point(0) * std::sin(polar_point(1));

	return query_map_value(xy_point, sig);
}

bool DiscreteExponentialMap::to_matlab(std::string script_out_path) {
	std::ofstream m(script_out_path, std::ofstream::out);

	if (m.is_open()) {
		m << "figure;" << std::endl;
		m << "hold on;" << std::endl;
		m << "axis equal;" << std::endl;
		m << "grid on;" << std::endl;

		auto vertices = get_raw_vertices();
		m << "v = [ ..." << std::endl;
		for (auto vert : vertices) {
			m << vert.second.transpose() << "; ..." << std::endl;
		}
		m << "];" << std::endl;

		const Eigen::MatrixXi& faces = get_reindexed_faces();
		m << "f = [ ..." << std::endl;
		for (Eigen::DenseIndex i = 0; i < faces.rows(); ++i) {
			m << faces.row(i) << "; ..." << std::endl;
		}
		m << "];" << std::endl;

		m << "for i=1:size(f,1)" << std::endl;
		m << "a = [v(f(i, 1) + 1, :), 0.0];" << std::endl;
		m << "b = [v(f(i, 2) + 1, :), 0.0];" << std::endl;
		m << "c = [v(f(i, 3) + 1, :), 0.0];" << std::endl;
		m << "cb = (c - b) / norm(c - b);" << std::endl;
		m << "ab = (a - b) / norm(a - b);" << std::endl;
		m << "n = cross(cb, ab);" << std::endl;
		m << "n = n / norm(n);" << std::endl;
		m << "C = 'g';" << std::endl;
		m << "if dot(n, [0, 0, 1]) < 1 - 1e-7" << std::endl;
		m << "	C = 'r';" << std::endl;
		m << "end" << std::endl;
		m << "h = fill([a(1), b(1), c(1)], [a(2), b(2), c(2)], C);" << std::endl;
		m << "set(h, 'facealpha', .5);" << std::endl;
		m << "end" << std::endl;

		m << "scatter(v(:,1), v(:,2), 'mo');" << std::endl;

		m.close();
	}
	else {
		return false;
	}

	return true;
}