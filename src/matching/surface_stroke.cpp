#include "surface_stroke.h"

#include <set>

#include <CatmullRom.h>
#include <BSpline.h>
#include <Bezier.h>

#include <geometry/patch.h>
#include <matching/parameterization/discrete_exponential_map.h>
#include <algorithm>
#include <matching/parameterization/curve_unrolling.h>
#include <geometry/tests/intersection.h>

double binomialCoeff(int n, int k) {
	int res = 1;

	// Since C(n, k) = C(n, n-k)  
	if (k > n - k)
		k = n - k;

	// Calculate value of  
	// [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]  
	for (int i = 0; i < k; ++i) {
		res *= (n - i);
		res /= (i + 1);
	}

	return static_cast<double>(res);
}


CubicBezierSegment::CubicBezierSegment(const Eigen::MatrixXd& pts): _pts(pts), _piecewise_length(0.0) {
	for (Eigen::DenseIndex i = 0; i < pts.cols() - 1; ++i) {
		_piecewise_length += (pts.col(i + 1) - pts.col(i)).norm();
	}
}

Eigen::Vector3d CubicBezierSegment::pt_at_t(double t) const {
	t = std::min(std::max(0.0, t), 1.0);

	Eigen::Vector3d P = Eigen::Vector3d::Zero();

	int n = static_cast<int>(_pts.cols() - 1);
	for (int i = 0; i < n; ++i) {
		P += binomialCoeff(n - 1, i) * std::pow(1.0 - t, (n - 1) - i) * std::pow(t, i) * _pts.col(i);
	}

	return P;
}

Eigen::Vector3d CubicBezierSegment::pt_at_dist(double dist) const {
	double t = dist / _piecewise_length;

	return pt_at_t(t);
}

struct SurfaceStrokeInstancer : public SurfaceStroke, std::enable_shared_from_this<SurfaceStroke> {
	public:
		SurfaceStrokeInstancer(std::shared_ptr<Mesh> mesh) : SurfaceStroke(mesh) { }
		SurfaceStrokeInstancer(std::shared_ptr<Mesh> mesh, const SurfaceStroke& other, Eigen::DenseIndex centroid_vid) : SurfaceStroke(mesh, other, centroid_vid) { }
};

std::shared_ptr<SurfaceStroke> SurfaceStroke::instantiate(std::shared_ptr<Mesh> mesh) {
	return std::make_shared<SurfaceStrokeInstancer>(mesh);
}

std::shared_ptr<SurfaceStroke> SurfaceStroke::instantiate(std::shared_ptr<Mesh> mesh, const SurfaceStroke& other, Eigen::DenseIndex centroid_vid) {
	return std::make_shared<SurfaceStrokeInstancer>(mesh, other, centroid_vid);
}

SurfaceStroke::SurfaceStroke(std::shared_ptr<Mesh> mesh): _mesh(mesh) {
} 

SurfaceStroke::SurfaceStroke(std::shared_ptr<Mesh> mesh, const SurfaceStroke& other, Eigen::DenseIndex centroid_vid): _mesh(mesh) {
	// Create copy of stroke at centroid vid at an arbitrary orientation
	std::shared_ptr<Patch> other_cover;
	std::shared_ptr<DiscreteExponentialMap> other_dem;
	Eigen::MatrixXd map_points = other.parameterized_space_points_2d(&other_cover, &other_dem);

	// Use largest radius of either DEM or patch for safety
	double extent;
	other.curve_center(&extent);
	extent *= 2.0;

	std::shared_ptr<Patch> cover = Patch::instantiate(mesh, centroid_vid, extent);
	//Eigen::VectorXi fids = Eigen::VectorXi::LinSpaced(_mesh->faces().rows(), 0, _mesh->faces().rows() - 1);
	//std::shared_ptr<Patch> cover = Patch::instantiate(_mesh, fids);
	std::shared_ptr<DiscreteExponentialMap> dem = std::make_shared<DiscreteExponentialMap>(cover, centroid_vid, &map_points);

	// For each point in map_points, find the triangle fid in DEM that is falls within, then calculate its barycentric coordinates, then add_curve_point(fid, bc)
	const Eigen::MatrixXi& dem_faces = mesh->faces();
	const std::map<Eigen::DenseIndex, Eigen::Vector2d>& dem_verts = dem->get_raw_vertices();

	for (Eigen::DenseIndex i = 0; i < map_points.cols(); ++i) {
		Eigen::DenseIndex j = 0;

		for (j = 0; j < dem_faces.rows(); ++j) {
			Eigen::DenseIndex r = dem_faces(j, 0);
			Eigen::DenseIndex s = dem_faces(j, 1);
			Eigen::DenseIndex t = dem_faces(j, 2);

			if (dem_verts.count(r) <= 0 ||
				dem_verts.count(s) <= 0 || 
				dem_verts.count(t) <= 0) {
				continue;
			}
			 
			Eigen::Vector2d a = dem_verts.at(r);
			Eigen::Vector2d b = dem_verts.at(s);
			Eigen::Vector2d c = dem_verts.at(t);

			if (point_in_triangle(map_points.col(i), a, b, c)) {
				break;
			}
		}

		if (j >= dem_faces.rows()) {
			throw std::logic_error("Every point should be in at least one triangle (possibly more as DEM is not bijective)!");
		}

		// Find barycentric coordinates of the point within the triangle
		// https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
		// TODO: Consider moving to utility/geometry class since this is now used in multiple locations
		//		 A submission deadline is the reason for the copy-paste laziness you see before you
		Eigen::DenseIndex r = dem_faces(j, 0);
		Eigen::DenseIndex s = dem_faces(j, 1);
		Eigen::DenseIndex t = dem_faces(j, 2);
		Eigen::Vector2d a = dem_verts.at(r);
		Eigen::Vector2d b = dem_verts.at(s);
		Eigen::Vector2d c = dem_verts.at(t);

		Eigen::Vector2d v0 = b - a;
		Eigen::Vector2d v1 = c - a;
		Eigen::Vector2d v2 = map_points.col(i) - a;
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

		add_curve_point(j, (Eigen::Vector3d() << u, v, w).finished());
	}
}

std::shared_ptr<SurfaceStroke> SurfaceStroke::clone() {
	std::shared_ptr<SurfaceStroke> copy = SurfaceStroke::instantiate(_mesh);

	copy->_curve_points = _curve_points;
	copy->_fids = _fids;

	return copy;
}


void SurfaceStroke::add_curve_point(Eigen::DenseIndex fid, Eigen::Vector3d bc) {
	_curve_points.push_back(BarycentricCoord(fid, bc));
	_fids.insert(fid);
}

void SurfaceStroke::transform(double x, double y, double radians, int blade_point_index) {
	// Create copy of stroke at centroid vid at an arbitrary orientation
	std::shared_ptr<Patch> cpatch;
	std::shared_ptr<DiscreteExponentialMap> cdem;
	Eigen::MatrixXd map_points = parameterized_space_points_2d(&cpatch, &cdem);

	Eigen::VectorXd offset;
	if (blade_point_index < 0) {
		std::size_t curve_center_index;
		curve_center(nullptr, &curve_center_index);
		offset = map_points.col(curve_center_index);
	} else {
		offset = map_points.col(blade_point_index);
	}

	Eigen::Rotation2D<double> R(radians);

	Eigen::Vector2d T; 
	T << x, y;

	map_points = (R.toRotationMatrix() * map_points).colwise() + T;
}

double SurfaceStroke::compare(const SurfaceStroke& other, std::shared_ptr<ShapeSignature> sig) {
	if (blade_points().size() != other.blade_points().size()) {
		throw std::invalid_argument("Surface strokes must be the same size in sampled points!");
	}

	assert(sig->sig_steps().size() == 1);

	Eigen::VectorXd comp(_curve_points.size());
	for (std::size_t i = 0; i < _curve_points.size(); ++i) {
		double O = _curve_points[i].to_sig_value(_mesh, sig->get_signature_values(sig->sig_steps()(0)))(0);
		double C = other.blade_points()[i].to_sig_value(_mesh, sig->get_signature_values(sig->sig_steps()(0)))(0);

		comp(i) = std::pow(O - C, 2.0);
	}

	return comp.sum();
}

// this minus other
Eigen::VectorXd SurfaceStroke::per_point_diff(const SurfaceStroke& other, std::shared_ptr<ShapeSignature> sig) {
	if (blade_points().size() != other.blade_points().size()) {
		throw std::invalid_argument("Surface strokes must be the same size in sampled points!");
	}

	assert(sig->sig_steps().size() == 1);

	Eigen::VectorXd comp(_curve_points.size());
	for (std::size_t i = 0; i < _curve_points.size(); ++i) {
		double O = _curve_points[i].to_sig_value(_mesh, sig->get_signature_values(sig->sig_steps()(0)))(0);
		double C = other.blade_points()[i].to_sig_value(_mesh, sig->get_signature_values(sig->sig_steps()(0)))(0);

		comp(i) = O - C;
	}

	return comp;
}

bool SurfaceStroke::is_disjoint_from(std::shared_ptr<SurfaceStroke> other) const {
	std::vector<Eigen::DenseIndex> intersection;

	if (_mesh != other->_mesh) {
		return true;
	}

	std::set_intersection(_fids.begin(), _fids.end(), other->_fids.begin(), other->_fids.end(), std::back_inserter(intersection));

	return intersection.size() <= 0;
}

Eigen::MatrixXd SurfaceStroke::to_world() const {
	Eigen::MatrixXd world_pts(3, _curve_points.size());

	for (std::size_t i = 0; i < _curve_points.size(); ++i) {
		world_pts.col(i) = _curve_points[i].to_world(_mesh);
	}	

	return world_pts;
}

void SurfaceStroke::display(igl::opengl::glfw::Viewer& viewer, bool draw_points, std::shared_ptr<Geometry> geo, bool clear, Eigen::Vector3d offset, Eigen::Vector3d color, std::shared_ptr<SurfaceStroke> this_shared) {
	// TODO: Not the best -- assumes viewer points and lines containers only contain this curve...
	//	     Need a massive rendering reorganization to bring all the unplanned new components together
	if (geo == nullptr) {
		geo = _mesh;
	}

	if (draw_points) {
		//viewer.data().add_points(pts, colors);
		auto P = to_world();

		if (P.size() <= 0) {
			return;
		}

		const double EUCLIDEAN_SCALE = 5.0;

		std::shared_ptr<Curve> curve = std::make_shared<BSpline>();

		double _piecewise_length = 0.0;
		for (Eigen::DenseIndex i = 0; i < P.cols() - 1; ++i) {
			_piecewise_length += (P.col(i + 1) - P.col(i)).norm();
		}

		int num_pts = static_cast<int>(std::ceil(_piecewise_length / EUCLIDEAN_SCALE));

		curve->set_steps(num_pts);

		for (int i = 0; i < P.cols(); ++i) {
			curve->add_way_point(Vector(P(0, i), P(1, i), P(2, i)));
		}

		Eigen::MatrixXd bpts(curve->node_count(), 3);
		int ptIndex = 0;
		if (curve->node_count() > 0) {
			bpts.row(ptIndex++) << curve->node(0).x, curve->node(0).y, curve->node(0).z;
		}

		for (int i = 1; i < curve->node_count(); ++i) {
			Eigen::Vector3d next_pt; next_pt << curve->node(i).x, curve->node(i).y, curve->node(i).z;
	
			if ((next_pt.transpose() - bpts.row(ptIndex - 1)).isZero()) {
				continue;
			}

			bpts.row(ptIndex++) = next_pt.transpose();
		}
		bpts.conservativeResize(ptIndex, bpts.cols());

		bool meshed_curve = true;

		if (!meshed_curve || bpts.rows() < 3) {
			// Just draw points	
			if (clear) {
				viewer.data().points.resize(0, 0);
				viewer.data().lines.resize(0, 0);
			}

			color = { 1.0, 0.0, 1.0 };
			Eigen::MatrixXd bcolors = color.transpose().replicate(_curve_points.size(), 1); // default is all blue points
			viewer.data().add_points(bpts, bcolors);
		} else {
			if (clear) {
				_mesh->deselect_all(viewer);
				//_mesh->display(viewer);
			}

			// Build a tube mesh with points
			const int TUBE_SIDES = 16;
			const double THICKNESS = 1e-2;
			const int N = bpts.rows();
			const int VCOUNT = (TUBE_SIDES * (N - 2)) + 2;
			const int FCOUNT = (TUBE_SIDES * 2 * (N - 3)) + (2 * TUBE_SIDES);
			const double ROT_STEP = (2.0 * M_PI) / TUBE_SIDES;

			Eigen::MatrixXd V(VCOUNT, 3);
			Eigen::MatrixXi F(FCOUNT, 3);

			Eigen::DenseIndex vIndex = 0;
			Eigen::DenseIndex fIndex = 0;

			// beginning cap
			V.row(vIndex) = bpts.row(vIndex);
			vIndex++;
			{
				Eigen::Vector3d rot_axis = (bpts.row(1) - bpts.row(0)).transpose().normalized();
				Eigen::Vector3d T = THICKNESS * basis_from_plane_normal(rot_axis).col(0);

				for (int i = 0; i < TUBE_SIDES; ++i) {
					Eigen::AngleAxis<double> R(static_cast<double>(i) *  ROT_STEP, rot_axis);

					auto RT = R * T;

					V.row(vIndex++) = (R * T).transpose() + bpts.row(1);

					if (i + 1 == TUBE_SIDES) {
						F.row(fIndex++) << 0, 1, i + 1;
					} else {
						F.row(fIndex++) << 0, i + 2, i + 1;
					}
				}
			}

			// middle segments
			for (int i = 2; i < bpts.rows() - 1; ++i) {
				Eigen::DenseIndex mIndex = vIndex - TUBE_SIDES;

				Eigen::Vector3d rot_axis = (bpts.row(i) - bpts.row(i - 1)).transpose().normalized();
				Eigen::Vector3d T = THICKNESS * basis_from_plane_normal(rot_axis).col(0);

				for (int j = 0; j < TUBE_SIDES; ++j) {
					Eigen::AngleAxis<double> R(static_cast<double>(j) *  ROT_STEP, rot_axis);

					auto RT = R * T;

					V.row(vIndex++) = (R * T).transpose() + bpts.row(i);

					int K = vIndex - 1;

					if (j + 1 == TUBE_SIDES) {
						F.row(fIndex++) << K - TUBE_SIDES, K + 1 - TUBE_SIDES, K;
						F.row(fIndex++) << K + 1 - TUBE_SIDES, K - TUBE_SIDES, K - 2 * TUBE_SIDES + 1;
					} else {			   
						F.row(fIndex++) << K - TUBE_SIDES, K + 1, K;
						F.row(fIndex++) << K + 1, K - TUBE_SIDES, K - TUBE_SIDES + 1;
					}
				}
			}

			// end cap
			Eigen::DenseIndex endIndex = vIndex;
			V.row(vIndex++) = bpts.row(bpts.rows() - 1);
			
			for (int i = 0; i < TUBE_SIDES; ++i) {
				if (i + 1 == TUBE_SIDES) {
					F.row(fIndex++) << endIndex, (endIndex - TUBE_SIDES) + i, endIndex - TUBE_SIDES;
				}
				else {
					F.row(fIndex++) << endIndex, (endIndex - TUBE_SIDES) + i, (endIndex - TUBE_SIDES) + i + 1;
				}
			}
			
			// Add V and F to active mesh
			_mesh->add_geometry(viewer, V, F, this_shared);
		}
	}
}

BarycentricCoord SurfaceStroke::curve_center(double* max_dist_from_center, std::size_t* curve_cindex) const {
	double curve_len = 0.0;

	auto V = _mesh->vertices();
	auto F = _mesh->faces();

	// Find curve length
	std::vector<double> curve_pt_dists;
	curve_pt_dists.push_back(0.0);

	Eigen::Vector3d prev_cp = _curve_points.begin()->to_world(_mesh).topRows<3>();
	for (auto it = _curve_points.begin() + 1; it != _curve_points.end(); ++it) {
		Eigen::Vector3d bc_world = it->to_world(_mesh).topRows<3>();
		curve_len += (bc_world.topRows<3>() - prev_cp).norm();
		curve_pt_dists.push_back(curve_len);
		prev_cp = bc_world;
	}

	/*
	double center_len = curve_len / 2.0;

	std::size_t curve_center_index = -1;
	double neg_proxy = curve_pt_dists[0] - center_len;
	for (int i = 1; i < curve_pt_dists.size(); ++i) {
		double step_dist = curve_pt_dists[i] - center_len;

		if (step_dist < 0.0) {
			neg_proxy = step_dist;
			continue;
		}

		if (std::fabs(neg_proxy) < step_dist) {
			curve_center_index = i - 1;
		} else {
			curve_center_index = i;
		}

		break;
	} */

	//BarycentricCoord center_bc = _curve_points[curve_center_index];
	std::size_t curve_center_index = _curve_points.size() / 2;
	BarycentricCoord center_bc = _curve_points[curve_center_index];

	// Find furthest distance from center_bc
	if (max_dist_from_center != nullptr) {
		*max_dist_from_center = std::max(std::fabs(curve_pt_dists[curve_center_index]), std::fabs(curve_len - curve_pt_dists[curve_center_index]));
	}

	if (curve_cindex != nullptr) {
		*curve_cindex = curve_center_index;
	}

	return center_bc;
}

std::shared_ptr<Patch> SurfaceStroke::cover_patch() const {
	double radius = 0.0;
	BarycentricCoord center = curve_center(&radius);
	Eigen::DenseIndex cover_center_vid = _mesh->closest_vertex_id(center._fid, center._coeff.topRows<3>().cast<float>());

	// Find adequate cover patch for curve
	// From curve center, find radius, and construct patch from there
	//std::shared_ptr<Patch> cover_patch = Patch::instantiate(_mesh, cover_center_vid, radius);
	const Eigen::MatrixXi& F = origin_mesh()->faces();

	assert(F.cols() == 3);

	std::set<Eigen::DenseIndex> fids;
	std::set<Eigen::DenseIndex> vids;

	for (auto fid : _fids) {
		for (Eigen::DenseIndex i = 0; i < F.cols(); ++i) {
			vids.insert(F(fid, i));
		}
	}

	// Find faces that contain vertices entirely included in the first-pass cover, but don't actually contain any curve points
	for (Eigen::DenseIndex i = 0; i < F.rows(); ++i) {
		int vcount = 0;

		for (Eigen::DenseIndex j = 0; j < F.cols(); ++j) {
			if (vids.count(F(i, j)) > 0) {
				vcount++;
			}
		}

		if (vcount == 3) {
			fids.insert(i);
		}
	}

	Eigen::VectorXi manifold_fids(fids.size());

	Eigen::DenseIndex index = 0;
	for (auto fid : fids) {
		manifold_fids(index++) = fid;
	}

	std::shared_ptr<Patch> cover_patch = Patch::instantiate(_mesh, manifold_fids);
	
	return cover_patch;
}

std::shared_ptr<DiscreteExponentialMap> SurfaceStroke::cover_patch_local_dem(std::shared_ptr<Patch> cover_patch) {
	std::shared_ptr<DiscreteExponentialMap> dem = std::make_shared<DiscreteExponentialMap>(cover_patch, cover_patch->get_centroid_vid_on_origin_mesh());

	return dem;
}

GeodesicFanBlade::SignatureTensor SurfaceStroke::blade_values(double angle_step, std::shared_ptr<ShapeSignature> sig, std::shared_ptr<DiscreteExponentialMap>* origin_dem) {
	Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic> fan;

	if (sig->feature_count() <= 0) {
		return fan;
	}
	
	// Find adequate cover patch for curve
	// From curve center, find radius, and construct patch from there
	std::shared_ptr<Patch> cpatch;
	std::shared_ptr<DiscreteExponentialMap> dem;
	Eigen::MatrixXd dem_curve = parameterized_space_points_2d(&cpatch, &dem);

	Eigen::DenseIndex rows = static_cast<unsigned int>(blade_points().size());
	Eigen::DenseIndex cols = static_cast<Eigen::DenseIndex>(std::floor(2.0 * M_PI / angle_step));
	
	fan = Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic>(rows, cols);

	// Sample curve at each angle step in [0, 2 * Pi]
	for (unsigned int i = 0; i < cols; ++i) {
		Eigen::Rotation2D<double> R(static_cast<double>(i) * angle_step);
		Eigen::MatrixXd rot_curve = R.toRotationMatrix() * dem_curve;

		for (unsigned int j = 0; j < rows; ++j) {
			fan(j,i) = dem->query_map_value(rot_curve.col(j), sig);
		}
	}

	if (origin_dem != nullptr) {
		*origin_dem = dem;
	}

	return fan;
}

bool SurfaceStroke::to_matlab(std::string matlab_file_path) const {
	std::shared_ptr<DiscreteExponentialMap> dem;
	Eigen::MatrixXd dem_curve = parameterized_space_points_2d(nullptr, &dem);

	return matlab_template(matlab_file_path, dem, dem_curve);
}

bool SurfaceStroke::matlab_template(std::string matlab_file_path, std::shared_ptr<DiscreteExponentialMap> dem, Eigen::MatrixXd curve) const {
	std::ofstream m(matlab_file_path, std::ofstream::out);

	if (m.is_open()) {
		m << "figure;" << std::endl;
		m << "hold on;" << std::endl;
		m << "axis equal;" << std::endl;
		m << "grid on;" << std::endl;

		auto vertices = dem->get_raw_vertices();
		m << "v = [ ..." << std::endl;
		for (auto vert : vertices) {
			m << vert.second.transpose() << "; ..." << std::endl;
		}
		m << "];" << std::endl;

		m << "s = [ ..." << std::endl;
		for (Eigen::DenseIndex i = 0; i < curve.cols(); ++i) {
			m << curve.col(i).transpose() << "; ..." << std::endl;
		}
		m << "];" << std::endl;

		const Eigen::MatrixXi& faces = dem->get_reindexed_faces();
		m << "f = [ ..." << std::endl;
		for (Eigen::DenseIndex i = 0; i < faces.rows(); ++i) {
			m << faces.row(i) << "; ..." << std::endl;
		}
		m << "];" << std::endl;

		m << "scatter(s(:,1), s(:,2), 'm.');" << std::endl;
		m << "scatter(v(:,1), v(:,2), 'go');" << std::endl;

		m << "for i=1:size(f,1)" << std::endl;
		m << "plot([v(f(i,1)+1,1), v(f(i,2)+1,1), v(f(i,3)+1,1), v(f(i,1)+1,1)], [v(f(i,1)+1,2), v(f(i,2)+1,2), v(f(i,3)+1,2), v(f(i,1)+1,2)], 'b-');" << std::endl;
		m << "end" << std::endl;

		m.close();
	}
	else {
		return false;
	}

	return true;
}

// TODO: Rework to use CurveUnrolling instead of DiscreteExponentialMap
Eigen::MatrixXd SurfaceStroke::parameterized_space_points_2d(std::shared_ptr<Patch>* cover, std::shared_ptr<DiscreteExponentialMap>* origin_map, Eigen::DenseIndex center_vid) const {
	// Find adequate cover patch for curve
	// From curve center, find radius, and construct patch from there
	std::shared_ptr<Patch> cpatch = cover_patch();

	// Construct DEM 
	if (center_vid < 0) {
		center_vid = cpatch->get_centroid_vid_on_origin_mesh();
	}
	std::shared_ptr<DiscreteExponentialMap> dem = cpatch->discrete_exponential_map(center_vid);

	Eigen::MatrixXd dem_curve;

	if (dem->get_center_vid() < 0) {
		return dem_curve;
	}

	// Transform curve into DEM space
	dem_curve = Eigen::MatrixXd(2, _curve_points.size());

	std::map<Eigen::DenseIndex, Eigen::Vector2d>& dem_verts = (std::map<Eigen::DenseIndex, Eigen::Vector2d>&)dem->get_raw_vertices();
	const Eigen::MatrixXi& dem_faces = dem->get_raw_faces();
	const Eigen::MatrixXi& mesh_faces = _mesh->faces();

	for (std::size_t i = 0; i < _curve_points.size(); ++i) {
		BarycentricCoord bc = _curve_points[i];

		Eigen::MatrixXd face_verts(2, dem_faces.cols());
		for (Eigen::DenseIndex j = 0; j < dem_faces.cols(); ++j) {
			face_verts.col(j) << dem_verts[static_cast<Eigen::DenseIndex>(mesh_faces(bc._fid, j))];
		}

		dem_curve.col(i) << face_verts * bc._coeff;
	}

	if (cover != nullptr) {
		*cover = cpatch;
	}

	if (origin_map != nullptr) {
		*origin_map = dem;
	}

	return dem_curve;
}