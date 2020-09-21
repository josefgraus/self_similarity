#include "curve_unrolling.h"

#include <algorithm>

#include <geometry/mesh.h>
#include <algorithms/shortest_path.h>
#include <geometry/tests/intersection.h>

Eigen::Vector3d LocalTriFace::face_center(const Eigen::MatrixXd& V, Eigen::Vector3d origin) {
	Eigen::Vector3d fc = Eigen::Vector3d::Zero();

	for (std::size_t i = 0; i < _ABC.size(); ++i) {
		fc += (V.row(_ABC[i]).transpose() - origin) / 3.0;
	}

	return fc;
}

LocalTriFace::Edge LocalTriFace::intersected_edge(const Eigen::MatrixXd& V, Eigen::DenseIndex mesh_fid, std::pair<Eigen::Vector3d, Eigen::Vector3d> segment, BarycentricCoord* intersection_point, double v_offset) {
	const std::size_t x = 0, y = 1, z = 2;

	Eigen::Vector3d fc = Eigen::Vector3d::Zero(); // face_center(V);
	Eigen::Vector3d p = segment.first;

	// Segment, with one point within triangle, and one point without
	// TODO: Check on this assumption -- that this is the most direct and distortion-free way to raise a vertex into the tangent plane of the origin
	//		 This distortion-free assumption is *VERY* important, as the curve unrolling routine relies on it to accurately follow the curve through faces
	Eigen::Vector3d q = local_log_map(fc, Eigen::Matrix3d::Identity(), segment.second);	// TODO: Replace this projection with something more resilient to foldover

	assert(std::fabs(p(2)) < std::numeric_limits<double>::epsilon());
	assert(std::fabs(q(2)) < std::numeric_limits<double>::epsilon());

	Eigen::Vector3d d = q - p;
	p += d.normalized() * v_offset;
	d -= d.normalized() * v_offset;

	// Find intersection, if one exists
	for (std::size_t i = 0; i < _ABC.size(); ++i) {
		Eigen::Vector3d a = V.row(_ABC[i]);
		Eigen::Vector3d b = V.row(_ABC[(i + 1) % _ABC.size()]).transpose();

		Eigen::Vector3d denom; denom << d[y] * (a[x] - b[x]) - d[x] * (a[y] - b[y]),
										d[z] * (a[x] - b[x]) - d[x] * (a[z] - b[z]),
										d[z] * (a[y] - b[y]) - d[y] * (a[z] - b[z]);

		Eigen::DenseIndex ind;
		denom.cwiseAbs().maxCoeff(&ind);

		if (std::fabs(denom(ind)) < std::numeric_limits<double>::epsilon()) {
			continue;
		}

		double s;

		switch (static_cast<Edge>(ind)) {
			case Edge::AB: {
				s = (a[x] * (b[y] - p[y]) - b[x] * (a[y] - p[y]) + p[x] * (a[y] - b[y])) / denom(ind);
				break;
			}
			case Edge::BC: {
				s = (a[x] * (b[z] - p[z]) - b[x] * (a[z] - p[z]) + p[x] * (a[z] - b[z])) / denom(ind);
				break;
			}
			case Edge::CA: {
				s = (a[y] * (b[z] - p[z]) - b[y] * (a[z] - p[z]) + p[y] * (a[z] - b[z])) / denom(ind);
				break;
			}
			default: {
				throw std::logic_error("LocalTriFace::intersected_edge() -- Edge index is out of range!");
				break;
			}
		}

		if (s < 0.0 || s > 1.0) {
			// Line segments don't intersect
			continue;
		}

		if (intersection_point != nullptr) {
			Eigen::Vector3d coeff = Eigen::Vector3d::Zero();
			coeff(ind) = 1.0 - s;
			coeff((ind + 1) % coeff.size()) = s;

			*intersection_point = BarycentricCoord(mesh_fid, coeff);
		}

		return static_cast<Edge>(i);
	}

	return Edge::None;
}

CurveUnrolling::CurveUnrolling() :
	_curve(nullptr),
	_root(nullptr),
	_origin_mesh(nullptr),
	_unrolled_stroke(nullptr) {

}

CurveUnrolling::CurveUnrolling(std::shared_ptr<SurfaceStroke> curve): _curve(curve), _root(nullptr), _unrolled_stroke(nullptr) {
	std::shared_ptr<Mesh> mesh = curve->origin_mesh();
	const Eigen::MatrixXi& F = mesh->faces();
	const Eigen::MatrixXd& V = mesh->vertices();
	const std::vector<BarycentricCoord>& curve_points = curve->blade_points();
	const Eigen::SparseMatrix<int>& adj = mesh->adjacency_matrix();
	const Eigen::MatrixXi& tri_adj = mesh->tri_adjacency_matrix();

	_vid_map.clear();
	_fid_map.clear();
	_origin_mesh = mesh;

	if (curve_points.size() <= 0) {
		return;
	}

	// Assuming triangulation
	assert(F.cols() == 3);

	_F = Eigen::MatrixXi(mesh->faces().rows(), 3);
	_V = Eigen::MatrixXd(_F.rows() * _F.cols(), 3);			// Will be 2D overlapping parameterization

	int fCount = 0;
	int vCount = 0;

	Eigen::Vector3d frame_N = Eigen::Vector3d::Zero();

	// Frame normal
	{
		Eigen::Vector3d fa = V.row(F(curve_points.begin()->_fid, 0)).leftCols<3>().transpose();
		Eigen::Vector3d fb = V.row(F(curve_points.begin()->_fid, 1)).leftCols<3>().transpose();
		Eigen::Vector3d fc = V.row(F(curve_points.begin()->_fid, 2)).leftCols<3>().transpose();

		frame_N = (fb - fa).normalized().cross(fc - fa).normalized();
	}

	std::shared_ptr<Eigen::Matrix3d> frame = std::make_shared<Eigen::Matrix3d>(basis_from_plane_normal(frame_N));
	Eigen::Matrix3d frame_inv = frame->inverse();
	_frame = *frame;

	Eigen::Vector3d param_N = frame_inv * frame_N;
	assert((param_N - Eigen::Vector3d::UnitZ()).isZero(1e-7));

	auto prev_point = curve_points.begin();

	// Initialize with first face
	Eigen::Vector3d origin = Eigen::Vector3d::Zero();
	for (Eigen::DenseIndex i = 0; i < F.cols(); ++i) {
		origin += V.row(F(curve_points.begin()->_fid, i)).leftCols<3>() / 3.0;
	}

	_F.row(fCount++) << 0, 1, 2;
	_fid_map.insert(std::make_pair(0, prev_point->_fid));

	for (Eigen::DenseIndex i = 0; i < F.cols(); ++i) {
		Eigen::DenseIndex vid = F(curve_points.begin()->_fid, i);

		_vid_map.insert(std::make_pair(vCount, vid));
		_V.row(vCount++) << (frame_inv * (V.row(vid).leftCols<3>().transpose() - origin).topRows<3>()).transpose();
	}

	// Track stroke as it's unrolled
	_unrolled_stroke = SurfaceStroke::instantiate(nullptr);

	_unrolled_stroke->add_curve_point(0, curve_points.begin()->_coeff);

	for (auto it = (curve_points.begin() + 1); it != curve_points.end(); ++it, ++prev_point) {
		Eigen::DenseIndex fid = it->_fid;

		const BarycentricCoord& next_bc = *it;
		Eigen::Vector3d p = frame_inv * (next_bc.to_world(mesh).topRows<3>() - origin);

		Eigen::DenseIndex start_fid = fCount - 1;

		if (fid == prev_point->_fid) {
			_unrolled_stroke->add_curve_point(fCount - 1, it->_coeff);

			continue;
		}		

		// Follow line segment created by it - prev_point, raising a distortion-free triangle into the frame along the shared edge as we go
		Eigen::DenseIndex from_fid = prev_point->_fid;

		std::vector<Eigen::DenseIndex> path = shortest_path::face_to_face(from_fid, fid, mesh);

		for (std::size_t i = 1; i < path.size(); ++i) {
			Eigen::DenseIndex prev_fid = path[i - 1];
			Eigen::DenseIndex next_fid = path[i];

			LocalTriFace::Edge e = LocalTriFace::Edge::None;
			LocalTriFace::Edge next_edge = LocalTriFace::Edge::None;

			for (Eigen::DenseIndex j = 0; j < F.cols(); ++j) {
				Eigen::DenseIndex prev_vid1 = F(prev_fid, j);
				Eigen::DenseIndex prev_vid2 = F(prev_fid, (j + 1) % F.cols());

				for (Eigen::DenseIndex k = 0; k < F.cols(); ++k) {
					Eigen::DenseIndex next_vid1 = F(next_fid, (k + 1) % F.cols());
					Eigen::DenseIndex next_vid2 = F(next_fid, k);

					if (prev_vid1 == next_vid1 && prev_vid2 == next_vid2) {
						e = static_cast<LocalTriFace::Edge>(j);
						next_edge = static_cast<LocalTriFace::Edge>(k);
						break;
					}
				}

				if (e != LocalTriFace::Edge::None || next_edge != LocalTriFace::Edge::None) {
					break;
				}
			}

			if (e == LocalTriFace::Edge::None) {
				throw std::logic_error("Adjacent edge should exist!");
			}

			// We've found the next triangle along the line segment -- it either contains the segment endpoint or is along the way to said endpoint
			// Create new LocalTriFace for this triangle, and continue along the segment if it doesn't contain the endpoint
			
			// Translate the third vertex of the new triangle relative to the edge of shared with the current triangle
			// then rotate the vertex about the edge to be in the face plane of the origin triangle
			Eigen::DenseIndex new_vid = F(next_fid, (static_cast<Eigen::DenseIndex>(next_edge) + 2) % 3);
			Eigen::DenseIndex new_a = F(next_fid, static_cast<Eigen::DenseIndex>(next_edge));
			Eigen::DenseIndex new_b = F(next_fid, (static_cast<Eigen::DenseIndex>(next_edge) + 1) % 3);
			Eigen::DenseIndex old_a = _F(fCount - 1, (static_cast<Eigen::DenseIndex>(e) + 1) % 3);
			Eigen::DenseIndex old_b = _F(fCount - 1, static_cast<Eigen::DenseIndex>(e));
			
			assert(new_a == _vid_map.at(old_a));
			assert(new_b == _vid_map.at(old_b));

			// Want to find old_C and rotate it into the frame tangent plane
			Eigen::Vector3d new_C = frame_inv * V.row(new_vid).leftCols<3>().transpose();
			Eigen::Vector3d new_A = frame_inv * V.row(new_a).leftCols<3>().transpose();
			Eigen::Vector3d new_B = frame_inv * V.row(new_b).leftCols<3>().transpose();
			Eigen::Vector3d old_A = _V.row(old_a).leftCols<3>().transpose();
			Eigen::Vector3d old_B = _V.row(old_b).leftCols<3>().transpose();
			Eigen::Vector3d new_N = (new_B - new_A).normalized().cross((new_C - new_A).normalized()).normalized();

			// Transform into frame by residual vector
			Eigen::Vector3d to_bitangent = (old_B - old_A).normalized();
			Eigen::Vector3d to_tangent = to_bitangent.cross(param_N).normalized();
			Eigen::Matrix3d to_TBN; to_TBN << to_tangent, to_bitangent, param_N;

			Eigen::Vector3d from_bitangent = (new_B - new_A).normalized();
			Eigen::Vector3d from_tangent = from_bitangent.cross(new_N).normalized();
			Eigen::Matrix3d from_TBN; from_TBN << from_tangent, from_bitangent, new_N;

			// Find rotation to align TBNs
			Eigen::Matrix3d R = to_TBN * from_TBN.inverse();

			// Rotate C and translate it to align the shared edge of the two triangles
			Eigen::Vector3d old_C = R * (new_C - new_A) + old_A;

			assert(((R * (new_A - new_A) + old_A) - old_A).isZero());
			assert(((R * (new_B - new_A) + old_A) - old_B).isZero());

			Eigen::Vector3d old_AB = (old_B - old_A).normalized();
			Eigen::Vector3d old_AC = (old_C - old_A).normalized();
			Eigen::Vector3d old_N = old_AB.cross(old_AC).normalized();
			{
				Eigen::Vector3d sanity_N = R * new_N;
				Eigen::Vector3d sanity_T = R * from_tangent;
				Eigen::Vector3d sanity_B = R * from_bitangent;

				//assert((old_N - param_N).isZero());
				assert((sanity_N - param_N).isZero());
				assert((sanity_T - to_tangent).isZero());
				assert((sanity_B - to_bitangent).isZero());
			}

			// Populate _V and _F with new face
			Eigen::Vector3i new_tri;
			new_tri(static_cast<Eigen::DenseIndex>(next_edge)) = _F(fCount - 1, (static_cast<Eigen::DenseIndex>(e) + 1) % 3);
			new_tri((static_cast<Eigen::DenseIndex>(next_edge) + 1) % 3) = _F(fCount - 1, static_cast<Eigen::DenseIndex>(e));
			new_tri((static_cast<Eigen::DenseIndex>(next_edge) + 2) % 3) = vCount;

			_fid_map.insert(std::make_pair(fCount, next_fid));
			_vid_map.insert(std::make_pair(vCount, new_vid));

			_V.row(vCount++) << old_C.transpose();
			_F.row(fCount++) << new_tri.transpose();
		}

		_unrolled_stroke->add_curve_point(fCount - 1, it->_coeff);
	}

	_F.conservativeResize(fCount, F.cols());
	_V.conservativeResize(vCount, 3);

	_unrolled_stroke->origin_mesh(Mesh::instantiate(_V, _F));
}

CurveUnrolling::CurveUnrolling(std::shared_ptr<Mesh> mesh, const Eigen::MatrixXd& map_points, BarycentricCoord origin, int origin_index, std::shared_ptr<Eigen::Matrix3d> frame, std::shared_ptr<Eigen::Matrix3d> prev_ABC) {
	_curve = nullptr;

	// TODO: Give a center curve index with a particular meaning?
	copy_to_init(mesh, map_points, origin_index, origin, frame, prev_ABC);
}

CurveUnrolling::CurveUnrolling(std::shared_ptr<Mesh> mesh, const CurveUnrolling& cu, Eigen::DenseIndex frame_fid) : _curve(nullptr), _root(nullptr), _unrolled_stroke(nullptr) {
	std::shared_ptr<SurfaceStroke> unrolled_source = cu.unrolled_stroke();
	_curve = unrolled_source;

	Eigen::MatrixXd map_points(2, unrolled_source->blade_points().size());
	const Eigen::MatrixXd& VF = cu.vertices();
	const Eigen::MatrixXi& FF = cu.faces();

	{
		Eigen::DenseIndex mapIndex = 0;

		for (auto bc = unrolled_source->blade_points().begin(); bc != unrolled_source->blade_points().end(); ++bc) {
			map_points.col(mapIndex++) << bc->to_world(VF, FF).topRows<2>();
		}
	}

	//std::size_t curve_center_index;
	//BarycentricCoord cbc = unrolled_source->curve_center(nullptr, &curve_center_index);
	BarycentricCoord cbc;
	cbc._fid = frame_fid; // cu._fid_map.at(cbc._fid);
	cbc._coeff << 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0;
	copy_to_init(mesh, map_points, 0, cbc);
}

void CurveUnrolling::copy_to_init(std::shared_ptr<Mesh> mesh, const Eigen::MatrixXd& mp, int curve_center_index, BarycentricCoord origin_bc, std::shared_ptr<Eigen::Matrix3d> frame, std::shared_ptr<Eigen::Matrix3d> prev_ABC) {
	// Create copy of stroke at centroid vid at an arbitrary orientation
	_unrolled_stroke = SurfaceStroke::instantiate(mesh);	
	_origin_mesh = mesh;
	
	const Eigen::MatrixXd& V = mesh->vertices();
	const Eigen::MatrixXi& F = mesh->faces();
	const Eigen::MatrixXi& tri_adj = mesh->tri_adjacency_matrix();

	//assert(F.cols() == 3);
	//assert(V.cols() == 3);

	_F = Eigen::MatrixXi(mesh->faces().rows(), 3);
	_V = Eigen::MatrixXd(_F.rows() * _F.cols(), 3);			// Will be 2D overlapping parameterization

	int fCount = 0;
	int vCount = 0;
	
	Eigen::DenseIndex frame_fid = origin_bc._fid;	

	Eigen::Vector3d frame_N;

	Eigen::Vector3d fa = V.row(F(frame_fid, 0)).leftCols<3>().transpose();
	Eigen::Vector3d fb = V.row(F(frame_fid, 1)).leftCols<3>().transpose();
	Eigen::Vector3d fc = V.row(F(frame_fid, 2)).leftCols<3>().transpose();

	// TODO: Frame is inconsistently built at the beginning, giving the curve an arbitrary rotation
	//		 This needs to be addressed in some consistent manner, so that the same inputs via difference constructors produce the same results (which is not currently the case
	Eigen::AngleAxis<double> F2F(0.0, Eigen::Vector3d::UnitX());		// Do not rotate at all
	Eigen::Vector3d N = (fb - fa).normalized().cross(fc - fa).normalized();
	if (frame == nullptr) {
		frame_N = N;

		frame = std::make_shared<Eigen::Matrix3d>(basis_from_plane_normal(frame_N));
	} else {
		frame_N = frame->col(2).normalized();

		double d = N.dot(frame_N);
		if (std::fabs(d - 1.0) > std::numeric_limits<double>::epsilon()) {
			F2F = Eigen::AngleAxis<double>(std::acos(std::min(std::max(d, 0.0), 1.0)), N.cross(frame_N).normalized());
		}
	}

	Eigen::Matrix3d frame_inv = frame->inverse();
	_frame = *frame;

	// Initialize with first face
	_F.row(fCount++) << 0, 1, 2;
	_fid_map.insert(std::make_pair(0, frame_fid));

	Eigen::Vector3d origin = origin_bc.to_world(V, F).topRows<3>();

	// Center of curve at origin
	Eigen::Vector2d cpoint = mp.col(curve_center_index);
	Eigen::MatrixXd map_points = mp.colwise() - cpoint;

	if (prev_ABC == nullptr) {
		for (Eigen::DenseIndex i = 0; i < F.cols(); ++i) {
			Eigen::DenseIndex vid = F(frame_fid, i);

			_vid_map.insert(std::make_pair(vCount, vid));
			_V.row(vCount++) << (frame_inv * F2F * (V.row(vid).leftCols<3>().transpose() - origin).topRows<3>()).transpose();
		}
	} else {
		for (Eigen::DenseIndex i = 0; i < prev_ABC->rows(); ++i) {
			Eigen::DenseIndex vid = F(frame_fid, i);

			_vid_map.insert(std::make_pair(vCount, vid));
			_V.row(vCount) << prev_ABC->row(i);
			_V.row(vCount).leftCols<2>() -= cpoint.transpose();
			vCount++;
		}
	}

	Eigen::Vector3d param_N = frame_inv * frame_N;

	{
		Eigen::Vector3i ABC; ABC << _F.row(fCount - 1).transpose();
		std::array<std::shared_ptr<LocalTriFace>, 3> neighbors = { nullptr, nullptr, nullptr };

		_root = std::make_shared<LocalTriFace>(fCount - 1, ABC, neighbors, frame, mesh);
	}

	Eigen::MatrixXd ABC = _V.block<3, 2>(0, 0).transpose();

	assert(point_in_triangle(map_points.col(curve_center_index), _V.block<1, 2>(0, 0).transpose(), _V.block<1, 2>(1, 0).transpose(), _V.block<1, 2>(2, 0).transpose()));

	std::vector<std::pair<Eigen::DenseIndex, Eigen::Vector3d>> unrolled_curve_points;

	int inc = -1;
	for (int n = 0; n < 2; ++n) {
		inc = (n == 0) ? -1 : 1;

		bool next_point_exists = (curve_center_index >= 0) && (curve_center_index + inc >= 0) && 
								 (curve_center_index < map_points.cols()) && (curve_center_index + inc < map_points.cols());

		Eigen::DenseIndex current_fid = frame_fid;
		std::shared_ptr<LocalTriFace> cur = _root;

		for (int i = curve_center_index + inc; next_point_exists; i += inc) {
			Eigen::Vector2d q = map_points.col(i);
			Eigen::Vector2d p = map_points.col(i - inc);
			Eigen::Vector2d r = q - p;

			// Find intersection of line from previous point to this one, and unfold the next triangle per edge intersection
			LocalTriFace::Edge entry_edge = LocalTriFace::Edge::None;
			while (!point_in_triangle(q, ABC.col(0), ABC.col(1), ABC.col(2))) {
				// Determine edge intersection
				LocalTriFace::Edge e = LocalTriFace::Edge::None;

				for (int check = 0; check < 2; ++check) {
					if (e != LocalTriFace::Edge::None) {
						break;
					}

					for (Eigen::DenseIndex j = 0; j < ABC.cols(); ++j) {
						if (entry_edge == static_cast<LocalTriFace::Edge>(j) && check <= 0) {
							continue;
						}

						Eigen::Vector2d intersection;
						if (intersection::line_and_line_segment(p, r, ABC.col(j), ABC.col((j + 1) % ABC.cols()) - ABC.col(j), intersection)) {
							if (e != LocalTriFace::Edge::None) {
								throw std::logic_error("The exit edge should not already be found!");
							}

							e = static_cast<LocalTriFace::Edge>(j);
						}
					}
				}

				if (e == LocalTriFace::Edge::None) {
					std::stringstream ss; ss << mesh->resource_dir() << "//matlab//curve_tri_intersect.m";
					std::ofstream m(ss.str(), std::ofstream::out);

					if (m.is_open()) {
						m << "figure;" << std::endl;
						m << "hold on;" << std::endl;
						m << "axis equal;" << std::endl;
						m << "grid on;" << std::endl;

						m << "v = [ ..." << std::endl;
						for (Eigen::DenseIndex i = 0; i < ABC.cols(); ++i) {
							m << ABC.col(i).transpose() << "; ..." << std::endl;
						}
						m << ABC.col(0).transpose() << "; ..." << std::endl;
						m << "];" << std::endl;

						m << "a = [ " << p.transpose() << " ];" << std::endl;
						m << "b = [ " << (p + r).transpose() << " ];" << std::endl;

						m << "for i=1:size(v,1)-1" << std::endl;
						m << "plot([v(i,1) v(i+1,1)], [v(i,2) v(i+1,2)], 'b-');" << std::endl;
						m << "end" << std::endl;

						m << "plot([a(1) b(1)], [a(2) b(2)], 'm-');" << std::endl;

						m.close();
					}

					for (Eigen::DenseIndex j = 0; j < ABC.cols(); ++j) {
						Eigen::Vector2d intersection;
						if (intersection::line_and_line_segment(p, r, ABC.col(j), ABC.col((j + 1) % ABC.cols()) - ABC.col(j), intersection)) {
							e = static_cast<LocalTriFace::Edge>(j);
						}
					}

					throw std::logic_error("Point is not within face, but line also doesn't cross any edge??");
				}

				// Find neighbor face sharing the intersected edge
				Eigen::DenseIndex next_fid = -1;
				//Eigen::VectorXi tri_adjs = tri_adj.row(current_fid).transpose();

				//assert(tri_adj.minCoeff() >= 0);

				Eigen::VectorXi current_vids = F.row(current_fid).transpose();

				Eigen::DenseIndex prev_vid1 = F(current_fid, static_cast<Eigen::DenseIndex>(e));
				Eigen::DenseIndex prev_vid2 = F(current_fid, (static_cast<Eigen::DenseIndex>(e) + 1) % F.cols());
				entry_edge = LocalTriFace::Edge::None;

				for (Eigen::DenseIndex j = 0; j < tri_adj.cols() && next_fid < 0; ++j) {
					// Find matching edge in next triangle
					Eigen::DenseIndex neighbor = tri_adj(current_fid, j);

					if (neighbor < 0 || neighbor >= F.rows()) {
						continue;
						// throw std::logic_error("Holes in the mesh are not supported!");
					}

					for (Eigen::DenseIndex k = 0; k < F.cols(); ++k) {
						Eigen::DenseIndex next_vid1 = F(neighbor, (k + 1) % F.cols());
						Eigen::DenseIndex next_vid2 = F(neighbor, k);

						if (prev_vid1 == next_vid1 && prev_vid2 == next_vid2) {
							entry_edge = static_cast<LocalTriFace::Edge>(k);
							next_fid = neighbor;
							break;
						}
					}
				}

				if (entry_edge == LocalTriFace::Edge::None) {
					throw std::logic_error("No adjacent edge exists when it should!");
				}

				if (next_fid < 0) {
					throw std::logic_error("There must be a next face along the path!");
				}

				// Bring next fid into frame
				// We've found the next triangle along the line segment -- it either contains the segment endpoint or is along the way to said endpoint
				// Create new LocalTriFace for this triangle, and continue along the segment if it doesn't contain the endpoint

				// Translate the third vertex of the new triangle relative to the edge of shared with the current triangle
				// then rotate the vertex about the edge to be in the face plane of the origin triangle
				Eigen::DenseIndex new_vid = F(next_fid, (static_cast<Eigen::DenseIndex>(entry_edge) + 2) % 3);
				Eigen::DenseIndex new_a = F(next_fid, static_cast<Eigen::DenseIndex>(entry_edge));
				Eigen::DenseIndex new_b = F(next_fid, (static_cast<Eigen::DenseIndex>(entry_edge) + 1) % 3);
				Eigen::DenseIndex old_a = _F(fCount - 1, (static_cast<Eigen::DenseIndex>(e) + 1) % 3);
				Eigen::DenseIndex old_b = _F(fCount - 1, static_cast<Eigen::DenseIndex>(e));

				assert(new_a == _vid_map.at(old_a));
				assert(new_b == _vid_map.at(old_b));

				// Want to find old_C and rotate it into the frame tangent plane
				Eigen::Vector3d new_C = frame_inv * V.row(new_vid).leftCols<3>().transpose();
				Eigen::Vector3d new_A = frame_inv * V.row(new_a).leftCols<3>().transpose();
				Eigen::Vector3d new_B = frame_inv * V.row(new_b).leftCols<3>().transpose();
				Eigen::Vector3d old_A = _V.row(old_a).leftCols<3>().transpose();
				Eigen::Vector3d old_B = _V.row(old_b).leftCols<3>().transpose();
				Eigen::Vector3d new_N = (new_B - new_A).normalized().cross((new_C - new_A).normalized()).normalized();

				// Transform into frame by residual vector
				Eigen::Vector3d to_bitangent = (old_B - old_A).normalized();
				Eigen::Vector3d to_tangent = to_bitangent.cross(param_N).normalized();
				Eigen::Matrix3d to_TBN; to_TBN << to_tangent, to_bitangent, param_N;

				Eigen::Vector3d from_bitangent = (new_B - new_A).normalized();
				Eigen::Vector3d from_tangent = from_bitangent.cross(new_N).normalized();
				Eigen::Matrix3d from_TBN; from_TBN << from_tangent, from_bitangent, new_N;

				// Find rotation to align TBNs
				Eigen::Matrix3d R = to_TBN * from_TBN.inverse();

				// Rotate C and translate it to align the shared edge of the two triangles
				Eigen::Vector3d old_C = R * (new_C - new_A) + old_A;

				assert(((R * (new_A - new_A) + old_A) - old_A).isZero());
				assert(((R * (new_B - new_A) + old_A) - old_B).isZero());

				Eigen::Vector3d old_AB = (old_B - old_A).normalized();
				Eigen::Vector3d old_AC = (old_C - old_A).normalized();
				Eigen::Vector3d old_N = old_AB.cross(old_AC).normalized();
				{
					Eigen::Vector3d sanity_N = R * new_N;
					Eigen::Vector3d sanity_T = R * from_tangent;
					Eigen::Vector3d sanity_B = R * from_bitangent;

					assert((sanity_N - param_N).isZero());
					assert((sanity_T - to_tangent).isZero());
					assert((sanity_B - to_bitangent).isZero());
				}

				// Populate _V and _F with new face
				Eigen::Vector3i new_tri;
				new_tri(static_cast<Eigen::DenseIndex>(entry_edge)) = _F(fCount - 1, (static_cast<Eigen::DenseIndex>(e) + 1) % 3);
				new_tri((static_cast<Eigen::DenseIndex>(entry_edge) + 1) % 3) = _F(fCount - 1, static_cast<Eigen::DenseIndex>(e));
				new_tri((static_cast<Eigen::DenseIndex>(entry_edge) + 2) % 3) = vCount;

				_fid_map.insert(std::make_pair(fCount, next_fid));
				_vid_map.insert(std::make_pair(vCount, new_vid));

				_V.row(vCount++) << old_C.transpose();
				_F.row(fCount++) << new_tri.transpose();

				// Create new triangle record
				std::shared_ptr<LocalTriFace> next;
				{
					std::array<std::shared_ptr<LocalTriFace>, 3> neighbors = { nullptr, nullptr, nullptr };
					neighbors[static_cast<std::size_t>(entry_edge)] = cur;

					next = std::make_shared<LocalTriFace>(fCount - 1, new_tri, neighbors, frame, mesh);
				}

				//assert(cur->_neighbors[static_cast<std::size_t>(e)] == nullptr);

				cur->_neighbors[static_cast<std::size_t>(e)] = next;
				cur = next;

				// Update current face, ABC, continue
				ABC << _V.block(_F(fCount - 1, 0), 0, 1, 2).transpose(),
					   _V.block(_F(fCount - 1, 1), 0, 1, 2).transpose(),
					   _V.block(_F(fCount - 1, 2), 0, 1, 2).transpose();

				current_fid = next_fid;
			}

			// Found triangle containing curve point -- now get barycentric coordinates
			if (current_fid >= F.rows()) {
				throw std::logic_error("Every point should be in at least one triangle (possibly more as DEM is not bijective)!");
			}

			// Find barycentric coordinates of the point within the triangle
			// https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
			// TODO: Consider moving to utility/geometry class since this is now used in multiple locations
			//		 A submission deadline is the reason for the copy-paste laziness you see before you
			Eigen::Vector2d a = ABC.col(0);
			Eigen::Vector2d b = ABC.col(1);
			Eigen::Vector2d c = ABC.col(2);

			Eigen::Vector2d v0 = b - a;
			Eigen::Vector2d v1 = c - a;
			Eigen::Vector2d v2 = q - a;
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

			unrolled_curve_points.push_back(std::make_pair(fCount - 1, (Eigen::Vector3d() << u, v, w).finished()));

			next_point_exists = (i + inc >= 0) && (i + inc < map_points.cols());
		}

		if (n == 0) {
			std::reverse(std::begin(unrolled_curve_points), std::end(unrolled_curve_points));

			for (std::pair<Eigen::DenseIndex, Eigen::Vector3d>& bc : unrolled_curve_points) {
				bc.first = std::abs(bc.first - (fCount - 1));
			}

			_F.topRows(fCount) = Eigen::Reverse<Eigen::MatrixXi, 0>(_F.topRows(fCount));

			ABC << _V.block(_F(fCount - 1, 0), 0, 1, 2).transpose(),
        		   _V.block(_F(fCount - 1, 1), 0, 1, 2).transpose(),
				   _V.block(_F(fCount - 1, 2), 0, 1, 2).transpose();

			// Remap faces to reverse order indices
			std::map<Eigen::DenseIndex, Eigen::DenseIndex> remapped_fids;
			
			for (auto fid_pair : _fid_map) {
				remapped_fids.insert(std::make_pair(std::abs(fid_pair.first - static_cast<Eigen::DenseIndex>(_fid_map.size() - 1)), fid_pair.second));
			}
			
			_fid_map = remapped_fids;

			// Next iteration start at the original curve point
			if (curve_center_index != 0) {
				curve_center_index--;
			} else {
				unrolled_curve_points.push_back(std::make_pair(fCount - 1, origin_bc._coeff));
			}
		}
	}

	for (auto bc_con : unrolled_curve_points) {
		_unrolled_stroke->add_curve_point(bc_con.first, bc_con.second);
	}

	assert(_unrolled_stroke->blade_points().size() == mp.cols());

	_F.conservativeResize(fCount, F.cols());
	_V.conservativeResize(vCount, 3);

	_V.leftCols<2>() = _V.leftCols<2>().rowwise() + cpoint.transpose();
}

CurveUnrolling::~CurveUnrolling() {
}

std::shared_ptr<CurveUnrolling> CurveUnrolling::clone() const {
	std::shared_ptr<CurveUnrolling> duplicate = std::shared_ptr<CurveUnrolling>(new CurveUnrolling());

	duplicate->_curve = _curve;	// This is the original curve that was unrolled, and should not be cloned
	duplicate->_V = _V;
	duplicate->_vid_map = _vid_map;
	duplicate->_F = _F;
	duplicate->_fid_map = _fid_map;
	duplicate->_root = nullptr;	// This hasn't found a use -- probably not worth the effort to duplicate face graph
	duplicate->_frame = _frame;
	duplicate->_origin_mesh = _origin_mesh;	// Same reasoning for not cloning _curve
	duplicate->_unrolled_stroke = _unrolled_stroke->clone();	// The generated flattened stroke, however, should be cloned

	return duplicate;
}

void CurveUnrolling::transform(double x, double y, double radians) {
	int blade_point_index = 0;

	Eigen::MatrixXd unrolled_map_points(2, _unrolled_stroke->blade_points().size() + 1);
	for (int i = 1; i < _unrolled_stroke->blade_points().size() + 1; ++i) {
		unrolled_map_points.col(i) << _unrolled_stroke->blade_points().at(i-1).to_world(_V, _F).topRows<2>();
	}

	// Transforms
	Eigen::Rotation2D<double> R(radians);
	Eigen::Vector2d T;
	T << x, y;

	// Transform points
	Eigen::MatrixXd transformed_map_points = (R.toRotationMatrix() * unrolled_map_points).colwise() + T;

	// Determine the triangle containing the curve center point, then proceed with curve walking/unrolling per second CurveUnroll constructor
	BarycentricCoord search_origin = _unrolled_stroke->blade_points().at(blade_point_index);
	transformed_map_points.col(0) << unrolled_map_points.col(blade_point_index + 1);

	// Frame and previous start face propagation
	std::shared_ptr<Eigen::Matrix3d> u_frame = std::make_shared<Eigen::Matrix3d>(_frame);
	std::shared_ptr<Eigen::Matrix3d> prev_ABC = std::make_shared<Eigen::Matrix3d>();
	for (int i = 0; i < 3; ++i) {
		prev_ABC->row(i) = _V.row(_F(0, i));
	}
	
	search_origin._fid = fid_map().at(search_origin._fid);
	
	CurveUnrolling tcu(_origin_mesh, transformed_map_points, search_origin, blade_point_index, u_frame, prev_ABC);

	/*std::stringstream ss; ss << _origin_mesh->resource_dir() << "//matlab//transformed.m";
	std::ofstream m(ss.str(), std::ofstream::out);

	if (m.is_open()) {
		m << "figure;" << std::endl;
		m << "hold on;" << std::endl;
		m << "axis equal;" << std::endl;
		m << "grid on;" << std::endl;

		m << "o = [ ..." << std::endl;
		for (Eigen::DenseIndex i = 1; i < unrolled_map_points.cols(); ++i) {
			m << unrolled_map_points.col(i).transpose() << "; ..." << std::endl;
		}
		m << "];" << std::endl;

		m << "t = [ ..." << std::endl;
		for (Eigen::DenseIndex i = 0; i < transformed_map_points.cols(); ++i) {
			m << transformed_map_points.col(i).transpose() << "; ..." << std::endl;
		}
		m << "];" << std::endl;

		m << "v = [ " << unrolled_map_points.col(blade_point_index + 1).transpose() << "; " << transformed_map_points.col(blade_point_index + 1).transpose() << "];" << std::endl;

		m << "Vo = [ ..." << std::endl;
		for (Eigen::DenseIndex i = 0; i < _V.rows(); ++i) {
			m << _V.row(i) << "; ..." << std::endl;
		}
		m << "];" << std::endl;

		m << "Fo = [ ..." << std::endl;
		for (Eigen::DenseIndex i = 0; i < _F.rows(); ++i) {
			m << _F.row(i) << "; ..." << std::endl;
		}
		m << "];" << std::endl;

		m << "Vt = [ ..." << std::endl;
		for (Eigen::DenseIndex i = 0; i < tcu.vertices().rows(); ++i) {
			m << tcu.vertices().row(i) << "; ..." << std::endl;
		}
		m << "];" << std::endl;

		m << "Ft = [ ..." << std::endl;
		for (Eigen::DenseIndex i = 0; i < tcu.faces().rows(); ++i) {
			m << tcu.faces().row(i) << "; ..." << std::endl;
		}
		m << "];" << std::endl;

		m << "T = [ " << T.transpose() << " ];" << std::endl;
		m << "r = " << radians << ";" << std::endl;

		m << "for i=1:size(o,1)-1" << std::endl;
		m << "plot([o(i,1) o(i+1,1)], [o(i,2) o(i+1,2)], 'b-');" << std::endl;
		m << "end" << std::endl;

		m << "for i=1:size(Fo,1)" << std::endl;
		m << "for j=1:size(Fo,2)" << std::endl;
		m << "nj = j+1;" << std::endl;
		m << "if nj > size(Fo,2)" << std::endl;
		m << "nj = 1;" << std::endl;
		m << "end" << std::endl;
		m << "plot([Vo(Fo(i,j)+1,1) Vo(Fo(i,nj)+1,1)], [Vo(Fo(i,j)+1,2) Vo(Fo(i,nj)+1,2)], 'g-');" << std::endl;
		m << "end" << std::endl;
		m << "end" << std::endl;

		m << "for i=1:size(t,1)-1" << std::endl;
		m << "plot([t(i,1) t(i+1,1)], [t(i,2) t(i+1,2)], 'm-');" << std::endl;
		m << "end" << std::endl;

		m << "for i=1:size(Ft,1)" << std::endl;
		m << "for j=1:size(Ft,2)" << std::endl;
		m << "nj = j+1;" << std::endl;
		m << "if nj > size(Ft,2)" << std::endl;
		m << "nj = 1;" << std::endl;
		m << "end" << std::endl;
		m << "plot([Vt(Ft(i,j)+1,1) Vt(Ft(i,nj)+1,1)], [Vt(Ft(i,j)+1,2) Vt(Ft(i,nj)+1,2)], 'r-');" << std::endl;
		m << "end" << std::endl;
		m << "end" << std::endl;

		m << "plot([v(1,1) v(2,1)], [v(1,2) v(2,2)], 'y-');" << std::endl;

		m.close();
	}*/

	// Remove first point of curve and all faces related to its unfolding up to the second point
	_unrolled_stroke = tcu.unrolled_stroke();
	_unrolled_stroke->_curve_points.erase(std::begin(_unrolled_stroke->_curve_points));

	auto F = tcu.faces();
	auto V = tcu.vertices();

	int i = 0;
	while (!point_in_triangle(transformed_map_points.col(blade_point_index + 1), 
		                      V.block<1,2>(F(i,0), 0).transpose(), 
		                      V.block<1,2>(F(i,1), 0).transpose(),
							  V.block<1,2>(F(i,2), 0).transpose())) {
		i++;
	}

	_F = tcu.faces().bottomRows(tcu.faces().rows() - i);
	
	// Update fid map (toss out face mappings to removed faces, and decrease fid value for each remaining by i
	std::map<Eigen::DenseIndex, Eigen::DenseIndex> culled_fid_map;

	for (auto& kv: tcu.fid_map()) {
		if (kv.first < i) {
			continue;
		}

		culled_fid_map.insert(std::make_pair(kv.first - i, kv.second));
	}
	
	for (int j = 0; j < i; ++j) {
		if (_unrolled_stroke->_fids.count(_fid_map[j]) > 0) {
			_unrolled_stroke->_fids.erase(_fid_map[j]);
		}
	}

	for (auto& p : _unrolled_stroke->_curve_points) {
		p._fid -= i;
	}

	_fid_map = culled_fid_map;

	_V = tcu.vertices();
	_vid_map = tcu.vid_map();
	
	_root = tcu._root;
	assert((_frame - tcu.frame()).isZero(1e-7));

	_origin_mesh = tcu.origin_mesh();

	// Don't update curve
}

// Redescribe the frame in terms of triangle containing _curve[blade_point_index]
void CurveUnrolling::reframe(Eigen::DenseIndex blade_point_index) {
	if (blade_point_index < 0) {
		std::size_t curve_center_index;
		_unrolled_stroke->curve_center(nullptr, &curve_center_index);
		blade_point_index = curve_center_index;
	}

	// Find new frame
	Eigen::DenseIndex local_fid = unrolled_stroke()->blade_points()[blade_point_index]._fid;
	Eigen::DenseIndex frame_fid = _fid_map.at(local_fid);

	Eigen::Vector3d frame_N = Eigen::Vector3d::Zero();

	const Eigen::MatrixXi& F = _origin_mesh->faces();
	const Eigen::MatrixXd& V = _origin_mesh->vertices();

	// Frame normal
	{
		Eigen::Vector3d fa = V.row(F(frame_fid, 0)).leftCols<3>().transpose();
		Eigen::Vector3d fb = V.row(F(frame_fid, 1)).leftCols<3>().transpose();
		Eigen::Vector3d fc = V.row(F(frame_fid, 2)).leftCols<3>().transpose();

		frame_N = (fb - fa).normalized().cross(fc - fa).normalized();
	}

	// TODO: Frame is inconsistently built at the beginning, giving the curve an arbitrary rotation
	//		 This needs to be addressed in some consistent manner, so that the same inputs via different constructors produce the same results (which is not currently the case)
	Eigen::Matrix3d new_frame = basis_from_plane_normal(frame_N);
	Eigen::MatrixXd new_frame_inv = new_frame.inverse();

	// Describe vertices in new frame
	Eigen::Vector2d p_f0 = (_V.row(_F(local_fid, 1)) - _V.row(_F(local_fid, 0))).leftCols<2>().transpose().normalized();
	Eigen::Vector2d p_f1 = (new_frame_inv * (V.row(F(frame_fid, 1)).leftCols<3>() - V.row(F(frame_fid, 0)).leftCols<3>()).transpose()).topRows<2>().normalized();
	
	double R_f0_f1 = std::acos(std::min(1.0, std::max(0.0, p_f0.dot(p_f1))));

	Eigen::Rotation2D<double> R(R_f0_f1);
	Eigen::Rotation2D<double> R_neg(-1.0 * R_f0_f1);

	//Eigen::Vector2d test = R.toRotationMatrix() * p_f0 - p_f1;

	if ((R.toRotationMatrix() * p_f0 - p_f1).norm() > (R_neg.toRotationMatrix() * p_f0 - p_f1).norm()) {
		R = R_neg;
	}

	Eigen::Vector2d offset = p_f1 - R.toRotationMatrix() * p_f0;

	//test = R.toRotationMatrix() * p_f0 - p_f1;

	assert((R.toRotationMatrix() * p_f0 - p_f1 + offset).isZero(1e-7));

	Eigen::Vector3d origin = _V.row(_F(local_fid, 0));

	_V.leftCols<2>() = ((R.toRotationMatrix() * (_V.rowwise() - origin.transpose()).leftCols<2>().transpose()).colwise() + offset).transpose().rowwise() + origin.topRows<2>().transpose();

	_frame = new_frame;
}

// This function will either return an identical curve to the source, if directly unrolled (first constructor), or will return a repositioned curve on the source mesh if copied (second constructor)
std::shared_ptr<SurfaceStroke> CurveUnrolling::unrolled_on_origin_mesh() const {
	if (_unrolled_stroke == nullptr) {
		return nullptr;
	}

	// Create curve back on source
	std::shared_ptr<SurfaceStroke> unrolled_on_origin = SurfaceStroke::instantiate(origin_mesh());

	for (int i = 0; i < _unrolled_stroke->blade_points().size(); ++i) {
		const BarycentricCoord& local_bc = _unrolled_stroke->blade_points().at(i);

		unrolled_on_origin->add_curve_point(_fid_map.at(local_bc._fid), local_bc._coeff);
	}

	return unrolled_on_origin;
}

Eigen::MatrixXd CurveUnrolling::curve_points_2d() {
	Eigen::MatrixXd unrolled_map_points(2, _unrolled_stroke->blade_points().size());
	for (int i = 0; i < _unrolled_stroke->blade_points().size(); ++i) {
		unrolled_map_points.col(i) << _unrolled_stroke->blade_points().at(i).to_world(_V, _F).topRows<2>();
	}

	return unrolled_map_points;
}

Eigen::Vector3d CurveUnrolling::bc_to_frame(Eigen::DenseIndex unroll_fid, BarycentricCoord mesh_bc) {
	if (unroll_fid < 0 || unroll_fid > _fid_map.size()) {
		throw std::invalid_argument("unroll fid must be within bounds!");
	}

	Eigen::Vector3d resolved = Eigen::Vector3d::Zero();

	for (Eigen::DenseIndex i = 0; i < _F.cols(); ++i) {
		resolved += _V.row(_F(unroll_fid, i)) * mesh_bc._coeff(i);
	}

	return resolved;
}

bool CurveUnrolling::to_matlab(std::string script_out_path, Eigen::DenseIndex start_fid, std::pair<BarycentricCoord, Eigen::Vector3d>* segment, std::shared_ptr<LocalTriFace> tri) {
	return false;

	std::ofstream m(script_out_path, std::ofstream::out);

	static std::vector<Eigen::Vector3d> ps;
	static std::vector<Eigen::Vector3d> qs;

	if (m.is_open()) {
		m << "figure;" << std::endl;
		m << "hold on;" << std::endl;
		m << "axis equal;" << std::endl;
		m << "grid on;" << std::endl;

		auto V = vertices();
		m << "v = [ ..." << std::endl;
		for (Eigen::DenseIndex i = 0; i < _vid_map.size(); ++i) {
			m << V.row(i) << "; ..." << std::endl;
		}
		m << "];" << std::endl;

		const Eigen::MatrixXi& F = faces();
		m << "f = [ ..." << std::endl;
		for (Eigen::DenseIndex i = 0; i < _fid_map.size(); ++i) {
			m << F.row(i) << "; ..." << std::endl;
		}
		m << "];" << std::endl;
		m << "nt = [ 0, 0, 1];" << std::endl;
		m << "for i=1:size(f,1)" << std::endl;
		m << "a = [v(f(i, 1) + 1, :)];" << std::endl;
		m << "b = [v(f(i, 2) + 1, :)];" << std::endl;
		m << "c = [v(f(i, 3) + 1, :)];" << std::endl;
		m << "cb = (c - b) / norm(c - b);" << std::endl;
		m << "ab = (a - b) / norm(a - b);" << std::endl;
		m << "n = cross(cb, ab);" << std::endl;
		m << "n = n / norm(n);" << std::endl;
		m << "if i == 1" << std::endl;
		m << "nt = n;" << std::endl;
		m << "end" << std::endl;
		m << "t = dot(n, nt);" << std::endl;
		m << "C = [1.0, 0.0, 0.0] * (1 - ((t + 1) / 2)) + [0.0, 1.0, 0.0] * ((t + 1) / 2);" << std::endl;
		m << "h = fill3([a(1), b(1), c(1)], [a(2), b(2), c(2)], [a(3), b(3), c(3)], C);" << std::endl;
		m << "set(h, 'facealpha', .5);" << std::endl;
		m << "end" << std::endl;

		m << "scatter3(v(:,1), v(:,2), v(:,3), 'mo');" << std::endl;

		if (segment != nullptr && tri != nullptr) {
			Eigen::Vector3d fc = Eigen::Vector3d::Zero(); // tri->face_center(V);
			Eigen::Vector3d p = bc_to_frame(start_fid, segment->first);
			Eigen::Vector3d q = local_log_map(fc, Eigen::Matrix3d::Identity(), segment->second);
			Eigen::Vector3d d = q - p;

			ps.push_back(p);
			qs.push_back(q);

			m << "p = [ ..." << std::endl;
			for (Eigen::DenseIndex i = 0; i < ps.size(); ++i) {
				m << ps[i].transpose() << "; ..." << std::endl;
			}
			m << "];" << std::endl;

			m << "q = [ ..." << std::endl;
			for (Eigen::DenseIndex i = 0; i < qs.size(); ++i) {
				m << qs[i].transpose() << "; ..." << std::endl;
			}
			m << "];" << std::endl;

			m << "fc = [ " << fc.transpose() << " ];" << std::endl;
			m << "for i=1:size(p,1)" << std::endl;
			m << "plot3([p(i,1) q(i,1)], [p(i,2) q(i,2)], [p(i,3) q(i,3)], 'b-');" << std::endl;
			m << "end" << std::endl;
			m << "scatter3(fc(1), fc(2), fc(3), 'rx');" << std::endl;
		}

		m << "T = [ " << Eigen::Vector3d::UnitX().transpose() << " ];" << std::endl;
		m << "B = [ " << Eigen::Vector3d::UnitY().transpose() << " ];" << std::endl;
		m << "N = [ " << Eigen::Vector3d::UnitZ().transpose() << " ];" << std::endl;

		m << "a = 0.1;" << std::endl;
		m << "plot3([fc(1), a * T(1) + fc(1)], [fc(2), a * T(2) + fc(2)], [fc(3), a * T(3) +  fc(3)], 'r-');" << std::endl;
		m << "plot3([fc(1), a * B(1) + fc(1)], [fc(2), a * B(2) + fc(2)], [fc(3), a * B(3) +  fc(3)], 'y-');" << std::endl;
		m << "plot3([fc(1), a * N(1) + fc(1)], [fc(2), a * N(2) + fc(2)], [fc(3), a * N(3) +  fc(3)], 'm-');" << std::endl;

		m.close();
	}
	else {
		return false;
	}

	return true;
}