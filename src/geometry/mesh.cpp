#include "mesh.h"

#include <array>
#include <algorithm>
#include <fstream>
#include <limits>
#include <random>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/grad.h>
#include <igl/cotmatrix.h>
#include <igl/colormap.h>
#include <igl/eigs.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/adjacency_matrix.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/per_vertex_normals.h>
#include <igl/arap.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>

#include <Eigen/Eigenvalues> 

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <geometry/patch.h>
#include <matching/geodesic_fan.h>
#include <shape_signatures/heat_kernel_signature.h>
#include <matching/surface_stroke.h>
#include <matching/parameterization/curve_unrolling.h>


struct MeshInstancer : public Mesh, std::enable_shared_from_this<Mesh> {
	MeshInstancer(std::string obj_path) : Mesh(obj_path) { }
	MeshInstancer(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F): Mesh(V, F) { }
};

Mesh::Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) :
	_V(Eigen::MatrixXd::Zero(0, 0)),
	_TC(Eigen::MatrixXd::Zero(0, 0)),
	_N(Eigen::MatrixXd::Zero(0, 0)),
	_F(Eigen::MatrixXi::Zero(0, 0)),
	_FTC(Eigen::MatrixXi::Zero(0, 0)),
	_FN(Eigen::MatrixXd::Zero(0, 0)),
	_VN(Eigen::MatrixXd::Zero(0, 0)),
	_adj(Eigen::SparseMatrix<int>(0, 0)),
	_base_colors(Eigen::MatrixXd::Zero(0, 0)),
	_face_colors(Eigen::MatrixXd::Zero(0, 0)),
	_loaded(false),
	_model_name(""),
	_data_dir(""),
	_selection_patch(nullptr),
	_displayed_points(Eigen::MatrixXd::Zero(0, 6)) {

	std::get<0>(_color_texture) = std::make_shared<ColorChannel>();
	std::get<1>(_color_texture) = std::make_shared<ColorChannel>();
	std::get<2>(_color_texture) = std::make_shared<ColorChannel>();
	std::get<3>(_color_texture) = std::make_shared<ColorChannel>();

	_V = V;
	_F = F;

	// Compute normals
	igl::per_vertex_normals(_V, _F, _VN);
	igl::per_face_normals(_V, _F, _FN);

	_VN.rowwise().normalize();
	_FN.rowwise().normalize();

	// Create blank color map
	_base_colors = Eigen::MatrixXd::Constant(_F.rows(), 3, 1.0);
	_face_colors = _base_colors;

	// Create adjacency matrix
	igl::adjacency_matrix(_F, _adj);
	igl::triangle_triangle_adjacency(_F, _tri_adj);

	_VC = _V;
	_FC = _F;
	_face_colors_C = _face_colors;
}

Mesh::Mesh(std::string obj_path) :
	_V(Eigen::MatrixXd::Zero(0, 0)), 
	_TC(Eigen::MatrixXd::Zero(0, 0)), 
	_N(Eigen::MatrixXd::Zero(0, 0)),
	_F(Eigen::MatrixXi::Zero(0, 0)), 
	_FTC(Eigen::MatrixXi::Zero(0, 0)), 
	_FN(Eigen::MatrixXd::Zero(0, 0)),
	_VN(Eigen::MatrixXd::Zero(0, 0)),
	_adj(Eigen::SparseMatrix<int>(0,0)),
	_base_colors(Eigen::MatrixXd::Zero(0, 0)),
	_face_colors(Eigen::MatrixXd::Zero(0, 0)),
	_loaded(false),
	_model_name(""),
	_data_dir(""),
	_selection_patch(nullptr), 
	_displayed_points(Eigen::MatrixXd::Zero(0,6)) {

	std::get<0>(_color_texture) = std::make_shared<ColorChannel>();
	std::get<1>(_color_texture) = std::make_shared<ColorChannel>();
	std::get<2>(_color_texture) = std::make_shared<ColorChannel>();
	std::get<3>(_color_texture) = std::make_shared<ColorChannel>();

	if (!(_loaded = load_obj(obj_path))) {
		//throw std::invalid_argument("Could not load model [ " + obj_path + " ]!\n");
		return;
	}

	_model_path = obj_path;

	// Scale Vertices to unit cube
	double scale_factor = _V.leftCols<3>().maxCoeff() / 2.0;
	_V.leftCols<3>() /= scale_factor;
	_V.leftCols<3>().rowwise() - Eigen::Vector3d(1.0, 1.0, 1.0).transpose();

	// Create blank color map
	_base_colors = Eigen::MatrixXd::Constant(_F.rows(), 3, 1.0);
	_face_colors = _base_colors;

	// Create adjacency matrix
	igl::adjacency_matrix(_F, _adj);
	igl::triangle_triangle_adjacency(_F, _tri_adj);

	// Right now only supporting triangle meshes
	if (_F.cols() == 3) {
		// Create halfedge structure
		std::vector<trimesh::triangle_t> triangles;
		triangles.resize(_F.rows());

		for (Eigen::DenseIndex i = 0; i < _F.rows(); ++i) {
			for (Eigen::DenseIndex j = 0; j < 3; ++j) {
				triangles[i].v[j] = _F(i, j);
			}
		}

		std::vector<trimesh::edge_t> edges;
		trimesh::unordered_edges_from_triangles(triangles.size(), &triangles[0], edges);

		halfedge_mesh.build(_V.rows(), triangles.size(), &triangles[0], edges.size(), &edges[0]);
	}

	_VC = _V;
	_FC = _F;
	_face_colors_C = _face_colors;
}

Mesh::~Mesh() {
}

std::shared_ptr<Mesh> Mesh::instantiate(std::string obj_path) {
	return std::make_shared<MeshInstancer>(obj_path);
}

std::shared_ptr<Mesh> Mesh::instantiate(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
	return std::make_shared<MeshInstancer>(V, F);
}

const trimesh::trimesh_t& Mesh::halfedge() {
	return halfedge_mesh;
}

void Mesh::display(igl::opengl::glfw::Viewer& viewer) {
	if (_V.cols() == 3) {
		viewer.data().set_mesh(_V.block(0, 0, _V.rows(), 3), _F);

		if (std::get<0>(_color_texture)->cols() > 0 && 
			std::get<1>(_color_texture)->cols() > 0 &&
			std::get<2>(_color_texture)->cols() >0 &&
			std::get<0>(_color_texture)->cols() == std::get<1>(_color_texture)->cols() &&
			std::get<1>(_color_texture)->cols() == std::get<2>(_color_texture)->cols()) {
			if (_TC.cols() > 0 && _FTC.cols() > 0) {
				viewer.data().set_uv(_TC, _FTC);
			}

			viewer.data().set_texture(*std::get<0>(_color_texture), *std::get<1>(_color_texture), *std::get<2>(_color_texture));
			viewer.data().show_texture = true;
		} else {
			viewer.data().set_colors(_face_colors);
		}
	} else if (_V.cols() == 6) {
		viewer.data().set_mesh(_V.block(0, 0, _V.rows(), 3), _F);
		viewer.data().set_colors(_V.block(0, 3, _V.rows(), 3));
	}

	viewer.core.align_camera_center(_V.block(0, 0, _V.rows(), 3), _F);
}

bool Mesh::set_scalar_vertex_color(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& color_per_vertex, igl::ColorMapType cm) {
	return set_scalar_vertex_color(viewer, color_per_vertex, color_per_vertex.minCoeff(), color_per_vertex.maxCoeff(), cm);
}

bool Mesh::set_scalar_vertex_color(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& color_per_vertex, double z_min, double z_max, igl::ColorMapType cm) {
	// Defaults to use inferno for now
	Eigen::MatrixXd per_vert_colors;

	igl::colormap(cm, Eigen::VectorXd(color_per_vertex.col(0)), z_min, z_max, per_vert_colors);

	if (std::fabs(per_vert_colors.maxCoeff() - per_vert_colors.minCoeff()) > 1e-7) {
		per_vert_colors = (per_vert_colors.array() - per_vert_colors.minCoeff());
		per_vert_colors /= per_vert_colors.maxCoeff();

		double max_coeff = per_vert_colors.maxCoeff();
		double min_coeff = per_vert_colors.minCoeff();

		assert(std::fabs(per_vert_colors.maxCoeff() - 1.0) < 1e-7 && std::fabs(per_vert_colors.minCoeff()) < 1e-7);
	}

	// Sum up vertex colors into a face color
	_base_colors = Eigen::MatrixXd::Zero(_F.rows(), 3);

	for (Eigen::DenseIndex i = 0; i < _base_colors.rows(); ++i) {
		for (Eigen::DenseIndex j = 0; j < _F.cols(); ++j) {
			_base_colors.row(i) += per_vert_colors.row(_F(i, j)) / _FC.cols();
		}
	}

	refresh_colormap();

	viewer.data().set_colors(_face_colors_C);

	return true;
}

bool Mesh::select_vid_with_point(igl::opengl::glfw::Viewer& viewer, Eigen::DenseIndex vid, Eigen::Vector3d color) {
	if (_V.rows() <= vid) {
		return false;
	}

	viewer.data().points = _displayed_points;
	viewer.data().add_points(_V.block(vid, 0, 1, 3), color.transpose());

	return true;
}

bool Mesh::write_obj(std::string path) {
	return igl::writeOBJ(path, _VC, _FC);
}

bool Mesh::load_obj(std::string objPath) {
	Eigen::MatrixXi FN;

	if (!igl::readOBJ(objPath, _V, _TC, _N, _F, _FTC, FN)) {
		return false;
	}

	if (FN.rows() <= 0 || FN.cols() <= 0) {
		// Compute normals since the file didn't include them
		igl::per_vertex_normals(_V, _F, _VN);
		igl::per_face_normals(_V, _F, _FN);
	} else {
		_N.rowwise().normalize();
		_VN = Eigen::MatrixXd::Zero(_V.rows(), 3);

		for (Eigen::DenseIndex i = 0; i < _F.rows(); ++i) {
			for (Eigen::DenseIndex j = 0; j < _F.cols(); ++j) {
				_VN.row(_F(i, j)) += _N.row(FN(i, j));
			}
		}

		_FN = _N;
	}

	_VN.rowwise().normalize();
	_FN.rowwise().normalize();

	std::replace(objPath.begin(), objPath.end(), '\\', '/');
	_model_name = objPath.substr(objPath.find_last_of('/') + 1, objPath.find_last_of('.'));
	_data_dir = objPath.substr(0, objPath.find_last_of('/') + 1);

	return true;
}

bool Mesh::load_texture(std::string texture_path) {
	int w, h, n;
	unsigned char *data = stbi_load(texture_path.c_str(), &w, &h, &n, 4);
	if (data == NULL) {
		return false;
	}

	int comp = 4;

	std::get<0>(_color_texture) = std::make_shared<ColorChannel>(w, h);
	std::get<1>(_color_texture) = std::make_shared<ColorChannel>(w, h);
	std::get<2>(_color_texture) = std::make_shared<ColorChannel>(w, h);
	std::get<3>(_color_texture) = std::make_shared<ColorChannel>(w, h);

	for (unsigned y = 0; y < h; ++y) {
		for (unsigned x = 0; x < w; ++x) {
			(*std::get<0>(_color_texture))(x, h - 1 - y) = data[comp * (x + w * y) + 0];
			(*std::get<1>(_color_texture))(x, h - 1 - y) = data[comp * (x + w * y) + 1];
			(*std::get<2>(_color_texture))(x, h - 1 - y) = data[comp * (x + w * y) + 2];
			(*std::get<3>(_color_texture))(x, h - 1 - y) = data[comp * (x + w * y) + 3];
		}
	}

	stbi_image_free(data);

	return true;
}

bool Mesh::save_texture(std::string texture_path) {
	int w = std::get<0>(_color_texture)->cols();
	int h = std::get<0>(_color_texture)->rows();
	int comp = 4;

	unsigned char* data = new unsigned char[w * h * 4];

	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			int i = comp * (x + w * y);
			data[i + 0] = (*std::get<0>(_color_texture))(x, h - 1 - y);
			data[i + 1] = (*std::get<1>(_color_texture))(x, h - 1 - y);
			data[i + 2] = (*std::get<2>(_color_texture))(x, h - 1 - y);
			data[i + 3] = (*std::get<3>(_color_texture))(x, h - 1 - y);
		}
	}

	stbi_write_png(texture_path.c_str(), w, h, comp, data, 0);

	delete data;

	return true;
}

int Mesh::closest_vertex_id(int fid, Eigen::Vector3f bc) {
	int vid;

	bc.maxCoeff(&vid);

	return _F(fid, vid);
}

bool Mesh::select_faces_with_color(igl::opengl::glfw::Viewer& viewer, std::set<Eigen::DenseIndex> fids, Eigen::Vector3d color) {
	for (auto fid : fids) {
		if (_face_colors_C.rows() <= fid) {
			return false;
		}
	}

	if (_selection_patch == nullptr) {
		std::shared_ptr<Mesh> mesh = dynamic_cast<MeshInstancer*>(this)->shared_from_this();
		_selection_patch = Patch::instantiate(mesh);
	}

	for (auto fid : fids) {
		_face_colors_C.row(fid) << color.transpose();
	}

	_selection_patch->add(fids);

	viewer.data().set_colors(_face_colors_C);

	return true;
}

bool Mesh::color_faces(igl::opengl::glfw::Viewer& viewer, std::vector<int> fids, std::vector<Eigen::Vector3d> colors) {
	for (std::size_t i = 0; i < std::min(fids.size(), colors.size()); ++i) {
		if (_face_colors_C.rows() <= fids[i]) {
			continue;
		}

		_face_colors_C.row(fids[i]) << colors[i].transpose();
	}

	viewer.data().set_colors(_face_colors_C);

	return true;
}

bool Mesh::select_contiguous_face(igl::opengl::glfw::Viewer& viewer, int fid) {
	// For a face to be contiguous with a patch, the face and the patch much share at least one edge
	// The best approach I can come with given the structures at hand is to find all common vertices between the patch and face, 
	// then see if any have an edge between them by checking the adjacency matrix
	if (_selection_patch == nullptr) {
		std::shared_ptr<Mesh> mesh = dynamic_cast<MeshInstancer*>(this)->shared_from_this();
		_selection_patch = Patch::instantiate(mesh);
	}

	const std::set<Eigen::DenseIndex>& vids = _selection_patch->vids();

	bool contiguous = false;

	if (vids.size() == 0) {
		contiguous = true;
	} else {
		std::vector<Eigen::DenseIndex> shared_vertices;
		for (Eigen::DenseIndex i = 0; i < _F.cols(); ++i) {
			if (vids.count(_F(fid, i)) > 0) {
				shared_vertices.push_back(_F(fid, i));
			}
		}

		for (Eigen::DenseIndex i = 0; i < shared_vertices.size(); ++i) {
			for (Eigen::DenseIndex j = 0; j < shared_vertices.size(); ++j) {
				if (i == j) {
					continue;
				}

				if (_adj.coeff(shared_vertices[i], shared_vertices[j]) != 0) {
					contiguous = true;
					break;
				}
			}

			if (contiguous) {
				break;
			}
		}
	}

	if (contiguous) {
		std::set<Eigen::DenseIndex> fids = { fid };
		select_faces_with_color(viewer, fids);
	}

	return contiguous;
}

bool Mesh::select_patch(igl::opengl::glfw::Viewer& viewer, std::shared_ptr<Patch> patch, Eigen::Vector4d color) {
	_displayed_patches.push_back(std::pair<std::shared_ptr<Patch>, Eigen::Vector4d>(patch, color));

	bool refreshed = refresh_colormap();

	viewer.data().set_colors(_face_colors);

	return refreshed;
}

bool Mesh::select_vertex(igl::opengl::glfw::Viewer& viewer, Eigen::DenseIndex vid, Eigen::Vector3d color) {
	_displayed_points.conservativeResize(_displayed_points.rows() + 1, _displayed_points.cols());

	_displayed_points.row(_displayed_points.rows() - 1) << _V.block<1, 3>(vid, 0), color.transpose();

	viewer.data().points.resize(0, 0);

	for (Eigen::DenseIndex i = 0; i < _displayed_points.rows(); ++i) {
		viewer.data().add_points(_displayed_points.block<1, 3>(i, 0), _displayed_points.block<1, 3>(i, 3));
	}

	return true;
}

bool Mesh::select_curve_cover(igl::opengl::glfw::Viewer& viewer, Eigen::DenseIndex fid) {
	if (fid < _F.rows() || fid >= _FC.rows()) {
		return false;
	}

	int cIndex = -1;

	for (int i = 0; i < _curve_face_indices.size(); ++i) {
		if (fid < _curve_face_indices[i].first) {
			break;
		}

		++cIndex;
	}

	if (cIndex < 0) {
		return false;
	}

	auto fids = _curve_face_indices[cIndex].second->fids();

	return select_faces_with_color(viewer, fids);
}

bool Mesh::deselect_all(igl::opengl::glfw::Viewer& viewer) {
	_face_colors = _base_colors;

	_displayed_patches.clear();
	_displayed_points.resize(0, 6);

	viewer.data().points.resize(0, 0);

	_VC = _V;
	_FC = _F;
	_face_colors_C = _face_colors;
	_curve_face_indices.clear();

	viewer.data().clear();
	viewer.data().set_mesh(_VC.block(0, 0, _VC.rows(), 3), _FC);
	viewer.data().set_colors(_face_colors_C);

	return true;
}

std::shared_ptr<Patch> Mesh::emit_selected_as_patch() {
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> dist(0.4, 1);

	auto ret = _selection_patch;

	if (_selection_patch != nullptr) {
		Eigen::Vector4d color = { dist(e2), dist(e2), dist(e2), 1.0 };
		_displayed_patches.push_back(std::pair<std::shared_ptr<Patch>, Eigen::Vector4d>(_selection_patch, color));
	}

	std::shared_ptr<Mesh> mesh = dynamic_cast<MeshInstancer*>(this)->shared_from_this();
	_selection_patch = Patch::instantiate(mesh);

	refresh_colormap();

	return ret;
}

bool Mesh::match_active_selection(igl::opengl::glfw::Viewer& viewer, const std::shared_ptr<ShapeSignature> signature, double t_scale) {
	const Eigen::MatrixXd& sig_values = signature->get_signature_values();

	unsigned int rows = sig_values.rows();
	unsigned int cols = std::max(1.0, std::floor(sig_values.cols() * t_scale));
	Eigen::MatrixXd features = sig_values.block(0, 0, rows, cols);

	Eigen::MatrixXd matches;

	std::shared_ptr<Patch> user_selection = emit_selected_as_patch();
	Eigen::DenseIndex feature_vid;
	std::vector<Eigen::DenseIndex> feature_matches;
	std::vector<std::shared_ptr<Patch>> similar = user_selection->find_similar(signature, feature_vid, feature_matches);

	if (similar.size() == 0) {
		return false;
	}

	// Visualize matches
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> dist(0.4, 1);

	unsigned int patch_nbr = 0;
	for (auto it = similar.begin(); it != similar.end(); ++it) {
		Eigen::Vector4d color = { 0.0, 0.0, dist(e2), 1.0 };

		_displayed_patches.push_back(std::pair<std::shared_ptr<Patch>, Eigen::Vector4d>(*it, color));

		Eigen::DenseIndex patch_center_vid = feature_matches[patch_nbr++];

		viewer.data().add_points(_V.block(patch_center_vid, 0, 1, 3), color.transpose());
	}

	Eigen::Vector3d feature_color = { 0.2, 0.8, 0.2 };
	viewer.data().add_points(_V.block(feature_vid, 0, 1, 3), feature_color.transpose());
	
	refresh_colormap();

	return true;
}

// TODO: This needs to be refactored as a separate mesh to be display -- does libigl support multi-mesh rendering?? I believe there's a pending PR.
bool Mesh::show_patch_map(igl::opengl::glfw::Viewer& viewer, const std::shared_ptr<ShapeSignature> signature) {
	//std::shared_ptr<Patch> user_selection = emit_selected_as_patch();
	Eigen::VectorXi fids = Eigen::VectorXi::LinSpaced(_F.rows(), 0, _F.rows() - 1);
	std::shared_ptr<Patch> user_selection = Patch::instantiate(dynamic_cast<MeshInstancer*>(this)->shared_from_this(), fids);

	std::vector<Eigen::Vector3d> shared;

	Eigen::DenseIndex centroid_vid = user_selection->get_centroid_vid_on_origin_mesh();

	if (centroid_vid < 0) {
		return false;
	}

	std::shared_ptr<DiscreteExponentialMap> dem = user_selection->discrete_exponential_map(centroid_vid);

	std::stringstream ss; ss << resource_dir() << "//matlab//" << name().substr(0, name().find_first_of('.')) << "_dem.m";
	dem->to_matlab(ss.str());

	Eigen::MatrixXd V_dem = dem->get_3d_vertices();
	Eigen::MatrixXi F_dem = dem->get_reindexed_faces();

	F_dem = F_dem.array() + _V.rows();

	// In case of nD-vertex data
	V_dem.conservativeResize(V_dem.rows(), _V.cols());
	V_dem = V_dem.rowwise() + _V.row(centroid_vid);

	Eigen::MatrixXd V_all(_V.rows() + V_dem.rows(), _V.cols());
	Eigen::MatrixXi F_all(_F.rows() + F_dem.rows(), _F.cols());;
	Eigen::MatrixXd C_all(_face_colors.rows() + F_dem.rows(), _face_colors.cols());

	V_all << _V, V_dem;
	F_all << _F, F_dem;
	C_all << _face_colors, Eigen::MatrixXd::Ones(F_dem.rows(), _face_colors.cols());

	viewer.data().clear();
	viewer.data().set_mesh(V_all.block(0, 0, V_all.rows(), 3), F_all);
	viewer.data().set_colors(C_all);

	return true;
}

bool Mesh::show_patch_map(igl::opengl::glfw::Viewer& viewer, bool draw_points, const std::shared_ptr<SurfaceStroke> stroke, std::shared_ptr<ShapeSignature> signature) {
	//std::shared_ptr<Patch> user_selection = stroke->cover_patch();
	//std::shared_ptr<DiscreteExponentialMap> dem = stroke->cover_patch_local_dem(user_selection);

	if (stroke->blade_points().size() <= 0) {
		return false;
	}

	std::vector<CurveUnrolling> unrolled;
	unrolled.emplace_back(CurveUnrolling(stroke));
	
	// Pick a random face, then an arbitrary vid in the face
	std::random_device r;
	std::default_random_engine e1(r());
	std::uniform_int_distribution<int> uniform_dist(0, stroke->origin_mesh()->faces().rows() - 1);
	Eigen::DenseIndex fid = static_cast<Eigen::DenseIndex>(uniform_dist(e1));

	unrolled.emplace_back(CurveUnrolling(stroke->origin_mesh(), unrolled[0], fid));

	Eigen::MatrixXd V = _V;
	Eigen::MatrixXi F = _F;
	auto face_colors = _face_colors;

	//std::vector<Eigen::Vector3d> shared;
	for (auto cu = unrolled.begin(); cu != unrolled.end(); ++cu) {
		Eigen::DenseIndex centroid_vid = cu->vid_map().at(0);

		if (centroid_vid < 0) {
			return false;
		}

		Eigen::MatrixXd V_dem = cu->vertices();
		Eigen::MatrixXi F_dem = cu->faces();
		Eigen::MatrixXd C_dem = Eigen::MatrixXd::Ones(F_dem.rows(), face_colors.cols());

		F_dem = F_dem.array() + V.rows();

		Eigen::Vector3d normal = cu->frame().col(2);
		Eigen::Vector3d tangent = cu->frame().col(0);
		Eigen::Vector3d bitangent = cu->frame().col(1);
		Eigen::Vector3d pt = V.row(centroid_vid).block<1, 3>(0, 0).transpose() + normal;

		// In case of nD-vertex data
		Eigen::VectorXd nd_pt = Eigen::VectorXd::Zero(V.cols());
		nd_pt.topRows(pt.size()) << pt;

		V_dem.conservativeResize(V_dem.rows(), V.cols());
		V_dem = V_dem.rowwise() + nd_pt.transpose();

		Eigen::MatrixXd V_all(V.rows() + V_dem.rows(), V.cols());
		Eigen::MatrixXi F_all(F.rows() + F_dem.rows(), F.cols());;
		Eigen::MatrixXd C_all(face_colors.rows() + F_dem.rows(), face_colors.cols());

		V_all << V, V_dem;
		F_all << F, F_dem;
		C_all << face_colors, C_dem;

		V = V_all;
		F = F_all;
		face_colors = C_all;

		// TODO: There is a bug where the appended faces draw vertices well outside of the vertex matrix
		//		 The face matrix needs to be reindexed
		// CLUE: It seems to be related to patch connectivity -- a patch must be fully connected in order to extra a DEM from it, or else it should be represented by two patches (disjoint shatter)
		assert(F_all.maxCoeff() < V_all.rows());

		viewer.data().clear();

		viewer.data().set_mesh(V_all.block(0, 0, V_all.rows(), 3), F_all);
		viewer.data().set_colors(C_all);

		/*viewer.data().add_edges(pt.transpose(), (pt + normal).transpose(), Eigen::RowVector3d(1.0, 0.0, 0.0));
		viewer.data().add_edges(pt.transpose(), (pt + tangent).transpose(), Eigen::RowVector3d(0.0, 1.0, 0.0));
		viewer.data().add_edges(pt.transpose(), (pt + bitangent).transpose(), Eigen::RowVector3d(0.0, 0.0, 1.0));*/

		cu->unrolled_stroke()->display(viewer, draw_points, cu->parameterized_mesh(), false, nd_pt.topRows<3>());
	}

	return true;
}

bool Mesh::add_geometry(igl::opengl::glfw::Viewer& viewer, Eigen::MatrixXd& V, Eigen::MatrixXi& F, std::shared_ptr<SurfaceStroke> stroke) {
	Eigen::MatrixXd C_dem = Eigen::MatrixXd::Ones(F.rows(), _face_colors_C.cols());

	F = F.array() + _VC.rows();

	// In case of nD-vertex data
	V.conservativeResize(V.rows(), _VC.cols());
	//V_dem = V_dem.rowwise() + nd_pt.transpose();

	Eigen::MatrixXd V_all(_VC.rows() + V.rows(), _VC.cols());
	Eigen::MatrixXi F_all(_FC.rows() + F.rows(), _FC.cols());;
	Eigen::MatrixXd C_all(_face_colors_C.rows() + F.rows(), _face_colors_C.cols());

	if (stroke != nullptr) {
		_curve_face_indices.push_back(std::make_pair(_FC.rows(), stroke));
	}

	V_all << _VC, V;
	F_all << _FC, F;
	C_all << _face_colors_C, C_dem;

	_VC = V_all;
	_FC = F_all;
	_face_colors_C = C_all;

	// TODO: There is a bug where the appended faces draw vertices well outside of the vertex matrix
	//		 The face matrix needs to be reindexed
	// CLUE: It seems to be related to patch connectivity -- a patch must be fully connected in order to extra a DEM from it, or else it should be represented by two patches (disjoint shatter)
	assert(F_all.maxCoeff() < V_all.rows());
	assert(C_all.rows() == F_all.rows());

	viewer.data().clear();
	viewer.data().set_mesh(V_all.block(0, 0, V_all.rows(), 3), F_all);
	viewer.data().set_colors(C_all);
	
	return true;
}

std::vector<std::string> Mesh::calc_obj_vertex_colors() {
	std::vector<std::vector<std::array<unsigned char, 3>>> VC;
	VC.resize(_V.rows());

	unsigned int height = std::get<0>(_color_texture)->cols();
	unsigned int width = std::get<0>(_color_texture)->rows();

	for (unsigned int i = 0; i < _FTC.rows(); ++i) {
		for (unsigned int j = 0; j < _FTC.cols(); ++j) {
			// Clamp UV [0,1]
			double u = std::min(1.0, std::max(0.0, _TC(_FTC(i, j), 0)));
			double v = std::min(1.0, std::max(0.0, _TC(_FTC(i, j), 1)));

			unsigned int imgX = static_cast<unsigned int>(u * (width - 1));
			unsigned int imgY = static_cast<unsigned int>(v * (height - 1));

			std::array<unsigned char, 3> color{
				(*std::get<0>(_color_texture))(imgY, imgX),
				(*std::get<1>(_color_texture))(imgY, imgX),
				(*std::get<2>(_color_texture))(imgY, imgX)
			};

			VC[_F(i, j)].push_back(color);
		}
	}

	std::vector<std::array<unsigned char, 3>> bVC;

	for (auto it = VC.begin(); it != VC.end(); ++it) {
		std::array<unsigned char, 3> bColor{ 0.0, 0.0, 0.0 };

		for (auto itt = it->begin(); itt != it->end(); ++itt) {
			for (unsigned int k = 0; k < itt->size(); ++k) {
				bColor[k] += (*itt)[k] / itt->size();
			}
		}

		bVC.push_back(bColor);
	}

	std::vector<std::string> vertex_color_entries;

	for (unsigned int i = 0; i < bVC.size(); ++i) {
		std::stringstream ss;
		ss.precision(6);
		ss << "v ";

		for (unsigned int j = 0; j < _V.cols(); ++j) {
			ss << std::fixed << _V(i, j) << " ";
		}

		for (unsigned int k = 0; k < bVC[i].size(); ++k) {
			ss << std::fixed << static_cast<double>(bVC[i][k]) / 255.0 << " ";
		}
		ss << std::endl;

		vertex_color_entries.push_back(ss.str());
	}

	return vertex_color_entries;
}

bool Mesh::refresh_colormap() {
	_face_colors = _base_colors;
	
	for (auto it = _displayed_patches.cbegin(); it != _displayed_patches.cend(); ++it) {
		const std::set<Eigen::DenseIndex>& fids = it->first->fids();

		for (auto itt = fids.cbegin(); itt != fids.cend(); ++itt) {
			_face_colors.row(*itt) << it->second.block<3, 1>(0, 0).transpose();
		}
	}

	_face_colors_C.topRows(_face_colors.rows()) << _face_colors;

	return true;
}

bool Mesh::mean_curvature(Eigen::MatrixXd& MC) {
	Eigen::SparseMatrix<double> L;
	Eigen::MatrixXd D;
	igl::cotmatrix(_V, _F, L);

	D = L * _V;
	MC = D.rowwise().norm();

	return true;
}

bool Mesh::unproject_onto_mesh(const Eigen::Vector2f& screen_point, const Eigen::Matrix4f& model, const Eigen::Matrix4f& proj, const Eigen::Vector4f& viewport, int & fid, Eigen::Vector3f& bc, bool include_extra) {
	if (include_extra) {
		Eigen::MatrixXd V_block = _VC.block(0, 0, _VC.rows(), 3);

		return igl::unproject_onto_mesh(screen_point, model, proj, viewport, V_block, _FC, fid, bc);
	} else {
		Eigen::MatrixXd V_block = _V.block(0, 0, _V.rows(), 3);

		return igl::unproject_onto_mesh(screen_point, model, proj, viewport, V_block, _F, fid, bc);
	}
}

std::shared_ptr<DiscreteExponentialMap> Mesh::arap_parameterization(Eigen::DenseIndex center) {
	// This was pulled directly from the libigl tutorial for ARAP parameterization
	Eigen::MatrixXd vertex_uv;

	if (_V.rows() >= 3) {
		Eigen::MatrixXd initial_guess;

		// Compute the initial solution for ARAP (harmonic parametrization)
		std::vector<Eigen::DenseIndex> loop;
		igl::boundary_loop(_F, loop);

		Eigen::VectorXi bnd;
		bnd.resize(loop.size());
		for (size_t i = 0; i < loop.size(); ++i)
			bnd(i) = loop[i];

		Eigen::MatrixXd bnd_uv;
		igl::map_vertices_to_circle(_V, bnd, bnd_uv);

		igl::harmonic(_V, _F, bnd, bnd_uv, 1, initial_guess);

		// Add dynamic regularization to avoid to specify boundary conditions
		igl::ARAPData arap_data;
		arap_data.with_dynamics = true;
		Eigen::VectorXi b = Eigen::VectorXi::Zero(0);
		Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(0, 0);

		// Initialize ARAP
		arap_data.max_iter = 100;
		// 2 means that we're going to *solve* in 2d
		arap_precomputation(_V, _F, 2, b, arap_data);

		// Solve arap using the harmonic map as initial guess
		vertex_uv = initial_guess;

		igl::arap_solve(bc, arap_data, vertex_uv);
	}

	// Pack points for map representation
	std::map<Eigen::DenseIndex, Eigen::Vector2d> DEM_points;
	for (Eigen::DenseIndex i = 0; i < vertex_uv.rows(); ++i) {
		// Subtract center vid UV
		DEM_points.insert(std::pair<Eigen::DenseIndex, Eigen::Vector2d>(i, vertex_uv.row(i) - vertex_uv.row(center)));
	}

	// Scale UV to make it representative of edge distances
	if (DEM_points.size() > 0) {
		auto ring = one_ring(center);

		double scale = 0.0;
		for (auto vid : ring) {
			if (vid == center) {
				continue;
			}

			scale += ((_V.block<1, 3>(vid, 0) - _V.block<1, 3>(center, 0)).norm() / _TC.row(vid).norm()) / static_cast<double>(ring.size());
		}

		if (scale > std::numeric_limits<double>::epsilon()) {
			for (auto it = DEM_points.begin(); it != DEM_points.end(); ++it) {
				it->second = scale * it->second;
			}
		}
	} else {
		DEM_points.insert(std::pair<Eigen::DenseIndex, Eigen::Vector2d>(center, Eigen::Vector2d::Zero()));
	}

	// Create a TBN for the parameterization
	Eigen::Matrix3d TBN = basis_from_plane_normal(_N.row(center));

	std::shared_ptr<Patch> mesh_as_patch = Patch::instantiate(dynamic_cast<MeshInstancer*>(this)->shared_from_this(), Eigen::VectorXi::LinSpaced(_F.rows(), 0, _F.rows() - 1));
	
	//return std::make_shared<DiscreteExponentialMap>(center, TBN, DEM_points, dynamic_cast<MeshInstancer*>(this)->shared_from_this());
	return std::make_shared<DiscreteExponentialMap>(center, TBN, DEM_points, mesh_as_patch);
}

std::vector<Eigen::DenseIndex> Mesh::one_ring(Eigen::DenseIndex center_vid) {
	std::vector<Eigen::DenseIndex> ring;

	if (center_vid < 0 || center_vid >= _V.rows()) {
		return ring;
	}

	ring.push_back(center_vid);

	for (Eigen::SparseMatrix<int>::InnerIterator it(_adj, static_cast<int>(center_vid)); it; ++it) {
		Eigen::DenseIndex neighbor = it.row();   // neighbor vid

		ring.push_back(neighbor);
	}

	return ring;
}

Eigen::Matrix3d Mesh::basis_from_plane_normal(const Eigen::Vector3d plane_normal) {
	// Set up basis
	Eigen::Vector3d tangent = plane_normal.cross(Eigen::Vector3d::UnitX()).normalized();

	if (tangent.isZero()) {
		// Just in case we're unlucky and plane_normal is (anti)parallel to UnitX(), then it can't also be (anti)parellel UnitY()
		tangent = plane_normal.cross(Eigen::Vector3d::UnitY()).normalized();
	}

	Eigen::Vector3d bitangent = plane_normal.cross(tangent).normalized();

	Eigen::Matrix3d TBN;
	TBN << tangent, bitangent, plane_normal;

	return TBN;
}

Eigen::DenseIndex Mesh::vid_to_origin_mesh(Eigen::DenseIndex vid) {
	return vid;
}

Eigen::DenseIndex Mesh::fid_to_origin_mesh(Eigen::DenseIndex fid) {
	return fid;
}