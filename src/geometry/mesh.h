#ifndef MESH_H_
#define MESH_H_

#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <exception>
#include <memory>
#include <unordered_map>
#include <set>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/colormap.h>

#include <geometry/geometry.h>
#include <shape_signatures/shape_signature.h>
#include <matching/parameterization/discrete_exponential_map.h>

typedef double Scalar;

class SurfaceStroke;

class Mesh: public Geometry {
	public:
		~Mesh();

		static std::shared_ptr<Mesh> instantiate(std::string obj_path);
		static std::shared_ptr<Mesh> instantiate(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

		virtual const Eigen::MatrixXd& vertices() const { return _V; }
		virtual const Eigen::MatrixXd& vertex_normals() const { return _VN; }
		virtual const Eigen::MatrixXd& vertex_uv() const { return _TC; }
		virtual const Eigen::MatrixXi& faces() const { return _F; }
		virtual const Eigen::MatrixXd& face_normals() const { return _FN; }
		virtual const Eigen::MatrixXi& faces_uv() const { return _FTC; }
		virtual const Eigen::SparseMatrix<int>& adjacency_matrix() const { return _adj; }
		virtual const Eigen::MatrixXi& tri_adjacency_matrix() const { return _tri_adj; }
		virtual ColorTexture& color_texture() { return _color_texture; } 
		virtual const trimesh::trimesh_t& halfedge();

		bool loaded() { return _loaded; }
		std::string name() { return _model_name; }
		std::string resource_dir() { return _data_dir; }
		std::string model_path() const { return _model_path; }

		bool load_texture(std::string texture_path);
		bool save_texture(std::string texture_path);

		void display(igl::opengl::glfw::Viewer& viewer);
		bool set_scalar_vertex_color(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& color_per_vertex, igl::ColorMapType cm = igl::COLOR_MAP_TYPE_INFERNO);
		bool set_scalar_vertex_color(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& color_per_vertex, double z_min, double z_max, igl::ColorMapType cm = igl::COLOR_MAP_TYPE_INFERNO);
		bool select_vid_with_point(igl::opengl::glfw::Viewer& viewer, Eigen::DenseIndex vid, Eigen::Vector3d color = { 0.0, 0.0, 1.0 });

		int closest_vertex_id(int fid, Eigen::Vector3f bc);
		bool select_faces_with_color(igl::opengl::glfw::Viewer& viewer, std::set<Eigen::DenseIndex> fids, Eigen::Vector3d color = { 0.2, 0.8, 0.2 });
		bool color_faces(igl::opengl::glfw::Viewer& viewer, std::vector<int> fids, std::vector<Eigen::Vector3d> colors);
		bool select_contiguous_face(igl::opengl::glfw::Viewer& viewer, int fid);
		bool select_patch(igl::opengl::glfw::Viewer& viewer, std::shared_ptr<Patch> patch, Eigen::Vector4d color);
		bool select_vertex(igl::opengl::glfw::Viewer& viewer, Eigen::DenseIndex vid, Eigen::Vector3d color);
		bool select_curve_cover(igl::opengl::glfw::Viewer& viewer, Eigen::DenseIndex fid);
		bool deselect_all(igl::opengl::glfw::Viewer& viewer);
		std::shared_ptr<Patch> emit_selected_as_patch();

		bool match_active_selection(igl::opengl::glfw::Viewer& viewer, const std::shared_ptr<ShapeSignature> signature, double t_scale);
		bool show_patch_map(igl::opengl::glfw::Viewer& viewer, const std::shared_ptr<ShapeSignature> signature);
		bool show_patch_map(igl::opengl::glfw::Viewer& viewer, bool draw_points, const std::shared_ptr<SurfaceStroke> stroke, std::shared_ptr<ShapeSignature> signature);
		bool add_geometry(igl::opengl::glfw::Viewer& viewer, Eigen::MatrixXd& V, Eigen::MatrixXi& F, std::shared_ptr<SurfaceStroke> stroke = nullptr);

		std::vector<std::string> calc_obj_vertex_colors();
		bool mean_curvature(Eigen::MatrixXd& MC);
		bool unproject_onto_mesh(const Eigen::Vector2f& screen_point, const Eigen::Matrix4f& model, const Eigen::Matrix4f& proj, const Eigen::Vector4f& viewport, int & fid, Eigen::Vector3f& bc, bool include_extra = false);

		virtual std::vector<Eigen::DenseIndex> one_ring(Eigen::DenseIndex center_vid);
//		virtual std::shared_ptr<DiscreteExponentialMap> discrete_exponential_map(Eigen::DenseIndex center);
		std::shared_ptr<DiscreteExponentialMap> arap_parameterization(Eigen::DenseIndex center);
		virtual Eigen::DenseIndex vid_to_origin_mesh(Eigen::DenseIndex vid);
		virtual Eigen::DenseIndex fid_to_origin_mesh(Eigen::DenseIndex fid);

		bool write_obj(std::string path);

	protected:
		Mesh(std::string obj_path);
		Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

		bool load_obj(std::string obj_path);
		Eigen::Matrix3d basis_from_plane_normal(const Eigen::Vector3d plane_normal);

		bool refresh_colormap();

		// Geometry and color
		Eigen::MatrixXd _V, _TC, _N;
		Eigen::MatrixXi _F, _FTC;
		Eigen::MatrixXd _VN, _FN;
		std::tuple<std::shared_ptr<ColorChannel>, std::shared_ptr<ColorChannel>, std::shared_ptr<ColorChannel>, std::shared_ptr<ColorChannel>> _color_texture;
		trimesh::trimesh_t halfedge_mesh;

		Eigen::SparseMatrix<int> _adj;
		Eigen::MatrixXi _tri_adj;

		bool _loaded;
		std::string _model_name;
		std::string _data_dir;
		std::string _model_path;

		Eigen::MatrixXd _base_colors;
		Eigen::MatrixXd _face_colors;
		std::shared_ptr<Patch> _selection_patch;	// Patch containing all "actively" selected faces on the mesh
		std::vector<std::pair<const std::shared_ptr<Patch>, Eigen::Vector4d>> _displayed_patches;	// A list of committed patches for display purpose only
		Eigen::MatrixXd _displayed_points;
		Eigen::MatrixXd _VC;
		Eigen::MatrixXi _FC;
		Eigen::MatrixXd _face_colors_C;

		// Juryrigged curve selection
		std::vector<std::pair<Eigen::DenseIndex, std::shared_ptr<SurfaceStroke>>> _curve_face_indices;
};

#endif