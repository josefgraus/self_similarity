#ifndef CURVE_UNROLLING_H_
#define CURVE_UNROLLING_H_

#include <memory>
#include <array>

#include <matching/surface_stroke.h>

struct LocalTriFace {
	public:
		// Counter clockwise
		enum class Edge : int {
			None = -1,
			AB = 0,
			BC = 1,
			CA = 2
		};

		LocalTriFace(Eigen::DenseIndex fid, Eigen::Vector3i ABC, std::array<std::shared_ptr<LocalTriFace>, 3> neighbors, std::shared_ptr<Eigen::Matrix3d> frame, std::shared_ptr<Mesh> mesh): _fid(fid), _ABC(ABC), _neighbors(neighbors), _frame(frame), _mesh(mesh) { }

		Eigen::DenseIndex _fid;
		std::array<std::shared_ptr<LocalTriFace>, 3> _neighbors;
		Eigen::Vector3i _ABC;
		std::shared_ptr<Eigen::Matrix3d> _frame;
		std::shared_ptr<Mesh> _mesh;
		
		Eigen::DenseIndex A() { return _ABC(0); }
		Eigen::DenseIndex B() { return _ABC(1); }
		Eigen::DenseIndex C() { return _ABC(2); }

		Eigen::Vector3d face_center(const Eigen::MatrixXd& V, Eigen::Vector3d origin = Eigen::Vector3d::Zero());
		Edge intersected_edge(const Eigen::MatrixXd& V, Eigen::DenseIndex mesh_fid, std::pair<Eigen::Vector3d, Eigen::Vector3d> segment, BarycentricCoord* intersection_point = nullptr, double v_offset = 0.0);

	private:
		LocalTriFace();
};

class CurveUnrolling {
	public:
		CurveUnrolling(std::shared_ptr<SurfaceStroke> curve);
		CurveUnrolling(std::shared_ptr<Mesh> mesh, const Eigen::MatrixXd& map_points, BarycentricCoord origin, int origin_index, std::shared_ptr<Eigen::Matrix3d> frame = nullptr, std::shared_ptr<Eigen::Matrix3d> prev_ABC = nullptr);
		CurveUnrolling(std::shared_ptr<Mesh> mesh, const CurveUnrolling& cu, Eigen::DenseIndex frame_fid);
		~CurveUnrolling();

		std::shared_ptr<CurveUnrolling> clone() const;

		void transform(double x, double y, double radians);

		void reframe(Eigen::DenseIndex blade_point_index = -1);

		std::shared_ptr<Mesh> parameterized_mesh() { return Mesh::instantiate(_V, _F); }
		std::shared_ptr<Mesh> origin_mesh() const { return _origin_mesh; }
		std::shared_ptr<SurfaceStroke> unrolled_stroke() const { return _unrolled_stroke; }
		std::shared_ptr<SurfaceStroke> unrolled_on_origin_mesh() const;

		Eigen::MatrixXd curve_points_2d();

		const Eigen::MatrixXd& vertices() const { return _V; }
		const std::map<Eigen::DenseIndex, Eigen::DenseIndex>& vid_map() const { return _vid_map; }
		
		const Eigen::MatrixXi& faces() const { return _F; }
		const std::map<Eigen::DenseIndex, Eigen::DenseIndex> fid_map() const { return _fid_map; }	// fid map goes from parameterization face back to mesh -- a mesh face can have *multiple* parameterized faces mapped to it!

		const Eigen::Matrix3d& frame() { return _frame; }

	protected:
		CurveUnrolling();

		void copy_to_init(std::shared_ptr<Mesh> mesh, const Eigen::MatrixXd& map_points, int curve_center_index, BarycentricCoord origin_bc, std::shared_ptr<Eigen::Matrix3d> frame = nullptr, std::shared_ptr<Eigen::Matrix3d> prev_ABC = nullptr);

		std::shared_ptr<SurfaceStroke> _curve;

		Eigen::MatrixXd _V;
		std::map<Eigen::DenseIndex, Eigen::DenseIndex> _vid_map;
		Eigen::MatrixXi _F;
		std::map<Eigen::DenseIndex, Eigen::DenseIndex> _fid_map;
		std::shared_ptr<LocalTriFace> _root;
		Eigen::Matrix3d _frame;
		std::shared_ptr<Mesh> _origin_mesh;
		std::shared_ptr<SurfaceStroke> _unrolled_stroke;

		Eigen::Vector3d bc_to_frame(Eigen::DenseIndex unroll_fid, BarycentricCoord mesh_bc);

		bool to_matlab(std::string script_out_path, Eigen::DenseIndex start_fid, std::pair<BarycentricCoord, Eigen::Vector3d>* segment = nullptr, std::shared_ptr<LocalTriFace> tri = nullptr);
};

#endif