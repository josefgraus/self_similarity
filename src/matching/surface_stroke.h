#ifndef SURFACE_STROKE_H_
#define SURFACE_STROKE_H_

#include <memory>

#include <matching/geodesic_fan.h>
#include <geometry/mesh.h>
#include <matching/self_similarity/self_similarity_map.h>

class CubicBezierSegment {
	public:
		CubicBezierSegment(const Eigen::MatrixXd& pts);

		double piecewise_length() const { return _piecewise_length; }

		Eigen::Vector3d pt_at_t(double t) const;
		Eigen::Vector3d pt_at_dist(double dist) const;

	protected:
		double _piecewise_length;
		Eigen::MatrixXd _pts;
};

class SurfaceStroke: public GeodesicFanBlade {
	public:
		friend class Mesh;
		friend class CurveUnrolling;

		static std::shared_ptr<SurfaceStroke> instantiate(std::shared_ptr<Mesh> mesh);
		static std::shared_ptr<SurfaceStroke> instantiate(std::shared_ptr<Mesh> mesh, const SurfaceStroke& other, Eigen::DenseIndex centroid_vid);

		std::shared_ptr<SurfaceStroke> clone();

		std::set<Eigen::DenseIndex> fids() const { return _fids; }

		void add_curve_point(Eigen::DenseIndex fid, Eigen::Vector3d barycentric_coord);
		void transform(double x, double y, double radians, int blade_point_index = -1);

		double compare(const SurfaceStroke& other, std::shared_ptr<ShapeSignature> sig);
		Eigen::VectorXd per_point_diff(const SurfaceStroke& other, std::shared_ptr<ShapeSignature> sig);
		bool is_disjoint_from(std::shared_ptr<SurfaceStroke> other) const;

		Eigen::MatrixXd SurfaceStroke::to_world() const;
		void display(igl::opengl::glfw::Viewer& viewer, bool draw_points, std::shared_ptr<Geometry> geo = nullptr, bool clear = true, Eigen::Vector3d offset = Eigen::Vector3d::Zero(), Eigen::Vector3d color = Eigen::Vector3d::UnitZ(), std::shared_ptr<SurfaceStroke> this_shared = nullptr);

		std::shared_ptr<Mesh> origin_mesh() const { return _mesh; }
		void origin_mesh(std::shared_ptr<Mesh> mesh) { _mesh = mesh; }
		const std::set<Eigen::DenseIndex> face_indices() const { return _fids; }
		const std::vector<BarycentricCoord>& blade_points() const { return _curve_points; }
		BarycentricCoord curve_center(double* max_dist_from_center = nullptr, std::size_t* curve_cindex = nullptr) const;
		virtual Eigen::MatrixXd parameterized_space_points_2d(std::shared_ptr<Patch>* cover_patch = nullptr, std::shared_ptr<DiscreteExponentialMap>* origin_map = nullptr, Eigen::DenseIndex center_vid = -1) const;
		virtual SignatureTensor blade_values(double angle_step, std::shared_ptr<ShapeSignature> sig, std::shared_ptr<DiscreteExponentialMap>* dem = nullptr);

		bool to_matlab(std::string matlab_file_path) const;

	protected:
		SurfaceStroke(std::shared_ptr<Mesh> mesh);
		SurfaceStroke(std::shared_ptr<Mesh> mesh, const SurfaceStroke& other, Eigen::DenseIndex centroid_vid);

		std::shared_ptr<Patch> cover_patch() const;
		std::shared_ptr<DiscreteExponentialMap> cover_patch_local_dem(std::shared_ptr<Patch> cover_patch);

		bool matlab_template(std::string matlab_file_path, std::shared_ptr<DiscreteExponentialMap> dem, Eigen::MatrixXd curve) const;

		std::shared_ptr<Mesh> _mesh;
		std::vector<BarycentricCoord> _curve_points;
		std::set<Eigen::DenseIndex> _fids;
};

#endif