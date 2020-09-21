#ifndef DISCRETE_EXPONENTIAL_MAP_H_
#define DISCRETE_EXPONENTIAL_MAP_H_

#include <map>
#include <memory>
#include <vector>

#include <Eigen/Dense>

class Geometry;
class Patch;
class ShapeSignature;
class SurfaceStroke;

class DiscreteExponentialMap {
	public:
		DiscreteExponentialMap();
		DiscreteExponentialMap(std::shared_ptr<Patch> patch, Eigen::DenseIndex p_vid, const Eigen::MatrixXd* guide_curve = nullptr);
		DiscreteExponentialMap(const Eigen::DenseIndex center_vid, Eigen::Matrix3d& TBN, const std::map<Eigen::DenseIndex, Eigen::Vector2d>& vertices, std::shared_ptr<Patch> geometry);
		~DiscreteExponentialMap();

	    const Eigen::MatrixXi& get_raw_faces() const { return _faces; }
		Eigen::MatrixXi get_reindexed_faces() const;
		const std::map<Eigen::DenseIndex, Eigen::Vector2d>& get_raw_vertices() const { return _vertices; }

		Eigen::Vector3d get_normal() const;
		Eigen::Vector3d get_tangent() const;
		Eigen::Vector3d get_bitangent() const;
		Eigen::Matrix3d TBN() const { return _TBN; }

		Eigen::DenseIndex get_center_vid() const;
		Eigen::DenseIndex get_center_fid() const;
		double get_radius() const;
		Eigen::MatrixXd get_3d_vertices() const;

		Eigen::Vector2d interpolated_polar(Eigen::Vector3d barycentric_coords, const std::vector<Eigen::DenseIndex>& vids);
		Eigen::DenseIndex nearest_vertex_by_polar(const Eigen::Vector2d& polar_point);
		Eigen::VectorXd query_map_value(const Eigen::Vector2d& xy_point, std::shared_ptr<ShapeSignature> sig) const;
		Eigen::VectorXd query_map_value_polar(const Eigen::Vector2d& polar_point, std::shared_ptr<ShapeSignature> sig) const;

		bool to_matlab(std::string script_out_path);

	private:
		bool init(const Eigen::DenseIndex center_vid, Eigen::Matrix3d& TBN, const std::map<Eigen::DenseIndex, Eigen::Vector2d>& vertices, std::shared_ptr<Patch> geometry);

		std::shared_ptr<Patch> _geometry;		// pointer to original geometry	
		Eigen::Matrix3d _TBN;
		Eigen::Matrix3d _TBN_inv;
		std::map<Eigen::DenseIndex, Eigen::Vector2d> _vertices;
		Eigen::MatrixXi _faces;
		Eigen::DenseIndex _center_vid;
		Eigen::DenseIndex _center_fid;
		std::map<Eigen::DenseIndex, Eigen::DenseIndex> _fid_remap;
};

#endif