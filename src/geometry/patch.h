#ifndef PATCH_H_
#define PATCH_H_

#include <vector>
#include <memory>
#include <set>
#include <unordered_map>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <matching/parameterization/discrete_exponential_map.h>
#include <geometry/mesh.h>
#include <shape_signatures/shape_signature.h>

class Patch: public Geometry {
	public:
		~Patch();

		static std::shared_ptr<Patch> instantiate(std::shared_ptr<Mesh> origin);
		static std::shared_ptr<Patch> instantiate(std::shared_ptr<Mesh> origin, Eigen::DenseIndex center_vid);
		static std::shared_ptr<Patch> instantiate(std::shared_ptr<Mesh> origin, const Eigen::VectorXi& fids);
		static std::shared_ptr<Patch> instantiate(std::shared_ptr<Mesh> origin, Eigen::DenseIndex center_vid, double geodesic_radius);

		static std::shared_ptr<Patch> one_ring(std::shared_ptr<Mesh> mesh, Eigen::DenseIndex center_vid);
		virtual std::vector<Eigen::DenseIndex> one_ring(Eigen::DenseIndex center_vid);
		std::vector<Eigen::DenseIndex> valence_faces(Eigen::DenseIndex vid);

		std::shared_ptr<Mesh> origin_mesh() const { return _origin_mesh; }
		virtual const Eigen::MatrixXd& vertices() const;
		virtual const Eigen::MatrixXd& vertex_normals() const;
		virtual const Eigen::MatrixXi& faces() const;
		virtual const Eigen::SparseMatrix<int>& adjacency_matrix() const;
		virtual ColorTexture& color_texture() { return _origin_mesh->color_texture(); }
		virtual const trimesh::trimesh_t& halfedge();

		bool add(Eigen::DenseIndex fid);
		bool add(std::set<Eigen::DenseIndex> fids);
		bool remove(Eigen::DenseIndex fid); 
		const std::set<Eigen::DenseIndex>& vids() const { return _vids; }
		const std::set<Eigen::DenseIndex>& fids() const { return _fids; }
		bool contains(Eigen::DenseIndex fid);

		bool is_disjoint_from(std::shared_ptr<Patch> other) const;
		Eigen::DenseIndex get_centroid_vid() const;
		Eigen::DenseIndex get_centroid_vid_on_origin_mesh() const;
		Eigen::DenseIndex get_centroid_fid() const;
		Eigen::DenseIndex get_centroid_fid_on_origin_mesh() const;
		Eigen::DenseIndex nearest_vid(Eigen::VectorXd point) const;
		virtual Eigen::DenseIndex vid_to_origin_mesh(Eigen::DenseIndex patch_vid);
		virtual Eigen::DenseIndex vid_to_local_patch(Eigen::DenseIndex origin_vid);
		virtual Eigen::DenseIndex fid_to_origin_mesh(Eigen::DenseIndex fid);
		
		double get_geodesic_extent(Eigen::DenseIndex center_vid);
		std::vector<std::shared_ptr<Patch>> find_similar(const std::shared_ptr<ShapeSignature> signature, Eigen::DenseIndex& feature_vid, std::vector<Eigen::DenseIndex>& feature_matches);

		std::vector<std::shared_ptr<Patch>> shatter();
		std::shared_ptr<DiscreteExponentialMap> discrete_exponential_map(Eigen::DenseIndex center);

	protected:
		Patch(std::shared_ptr<Mesh> origin);
		Patch(std::shared_ptr<Mesh> origin_mesh, Eigen::DenseIndex vid);
		Patch(std::shared_ptr<Mesh> origin, const Eigen::VectorXi& fids);
		Patch(std::shared_ptr<Mesh> origin, Eigen::DenseIndex center_vid, double geodesic_radius);

		// Make local copies of only parts of the origin_mesh data that is used to construct this patch
		void reindex_submatrices();

		double automatic_threshold(const Eigen::MatrixXd& features);
		Eigen::DenseIndex approximate_patch_feature(const Eigen::MatrixXd& features, double threshold, Eigen::MatrixXd& matches);
		std::unordered_map<Eigen::DenseIndex, std::vector<Eigen::DenseIndex>> bipartite_threshold_matching(std::shared_ptr<Patch> other, const Eigen::MatrixXd& features, double threshold);

		std::shared_ptr<Mesh> _origin_mesh;
		std::set<Eigen::DenseIndex> _vids;
		std::set<Eigen::DenseIndex> _fids;

		// Local copies of vertices, faces, normals, and other patch data taken from _origin_mesh
		std::unordered_map<Eigen::DenseIndex, Eigen::DenseIndex> _remapping_vid;
		std::unordered_map<Eigen::DenseIndex, Eigen::DenseIndex> _remapping_fid;
		std::unordered_map<Eigen::DenseIndex, Eigen::DenseIndex> _remapping_vid_rev;
		std::unordered_map<Eigen::DenseIndex, Eigen::DenseIndex> _remapping_fid_rev;
		Eigen::MatrixXd _vertices;
		Eigen::MatrixXd _vertex_normals;
		Eigen::MatrixXi _faces;
		Eigen::SparseMatrix<int> _adj;
		
		trimesh::trimesh_t _halfedge_patch;
};

#endif