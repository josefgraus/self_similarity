#ifndef SPECTRAL_CLUSTERING_H_
#define SPECTRAL_CLUSTERING_H_

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <geometry/geometry.h>
#include <geometry/component.h>

// Class that implements Segmentation of 3D Meshes through Spectral Clustering
// https://dl.acm.org/citation.cfm?id=1026053
class SpectralClustering {
	public:
		SpectralClustering(std::shared_ptr<Geometry> geometry, unsigned int k, double delta = 0.03, double eta = 0.15);
		~SpectralClustering();

		const Eigen::VectorXi& segment_by_face() const;
		//std::vector<std::shared_ptr<Component>> components();

	private:
		Eigen::MatrixXd affinity_matrix(std::shared_ptr<Geometry> geometry);
		inline double geodesic_distance(std::array<Eigen::DenseIndex, 2> edge_verts, const Eigen::VectorXi& face1, const Eigen::VectorXi& face2, const Eigen::MatrixXd& V) const;
		inline double angular_distance(const Eigen::VectorXi& face1, const Eigen::VectorXi& face2, const Eigen::MatrixXd& V) const;

		// k-means clustering
		Eigen::VectorXi k_means_lloyds(const Eigen::VectorXi& guess, const Eigen::MatrixXd& V, unsigned int k);
		Eigen::VectorXi initial_guess(const Eigen::MatrixXd& Q, unsigned int k);

		double _delta;
		double _eta;

		std::vector<std::shared_ptr<Component>> _components;
		Eigen::VectorXi _segment_by_face;
};

#endif