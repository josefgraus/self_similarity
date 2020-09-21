#ifndef PT_UNFOLDING_H_
#define PT_UNFOLDING_H_

#include <map>

#include <Eigen/Dense>

#include <geometry/patch.h>

class PTUnfolding {
	public:
		PTUnfolding(std::shared_ptr<Patch> patch, double ball_radius);
		~PTUnfolding();

		const Eigen::MatrixXi& get_raw_faces() const { return _faces; }
		Eigen::MatrixXi get_reindexed_faces() const;
		const std::map<Eigen::DenseIndex, Eigen::Vector2d>& get_raw_vertices() const { return _vertices; }

	protected:
		Eigen::MatrixXd parallel_transport_dijkstra(std::shared_ptr<Patch> patch, std::map<Eigen::DenseIndex, Eigen::DenseIndex>& vert_indices);

		std::map<Eigen::DenseIndex, Eigen::Vector2d> _vertices;
		Eigen::MatrixXi _faces;
};

#endif