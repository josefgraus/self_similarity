#include "pt_unfolding.h"

#include <queue>

#include <mathtoolbox/classical-mds.hpp>

#include <algorithms/shortest_path.h>

using namespace shortest_path;

PTUnfolding::PTUnfolding(std::shared_ptr<Patch> patch, double ball_radius) {
	// Parallel Transport Unfolding https://arxiv.org/pdf/1806.09039.pdf
	// Used for 3D->2D parameterization

	std::map<Eigen::DenseIndex, Eigen::DenseIndex> vert_indices;
	Eigen::MatrixXd D = parallel_transport_dijkstra(patch, vert_indices);

	Eigen::MatrixXd V = mathtoolbox::ComputeClassicalMds(D, 2);
}

PTUnfolding::~PTUnfolding() {
}

Eigen::MatrixXi PTUnfolding::get_reindexed_faces() const {
	throw std::exception("Not Implemented!");
}

Eigen::MatrixXd PTUnfolding::parallel_transport_dijkstra(std::shared_ptr<Patch> patch, std::map<Eigen::DenseIndex, Eigen::DenseIndex>& vert_indices) {
	const std::shared_ptr<Mesh> mesh = patch->origin_mesh();
	const std::set<Eigen::DenseIndex> vids = patch->vids();
	const Eigen::MatrixXd& V = patch->vertices();
	const Eigen::MatrixXd& N = patch->vertex_normals();
	const int n = vids.size();

	Eigen::MatrixXd D = Eigen::MatrixXd::Constant(n, n, std::numeric_limits<double>::infinity());

	if (n <= 0) {
		return D;
	}

	const Eigen::SparseMatrix<int>& adj = patch->origin_mesh()->adjacency_matrix();
	std::priority_queue<std::shared_ptr<DjikstraVertexNode>, std::vector<std::shared_ptr<DjikstraVertexNode>>, DjikstraDist> q;

	vert_indices.clear();

	std::map<Eigen::DenseIndex, Eigen::Matrix3d> R;
	std::map<Eigen::DenseIndex, Eigen::Matrix3d> T;

	int vIndex = 0;
	for (auto vid : vids) {
		vert_indices.insert(std::make_pair(vid, vIndex++));
		T.insert(std::make_pair(vid, basis_from_plane_normal(N.row(vid).leftCols<3>().transpose())));
	}

	std::set<Eigen::DenseIndex>::iterator vid = vids.begin();
	for (Eigen::DenseIndex i = 0; i < n; i++, vid++) {
		
		std::map<Eigen::DenseIndex, std::shared_ptr<DjikstraVertexNode>> nodes;
		Eigen::DenseIndex source = *vids.begin();
		auto source_node = std::make_shared<DjikstraVertexNode>(nullptr, 0.0, source);
		q.push(source_node);
		nodes.insert(std::pair<Eigen::DenseIndex, std::shared_ptr<DjikstraVertexNode>>(source, source_node));

		R.insert(std::make_pair(*vid, Eigen::Matrix3d::Identity()));

		std::map<Eigen::DenseIndex, Eigen::Vector3d> v;
		v.insert(std::make_pair(*vid, Eigen::Vector3d::Zero()));

		std::map<Eigen::DenseIndex, double> geo_dist;

		while (!q.empty()) {
			std::vector<std::shared_ptr<DjikstraVertexNode>> stored;

			auto xr = q.top();
			q.pop();

			while (!q.empty()) {
				stored.push_back(q.top());
				q.pop();
			}

			auto s = xr->_prev;
			
			// Presuming most of my patch problems are going to be relatively small, so going with Eigen JacobiSVD
			Eigen::Matrix3d Tr = T.at(xr->_vid);
			Eigen::Matrix3d Ts = T.at(s->_vid);
			Eigen::MatrixXd m = Ts.transpose() * Tr;	// Tq^t * Tr from alg 2 in paper
			Eigen::JacobiSVD<Eigen::MatrixXd> svd(m);
			R[xr->_vid] = R.at(s->_vid) * svd.matrixU() * svd.matrixV().transpose();
			v[xr->_vid] = v.at(s->_vid) + R.at(s->_vid) * Ts.transpose() * (V.row(xr->_vid) - V.row(s->_vid)).transpose();
			geo_dist.insert(std::make_pair(xr->_vid, v[xr->_vid].norm()));

			for (Eigen::SparseMatrix<int>::InnerIterator it(adj, static_cast<int>(xr->_vid)); it; ++it) {
				Eigen::DenseIndex neighbor = it.row();   // neighbor vid

				if (patch->vids().count(neighbor) == 0) {
					continue;
				}

				double dist = (V.row(neighbor).block<1, 3>(0, 0) - V.row(xr->_vid).block<1, 3>(0, 0)).norm() + xr->_dist;
				auto node = nodes.find(neighbor);

				if (node == nodes.cend()) {
					auto neighbor_node = std::make_shared<DjikstraVertexNode>(xr, dist, neighbor);

					q.push(neighbor_node);
					nodes.insert(std::pair<Eigen::DenseIndex, std::shared_ptr<DjikstraVertexNode>>(neighbor, neighbor_node));
				}
				else {
					if (dist < node->second->_dist) {
						node->second->_dist = dist;
						node->second->_prev = xr;
					}
				}
			}

			// Changing values in a priority_queue invalidates the queue, so update it
			std::priority_queue<std::shared_ptr<DjikstraVertexNode>, std::vector<std::shared_ptr<DjikstraVertexNode>>, DjikstraDist> new_q;
			for (unsigned int i = 0; i < stored.size(); ++i) {
				q.push(stored[i]);
			}
		}

		// Populate row of D
		for (auto dist : geo_dist) {
			D(i, vert_indices[dist.first]) = dist.second;
		}
	}

	// Symmetrize distance matrix D
	D = (D + D.transpose()) / 2.0;

	assert(D.maxCoeff() < std::numeric_limits<double>::infinity());

	// q is now empty, and D contains all pairwise geodesic distances between patch vertices
	return D;
}