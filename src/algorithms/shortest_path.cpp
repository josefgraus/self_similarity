#include "shortest_path.h"

#include <iostream>
#include <memory>
#include <vector>
#include <queue>
#include <set>

#include <Eigen/Dense>

#include <geometry/patch.h>

namespace shortest_path {
	std::map<Eigen::DenseIndex, std::shared_ptr<DjikstraVertexNode>> djikstras_algorithm(std::shared_ptr<Patch> patch, Eigen::DenseIndex source, std::set<Eigen::DenseIndex> exclude) {
		std::priority_queue<std::shared_ptr<DjikstraVertexNode>, std::vector<std::shared_ptr<DjikstraVertexNode>>, DjikstraDist> q;
		std::map<Eigen::DenseIndex, std::shared_ptr<DjikstraVertexNode>> nodes;

		const Eigen::MatrixXd& V = patch->origin_mesh()->vertices();

		if (exclude.count(source) > 0) {
			// Why would you do that??
			return nodes;
		}

		const Eigen::SparseMatrix<int>& adj = patch->origin_mesh()->adjacency_matrix();
		auto source_node = std::make_shared<DjikstraVertexNode>(nullptr, 0.0, source);

		q.push(source_node);
		nodes.insert(std::pair<Eigen::DenseIndex, std::shared_ptr<DjikstraVertexNode>>(source, source_node));

		while (!q.empty()) {
			std::vector<std::shared_ptr<DjikstraVertexNode>> stored;

			auto u = q.top();
			q.pop();

			while (!q.empty()) {
				stored.push_back(q.top());
				q.pop();
			}

			for (Eigen::SparseMatrix<int>::InnerIterator it(adj, static_cast<int>(u->_vid)); it; ++it) {
				Eigen::DenseIndex neighbor = it.row();   // neighbor vid

				if (patch->vids().count(neighbor) == 0) {
					continue;
				}

				double dist = (V.row(neighbor).block<1, 3>(0, 0) - V.row(u->_vid).block<1, 3>(0, 0)).norm() + u->_dist;
				auto node = nodes.find(neighbor);

				if (node == nodes.cend()) {
					auto neighbor_node = std::make_shared<DjikstraVertexNode>(u, dist, neighbor);

					if (exclude.count(neighbor) <= 0) {
						q.push(neighbor_node);
						nodes.insert(std::pair<Eigen::DenseIndex, std::shared_ptr<DjikstraVertexNode>>(neighbor, neighbor_node));
					}
				}
				else {
					if (dist < node->second->_dist) {
						node->second->_dist = dist;
						node->second->_prev = u;
					}
				}
			}

			// Changing values in a priority_queue invalidates the queue, so update it
			std::priority_queue<std::shared_ptr<DjikstraVertexNode>, std::vector<std::shared_ptr<DjikstraVertexNode>>, DjikstraDist> new_q;
			for (unsigned int i = 0; i < stored.size(); ++i) {
				q.push(stored[i]);
			}
		}

		// q is now empty, and nodes contains all _vid vertices and their shortest paths from source
		return nodes;
	}

	// Nothing fancy, just a straight implementation of the most basic Floyd-Warshall algorithm for shortest path distance values
	Eigen::MatrixXd floyd_warshall(const Eigen::SparseMatrix<double> edge_weights) {
		if (edge_weights.rows() != edge_weights.cols()) {
			throw std::logic_error("Edge weights is not symmetric!");
		}

		Eigen::DenseIndex V = edge_weights.rows();

		Eigen::MatrixXd dist = Eigen::MatrixXd::Constant(V, V, std::numeric_limits<double>::infinity());

		for (Eigen::DenseIndex i = 0; i < V; ++i) {
			dist(i, i) = 0;
		}

		for (int k = 0; k < edge_weights.outerSize(); ++k) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(edge_weights, k); it; ++it) {
				dist(it.row(), it.col()) = it.value();
			}
		}

		for (Eigen::DenseIndex k = 0; k < V; ++k) {
			for (Eigen::DenseIndex i = 0; i < V; ++i) {
				for (Eigen::DenseIndex j = 0; j < V; ++j) {
					if (dist(i, j) > dist(i, k) + dist(k, j)) {
						dist(i, j) = dist(i, k) + dist(k, j);
					}
				}
			}
		}

		return dist;
	}

	struct EdgeComparator {
		bool operator()(const std::pair<Eigen::DenseIndex, Eigen::DenseIndex>& a, const std::pair<Eigen::DenseIndex, Eigen::DenseIndex>& b) const {
			// order of the face identifiers of the edge doesn't matter, only that the combination is the same
			bool res = (a.first == b.first && a.second == b.second) || (a.first == b.second && a.second == b.first);

			if (res) {
				// They're equal
				return false;
			}

			// Otherwise, order by vid (arbitrary)
			if (a.first == b.first) {
				return a.second < b.second;
			} 

			return a.first < b.first;
		}
	};

	// This will assert if the two faces are not neighbors
	Eigen::VectorXd shared_edge_midpoint(Eigen::DenseIndex fid1, Eigen::DenseIndex fid2, std::shared_ptr<Mesh> mesh) {
		const Eigen::MatrixXd& V = mesh->vertices();
		const Eigen::MatrixXi& F = mesh->faces();

		std::vector<Eigen::DenseIndex> endpoints;
		for (Eigen::DenseIndex j = 0; j < F.cols(); ++j) {
			Eigen::DenseIndex adj_vid = F(fid2, j);

			if ((F.row(fid1).array() == adj_vid).any()) {
				endpoints.push_back(adj_vid);
			}
		}

		assert(endpoints.size() == 2);

		Eigen::VectorXd midpoint = (V.row(endpoints[0]) + V.row(endpoints[1])).transpose() / 2.0;

		return midpoint;
	}

	std::vector<Eigen::DenseIndex> face_to_face(Eigen::DenseIndex source, Eigen::DenseIndex sink, std::shared_ptr<Mesh> mesh) {
		std::vector<Eigen::DenseIndex> path;

		if (mesh == nullptr) {
			return path;
		}

		std::priority_queue<std::shared_ptr<DjikstraFaceNode>, std::vector<std::shared_ptr<DjikstraFaceNode>>, DjikstraDist> q;
		std::map<std::pair<Eigen::DenseIndex, Eigen::DenseIndex>, std::shared_ptr<DjikstraFaceNode>, EdgeComparator> nodes;

		const Eigen::MatrixXd& V = mesh->vertices();
		const Eigen::MatrixXi& F = mesh->faces();
		const Eigen::MatrixXi& tri_adj = mesh->tri_adjacency_matrix();

		{ // Bootstrap source (the source is actually made up of a number of sources equal to the number of edges of the source face
			for (Eigen::DenseIndex i = 0; i < tri_adj.cols(); ++i) {
				Eigen::DenseIndex adj_fid = tri_adj(source, i);

				if (adj_fid < 0 || adj_fid >= F.rows()) {
					continue;
				}

				Eigen::VectorXd midpoint = shared_edge_midpoint(source, adj_fid, mesh);

				// Note: order here is *important* -- (from_fid, to_fid)
				auto edge = std::make_pair(source, adj_fid);
				
				auto source_node = std::make_shared<DjikstraFaceNode>(nullptr, 0.0, edge, midpoint);

				q.push(source_node);
				
				nodes.insert(std::make_pair(edge, source_node));
				nodes.insert(std::make_pair(std::make_pair(adj_fid, source), source_node));

				auto check = nodes.find(edge);

				if (check == nodes.end()) {
					throw std::logic_error("This is dead wrong!");
				}
			}
		}

		// Search
		std::shared_ptr<DjikstraFaceNode> sink_node = nullptr;

		while (!q.empty()) {
			std::vector<std::shared_ptr<DjikstraFaceNode>> stored;

			auto u = q.top();
			q.pop();

			while (!q.empty()) {
				stored.push_back(q.top());
				q.pop();
			}

			Eigen::DenseIndex cur_face = u->_edge.second;
			Eigen::DenseIndex prev_face = u->_edge.first;

			if (cur_face == sink) {
				sink_node = u;
				break;
			}

			for (Eigen::DenseIndex i = 0; i < tri_adj.cols(); ++i) {
				Eigen::DenseIndex neighbor = tri_adj(cur_face, i);   // neighbor fid

				if (neighbor == prev_face || neighbor < 0 || neighbor >= F.rows()) {
					continue;
				}

				Eigen::VectorXd midpoint = shared_edge_midpoint(cur_face, neighbor, mesh);

				auto neighbor_edge = std::make_pair(cur_face, neighbor);

				double dist = u->_dist + (u->_midpoint - midpoint).norm();

				auto node = nodes.find(neighbor_edge);

				if (node == nodes.cend()) {
					auto neighbor_node = std::make_shared<DjikstraFaceNode>(u, dist, neighbor_edge, midpoint);
	
					q.push(neighbor_node);

					nodes.insert(std::make_pair(neighbor_edge, neighbor_node));
				} else {
					if (dist < node->second->_dist) {
						node->second->_dist = dist;

						if (node->second->_prev == nullptr || node->second->_dist <= 0.0) {
							throw std::logic_error("How are we replacing the prev of a root note??");
						}

						node->second->_prev = u;
					}
				}
			}

			// Changing values in a priority_queue invalidates the queue, so update it
			//std::priority_queue<std::shared_ptr<DjikstraFaceNode>, std::vector<std::shared_ptr<DjikstraFaceNode>>, DjikstraDist> new_q;
			for (unsigned int i = 0; i < stored.size(); ++i) {
				q.push(stored[i]);
			}
		}

		// Populate shortest path from source to sink
		if (sink_node == nullptr) {
			return path;
		}

		while (true) {
			path.push_back(sink_node->_edge.second);

			if (sink_node->_prev == nullptr) {
				break;
			}

			sink_node = sink_node->_prev;
		}

		assert(sink_node->_edge.first == source);

		path.push_back(source);

		std::reverse(path.begin(), path.end());

		return path;
	}
}